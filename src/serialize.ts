/**
 * Serialize harness requests into the `openai-completions` wire dialect, and
 * dispatch reasoning efforts through the spellings a route's catalog
 * resolved. Text-only for now: image input and request-size offload are the
 * image ticket, and a request carrying image content is refused here rather
 * than silently flattened.
 *
 * @module dsh-llm-ai/serialize
 */

import { contentHasImage, LlmError } from '@deepseek-ai/dsh-llm'
import type { ContentBlock, GenerateOptions, Message } from '@deepseek-ai/dsh-llm'
import { REASONING_LEVELS } from './catalog.ts'
import type { ReasoningLevel, ResolvedModel } from './catalog.ts'
import type { ResolvedLlmAiProfile } from './config.ts'
import type { WireMessage, WireRequest, WireTool } from './types.ts'

/**
 * The request-shape choices a wire dialect makes. The protocol default is
 * fixed today; the compat-switch activation work resolves a dialect per
 * request (model → route → registry inference → protocol default) and hands
 * it to {@link serializeRequest} through the `dialect` parameter, which is
 * the single seam it plugs into. The thinking serialization keeps its own
 * seam: {@link serializeThinking} owns the protocol-default spelling and
 * grows the `thinkingFormat` split there.
 */
export interface WireDialect {
  /** Role the system slot is sent under; the protocol default is `system`. */
  systemRole: 'system' | 'developer'
  /** Output-cap field the endpoint reads; the modern baseline is `max_completion_tokens`. */
  maxTokensField: 'max_completion_tokens' | 'max_tokens'
}

/** The `openai-completions` protocol defaults, used until compat activation resolves a dialect. */
export const PROTOCOL_DEFAULT_DIALECT: WireDialect = {
  systemRole: 'system',
  maxTokensField: 'max_completion_tokens',
}

/** One dispatched reasoning selection, already resolved to its wire facts. */
export type ThinkingDispatch =
  | { state: 'none' }
  | { state: 'disabled' }
  | { state: 'enabled'; effort: string }

/**
 * Whether a requested effort is one of the canonical levels at all. A value
 * outside the vocabulary can never be declared, so it fails dispatch like an
 * undeclared level instead of leaking to the wire.
 * @param value - the requested effort id.
 * @returns true when the value names a canonical reasoning level.
 */
function isReasoningLevel(value: string): value is ReasoningLevel {
  return (REASONING_LEVELS as readonly string[]).includes(value)
}

/**
 * Dispatch one request's reasoning effort through a model's resolved
 * spellings. A level the model does not offer — undeclared, stripped, or
 * outside the canonical vocabulary — fails here, before any network I/O,
 * naming the route, model, and level. A request naming no effort uses the
 * profile's route default when the model offers it and otherwise dispatches
 * nothing.
 * @param profile - the route's resolved profile, for the default level and refusal naming.
 * @param model - the resolved model whose effort map decides selectability.
 * @param requested - the request's selected level, when it named one.
 * @returns the wire dispatch; `disabled` is the valueless `off` spelling, `none` sends no reasoning parameter.
 */
export function dispatchReasoning(
  profile: ResolvedLlmAiProfile,
  model: ResolvedModel,
  requested: string | undefined,
): ThinkingDispatch {
  const level = requested === undefined ? profile.reasoning : requested
  if (level === undefined) return { state: 'none' }
  const wire = isReasoningLevel(level) ? model.reasoningEfforts.get(level) : undefined
  if (wire === undefined) {
    throw new LlmError(
      `llm-ai: provider "${profile.provider}" model "${model.id}" does not offer reasoning effort "${level}"`,
      'UNSUPPORTED_REASONING_EFFORT',
    )
  }
  // A valueless declaration is legal for `off` alone: "supported, send the
  // disabled spelling". Every valued level — `off` included — sends its value.
  return wire === null
    ? { state: 'disabled' }
    : { state: 'enabled', effort: wire }
}

/**
 * Serialize one dispatch into request fields under the protocol's default
 * reasoning dialect: a valued level sends the enabled spelling beside its
 * effort, the valueless `off` sends only the disabled spelling, and nothing
 * dispatched sends nothing at all.
 * @param dispatch - the resolved reasoning dispatch.
 * @returns the request fields for it; empty when no reasoning was dispatched.
 */
export function serializeThinking(dispatch: ThinkingDispatch): Pick<Partial<WireRequest>, 'thinking' | 'reasoning_effort'> {
  switch (dispatch.state) {
    case 'none': return {}
    case 'disabled': return { thinking: { type: 'disabled' } }
    case 'enabled': return { thinking: { type: 'enabled' }, reasoning_effort: dispatch.effort }
  }
}

/** Join the text blocks of a message (used for user/tool-result content). */
function flattenText(blocks: readonly ContentBlock[]): string {
  return blocks
    .filter(block => block.type === 'text')
    .map(block => block.text)
    .join('')
}

/** Reject core image content before any text-flattening path can silently erase it. */
function assertTextOnly(provider: string, model: string, blocks: readonly ContentBlock[]): void {
  if (contentHasImage(blocks)) {
    throw new LlmError(
      `llm-ai: provider "${provider}" model "${model}" received image content, which the`
        + ' openai-completions runtime does not serialize yet',
      'UNSUPPORTED_CONTENT',
    )
  }
}

/** Serialize one assistant message (text + reasoning + tool calls). */
function serializeAssistant(message: Message): WireMessage {
  const text = flattenText(message.content)
  const reasoning = message.content
    .filter(block => block.type === 'reasoning')
    .map(block => block.text)
    .join('')
  const toolCalls = message.content
    .filter(block => block.type === 'tool-call')
    .map(block => ({
      id: block.id,
      type: 'function' as const,
      function: { name: block.name, arguments: block.arguments },
    }))

  return {
    role: 'assistant',
    content: text,
    ...reasoning.length > 0 ? { reasoning_content: reasoning } : {},
    ...toolCalls.length > 0 ? { tool_calls: toolCalls } : {},
  }
}

/**
 * Serialize the conversation. `tool-result` blocks become standalone
 * `{role: 'tool'}` messages; the harness puts each tool result in its own
 * user-role message, so a mixed user message contributes its text first and
 * its tool results as separate wire messages after.
 * @param profile - the route's resolved profile, for refusal naming.
 * @param model - the resolved model, for refusal naming.
 * @param messages - the harness conversation, in order.
 * @param dialect - the wire dialect naming the system-slot role.
 * @returns the wire messages; order preserved, each tool result expanded into its own entry.
 */
export function serializeMessages(
  profile: ResolvedLlmAiProfile,
  model: ResolvedModel,
  messages: readonly Message[],
  dialect: WireDialect,
): WireMessage[] {
  const wire: WireMessage[] = []
  for (const message of messages) {
    assertTextOnly(profile.provider, model.id, message.content)
    if (message.role === 'system') {
      wire.push({ role: dialect.systemRole, content: flattenText(message.content) })
      continue
    }
    if (message.role === 'assistant') {
      wire.push(serializeAssistant(message))
      continue
    }
    // user role: tool results ride in user messages in the harness
    // vocabulary, but the wire wants them as role:'tool' messages.
    const toolResults = message.content.filter(block => block.type === 'tool-result')
    const text = flattenText(message.content)
    if (text.length > 0 || toolResults.length === 0) {
      wire.push({ role: 'user', content: text })
    }
    for (const result of toolResults) {
      wire.push({
        role: 'tool',
        tool_call_id: result.toolCallId,
        // Empty tool output still needs SOME content on the wire.
        content: flattenText(result.content) || '(no output)',
      })
    }
  }
  return wire
}

/**
 * Build the full wire request. Always streaming (`stream: true`, usage
 * reporting on); optional fields are omitted rather than sent as null, so
 * endpoint defaults apply. Tool-choice and stop-sequence mapping stay cut.
 * @param options - the harness request (model, history, system, tools, sampling).
 * @param profile - the route's resolved profile, for refusal naming.
 * @param model - the resolved model the request addresses.
 * @param thinking - the dispatched reasoning selection.
 * @param dialect - the wire dialect; defaults to the protocol defaults until compat activation resolves one.
 * @returns the chat-completions request body.
 */
export function serializeRequest(
  options: GenerateOptions,
  profile: ResolvedLlmAiProfile,
  model: ResolvedModel,
  thinking: ThinkingDispatch,
  dialect: WireDialect = PROTOCOL_DEFAULT_DIALECT,
): WireRequest {
  const messages: WireMessage[] = []
  if (options.system !== undefined) {
    messages.push({ role: dialect.systemRole, content: options.system })
  }
  messages.push(...serializeMessages(profile, model, options.messages, dialect))
  const tools: WireTool[] | undefined = options.tools?.map(tool => ({
    type: 'function',
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters,
    },
  }))
  return {
    model: options.model,
    messages,
    stream: true,
    stream_options: { include_usage: true },
    ...serializeThinking(thinking),
    ...tools !== undefined && tools.length > 0 ? { tools } : {},
    ...options.temperature !== undefined ? { temperature: options.temperature } : {},
    ...options.maxTokens === undefined ? {} : { [dialect.maxTokensField]: options.maxTokens },
  }
}
