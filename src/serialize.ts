/**
 * Serialize harness requests into the `openai-completions` wire dialect, and
 * dispatch reasoning efforts through the spellings a route's catalog
 * resolved. Text-only requests stay on the compact sync path; image-bearing
 * ones resolve durable attachments into ordered base64 data-URL parts, with
 * over-budget history offloaded to the fixed placeholder first.
 *
 * @module dsh-llm-ai/serialize
 */

import { contentHasImage, LlmError, offloadRequestImages } from '@deepseek-ai/dsh-llm'
import type { ContentBlock, GenerateOptions, Message } from '@deepseek-ai/dsh-llm'
import { AttachmentError } from '@deepseek-ai/dsh-attachment'
import type { AttachmentStore } from '@deepseek-ai/dsh-attachment'
import { isReasoningLevel } from './catalog.ts'
import type { CompatProfile, MaxTokensField, ResolvedModel, ThinkingFormat } from './catalog.ts'
import type { ResolvedLlmAiProfile } from './config.ts'
import type {
  WireImageContentPart,
  WireMessage,
  WireRequest,
  WireTool,
  WireUserContentPart,
} from './types.ts'

/**
 * The request-shape choices a wire dialect makes. {@link resolveDialect}
 * computes one per request — the model entry's compat over the route's over
 * {@link PROTOCOL_DEFAULT_DIALECT}, per field — and the adapter hands it to
 * both serializer entry points, {@link serializeRequest} and
 * {@link serializeRequestWithImages}, through their trailing `dialect`
 * parameter, the single seam compat activation plugs into. The thinking
 * serialization keeps its own seam: {@link serializeThinking} splits on
 * `thinkingFormat`.
 */
export interface WireDialect {
  /** Role the system slot is sent under; the protocol default is `system`. */
  systemRole: 'system' | 'developer'
  /** Output-cap field the endpoint reads; the protocol default is `max_completion_tokens`. */
  maxTokensField: MaxTokensField
  /** Reasoning-parameter dialect; the protocol default is `deepseek`. */
  thinkingFormat: ThinkingFormat
}

/** The `openai-completions` protocol defaults: modern baseline fields, DeepSeek thinking dialect. */
export const PROTOCOL_DEFAULT_DIALECT: WireDialect = {
  systemRole: 'system',
  maxTokensField: 'max_completion_tokens',
  thinkingFormat: 'deepseek',
}

/**
 * Resolve the wire dialect one request serializes under, per field: the model
 * entry's compat wins, then the route profile's, then the protocol default.
 * models.dev records no wire-dialect facts of its own, so no registry layer
 * sits between route and default to consult, and there is no spelling for
 * handing a field back short of restating its value.
 * @param model - the resolved model the request addresses.
 * @param routeCompat - the route profile's compat switches, when any.
 * @returns the dialect both serializer entry points read.
 */
export function resolveDialect(
  model: Pick<ResolvedModel, 'compat'>,
  routeCompat: CompatProfile | undefined,
): WireDialect {
  const supportsDeveloperRole = model.compat?.supportsDeveloperRole ?? routeCompat?.supportsDeveloperRole
  return {
    systemRole: supportsDeveloperRole === true ? 'developer' : 'system',
    maxTokensField: model.compat?.maxTokensField ?? routeCompat?.maxTokensField
      ?? PROTOCOL_DEFAULT_DIALECT.maxTokensField,
    thinkingFormat: model.compat?.thinkingFormat ?? routeCompat?.thinkingFormat
      ?? PROTOCOL_DEFAULT_DIALECT.thinkingFormat,
  }
}

/** One dispatched reasoning selection, already resolved to its wire facts. */
export type ThinkingDispatch =
  | { state: 'none' }
  | { state: 'disabled' }
  | { state: 'enabled'; effort: string }

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
 * Serialize one dispatch into request fields under the named reasoning
 * dialect. The DeepSeek dialect toggles with a `thinking` object beside its
 * effort; the OpenAI dialect sends the effort alone and represents a valueless
 * `off` as the parameter's absence; the OpenRouter dialect nests the effort in
 * a `reasoning` object and likewise omits it entirely for a valueless `off`.
 * @param dispatch - the resolved reasoning dispatch.
 * @param format - the reasoning dialect the resolved wire dialect named.
 * @returns the request fields for it; empty when the dialect sends nothing for the dispatch.
 */
export function serializeThinking(
  dispatch: ThinkingDispatch,
  format: ThinkingFormat,
): Pick<Partial<WireRequest>, 'thinking' | 'reasoning_effort' | 'reasoning'> {
  switch (format) {
    case 'deepseek':
      switch (dispatch.state) {
        case 'none': return {}
        case 'disabled': return { thinking: { type: 'disabled' } }
        case 'enabled': return { thinking: { type: 'enabled' }, reasoning_effort: dispatch.effort }
      }
    case 'openai':
      switch (dispatch.state) {
        case 'none': return {}
        // For most endpoints not thinking is the parameter's absence.
        case 'disabled': return {}
        case 'enabled': return { reasoning_effort: dispatch.effort }
      }
    case 'openrouter':
      switch (dispatch.state) {
        case 'none': return {}
        case 'disabled': return {}
        case 'enabled': return { reasoning: { effort: dispatch.effort } }
      }
  }
}

/**
 * The sync text path rejects image content before any text-flattening can
 * silently erase it. The adapter routes image-bearing requests to
 * {@link serializeRequestWithImages}; this guard keeps the exported text path
 * honest if it is handed images anyway.
 */
function assertTextOnly(provider: string, model: string, blocks: readonly ContentBlock[]): void {
  if (contentHasImage(blocks)) {
    throw new LlmError(
      `llm-ai: provider "${provider}" model "${model}" received image content outside the image serialization path`,
      'UNSUPPORTED_CONTENT',
    )
  }
}

/** Dependencies required only when the request contains image input. */
export interface ImageSerializationOptions {
  /** Durable resolver for canonical image references; verifies stored bytes on read. */
  attachments: AttachmentStore
  /** Positive bound on accumulated base64 image payload. */
  maxRequestImageBytes: number
  /** Cancellation shared with the provider request. */
  signal: AbortSignal
}

/** Prefix of the user message that carries tool-result images after their tool messages. */
const TOOL_RESULT_IMAGE_TEXT = 'Attached image(s) from tool result:'

/**
 * Reject image content in roles whose wire form cannot carry it. Only user
 * messages (which hold the harness's tool results) serialize images; a system
 * or assistant message bearing an image block is a vocabulary change this
 * runtime has not followed, and flattening it away would lose model-visible
 * input silently.
 */
function assertSupportedImageRoles(provider: string, model: string, messages: readonly Message[]): void {
  for (const message of messages) {
    if (message.role !== 'user' && contentHasImage(message.content)) {
      throw new LlmError(
        `llm-ai: provider "${provider}" model "${model}" cannot serialize image content in a ${message.role} message`,
        'UNSUPPORTED_CONTENT',
      )
    }
  }
}

/**
 * Resolve one durable image into its transient wire data-URL part. The
 * attachment store verifies the stored bytes against the reference; a
 * verification failure surfaces under its own stable code rather than a
 * generic serialization error.
 * @param block - the image block whose attachment to read.
 * @param attachments - durable image resolver.
 * @param signal - cancellation for the attachment read.
 * @returns the base64 image_url part.
 */
async function imagePart(
  block: Extract<ContentBlock, { type: 'image' }>,
  attachments: AttachmentStore,
  signal: AbortSignal,
): Promise<WireImageContentPart> {
  try {
    const stored = await attachments.readImage(block.attachment, signal)
    return {
      type: 'image_url',
      image_url: {
        url: `data:${stored.ref.mediaType};base64,${Buffer.from(stored.data).toString('base64')}`,
      },
    }
  } catch (error: unknown) {
    if (error instanceof AttachmentError) {
      throw new LlmError(error.message, error.code, { cause: error })
    }
    throw error
  }
}

/**
 * Convert user or nested tool-result blocks into ordered wire parts.
 * @param blocks - typed model content, possibly nesting tool-result content.
 * @param attachments - durable image resolver.
 * @param signal - cancellation for attachment reads.
 * @returns text and image parts in block order.
 */
async function contentParts(
  blocks: readonly ContentBlock[],
  attachments: AttachmentStore,
  signal: AbortSignal,
): Promise<WireUserContentPart[]> {
  const parts: WireUserContentPart[] = []
  for (const block of blocks) {
    switch (block.type) {
      case 'text':
        if (block.text.length > 0) parts.push({ type: 'text', text: block.text })
        break
      case 'image':
        parts.push(await imagePart(block, attachments, signal))
        break
      case 'tool-result':
        parts.push(...await contentParts(block.content, attachments, signal))
        break
      default:
        // Other merge-extensible blocks are not user-input wire vocabulary.
        break
    }
  }
  return parts
}

/** Keep text-only user messages on the compact string wire form. */
function userContent(parts: readonly WireUserContentPart[]): string | WireUserContentPart[] {
  const text: string[] = []
  for (const part of parts) {
    if (part.type === 'image_url') return [...parts]
    text.push(part.text)
  }
  return text.join('')
}

/** Join the text blocks of a message (used for system/assistant content). */
function flattenText(blocks: readonly ContentBlock[]): string {
  return blocks
    .filter(block => block.type === 'text')
    .map(block => block.text)
    .join('')
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
 * Serialize image-capable history after resolving durable attachments.
 * Tool-result content stays in its string-only `tool` message, and the images
 * it carried follow in one user message after the run of tool messages, so
 * the wire keeps every image in the conversation position the model saw it.
 * @param messages - transient request history after request-size offloading.
 * @param attachments - durable image resolver.
 * @param signal - cancellation for attachment reads.
 * @param dialect - the wire dialect naming the system-slot role.
 * @returns the wire messages; order preserved, tool-result images grouped into one following user message.
 */
export async function serializeMessagesWithImages(
  messages: readonly Message[],
  attachments: AttachmentStore,
  signal: AbortSignal,
  dialect: WireDialect,
): Promise<WireMessage[]> {
  const wire: WireMessage[] = []
  let pendingToolImages: WireImageContentPart[] = []
  const flushToolImages = (): void => {
    if (pendingToolImages.length === 0) return
    wire.push({
      role: 'user',
      content: [{ type: 'text', text: TOOL_RESULT_IMAGE_TEXT }, ...pendingToolImages],
    })
    pendingToolImages = []
  }

  for (const message of messages) {
    if (message.role === 'system') {
      flushToolImages()
      wire.push({ role: dialect.systemRole, content: flattenText(message.content) })
      continue
    }
    if (message.role === 'assistant') {
      flushToolImages()
      wire.push(serializeAssistant(message))
      continue
    }

    const regular = message.content.filter(block => block.type !== 'tool-result')
    const toolResults = message.content.filter((block): block is Extract<ContentBlock, { type: 'tool-result' }> => (
      block.type === 'tool-result'
    ))
    const content = userContent(await contentParts(regular, attachments, signal))
    if (content.length > 0 || toolResults.length === 0) {
      flushToolImages()
      wire.push({ role: 'user', content })
    }
    for (const result of toolResults) {
      const parts = await contentParts(result.content, attachments, signal)
      const images = parts.filter((part): part is WireImageContentPart => part.type === 'image_url')
      const text = parts.filter(part => part.type === 'text').map(part => part.text).join('')
      wire.push({
        role: 'tool',
        tool_call_id: result.toolCallId,
        content: text || (images.length > 0 ? '(see attached image)' : '(no output)'),
      })
      pendingToolImages.push(...images)
    }
  }
  flushToolImages()
  return wire
}

/** Assemble the request fields every conversion path shares once messages are built. */
function requestWithMessages(
  options: GenerateOptions,
  messages: WireMessage[],
  thinking: ThinkingDispatch,
  dialect: WireDialect,
): WireRequest {
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
    ...serializeThinking(thinking, dialect.thinkingFormat),
    ...tools !== undefined && tools.length > 0 ? { tools } : {},
    ...options.temperature !== undefined ? { temperature: options.temperature } : {},
    ...options.maxTokens === undefined ? {} : { [dialect.maxTokensField]: options.maxTokens },
  }
}

/**
 * Build the full wire request for text-only content. Always streaming
 * (`stream: true`, usage reporting on); optional fields are omitted rather
 * than sent as null, so endpoint defaults apply. Tool-choice and
 * stop-sequence mapping stay cut.
 * @param options - the harness request (model, history, system, tools, sampling).
 * @param profile - the route's resolved profile, for refusal naming.
 * @param model - the resolved model the request addresses.
 * @param thinking - the dispatched reasoning selection.
 * @param dialect - the resolved wire dialect; the protocol defaults apply when omitted.
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
  return requestWithMessages(options, messages, thinking, dialect)
}

/**
 * Build one image-capable request while keeping durable bytes out of the
 * request: over-budget history is offloaded to the fixed placeholder first —
 * a pure function of history and bound, so omitted attachments are never
 * read — and surviving images resolve through the attachment store as
 * transient data URLs.
 * @param options - the harness request (model, history, system, tools, sampling).
 * @param profile - the route's resolved profile, for refusal naming.
 * @param model - the resolved model the request addresses, for refusal naming.
 * @param thinking - the dispatched reasoning selection.
 * @param images - attachment resolver, request bound, and cancellation.
 * @param dialect - the resolved wire dialect; the protocol defaults apply when omitted.
 * @returns the fully materialized chat-completions request body.
 */
export async function serializeRequestWithImages(
  options: GenerateOptions,
  profile: ResolvedLlmAiProfile,
  model: ResolvedModel,
  thinking: ThinkingDispatch,
  images: ImageSerializationOptions,
  dialect: WireDialect = PROTOCOL_DEFAULT_DIALECT,
): Promise<WireRequest> {
  assertSupportedImageRoles(profile.provider, model.id, options.messages)
  const requestMessages = offloadRequestImages(options.messages, images.maxRequestImageBytes)
  const messages: WireMessage[] = []
  if (options.system !== undefined) {
    messages.push({ role: dialect.systemRole, content: options.system })
  }
  messages.push(...await serializeMessagesWithImages(
    requestMessages,
    images.attachments,
    images.signal,
    dialect,
  ))
  return requestWithMessages(options, messages, thinking, dialect)
}
