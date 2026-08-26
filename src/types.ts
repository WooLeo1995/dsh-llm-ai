/**
 * The `openai-completions` wire vocabulary: request, streamed chunk, usage,
 * and error shapes as generic OpenAI-compatible endpoints exchange them.
 * Types only; every value the adapter cannot trust from the wire stays
 * optional here and is validated where it is read.
 *
 * @module dsh-llm-ai/types
 */

/** Request body for `POST {baseURL}/chat/completions`. */
export interface WireRequest {
  model: string
  messages: WireMessage[]
  stream: true
  stream_options: { include_usage: true }
  /** Thinking-mode toggle of the protocol's default reasoning dialect. */
  thinking?: { type: 'enabled' | 'disabled' }
  /** Wire spelling of the selected reasoning level. */
  reasoning_effort?: string
  tools?: WireTool[]
  temperature?: number
  /** Output cap, on the field the resolved dialect names. */
  max_completion_tokens?: number
  max_tokens?: number
}

/** System-slot message: a single string of instructions. */
export interface WireSystemMessage {
  role: 'system' | 'developer'
  content: string
}

/** User-role message with string content; multimodal parts arrive with image support. */
export interface WireUserMessage {
  role: 'user'
  content: string
}

/** Tool-role message: the result of one tool call, keyed by its call id. */
export interface WireToolMessage {
  role: 'tool'
  tool_call_id: string
  content: string
}

/**
 * Assistant-role history message. `content` replays as `""` — never null —
 * because some gateways reject null outright and the turn sits durably in the
 * session log, where a null would brick every later turn of that session.
 */
export interface WireAssistantMessage {
  role: 'assistant'
  content: string
  /** Reasoning passback, present on every turn whose assistant content carried it. */
  reasoning_content?: string
  tool_calls?: WireToolCall[]
}

/** One entry of the request `messages` array, discriminated on `role`. */
export type WireMessage =
  | WireSystemMessage
  | WireUserMessage
  | WireAssistantMessage
  | WireToolMessage

/** A completed tool call replayed on an assistant history message; `arguments` is the raw JSON string. */
export interface WireToolCall {
  id: string
  type: 'function'
  function: { name: string; arguments: string }
}

/** One entry of the request `tools` array; `parameters` is a JSON Schema object. */
export interface WireTool {
  type: 'function'
  function: {
    name: string
    description: string
    parameters: Record<string, unknown>
  }
}

/** One parsed SSE `data:` payload (a chat.completion.chunk). */
export interface WireChunk {
  choices?: WireChoice[]
  /** Arrives attached to the finish chunk and/or as a trailing usage-only chunk. */
  usage?: WireUsage | null
}

/** One streamed choice (requests always ask for a single one); `finish_reason` is non-null only on its terminal chunk. */
export interface WireChoice {
  delta?: WireDelta
  finish_reason?: string | null
}

/**
 * The incremental content of one streamed choice; any subset of fields may be
 * present per chunk. Reasoning arrives under `reasoning_content` (the
 * de-facto OpenAI-compatible spelling) or `reasoning`; the translator reads
 * the first non-empty one so endpoints that emit both never duplicate.
 */
export interface WireDelta {
  role?: string
  /** Visible text. Null/empty on reasoning/tool-call chunks. */
  content?: string | null
  reasoning_content?: string | null
  reasoning?: string | null
  tool_calls?: WireToolCallDelta[]
}

/** A streamed fragment of one tool call; fragments sharing an `index` concatenate into one call. */
export interface WireToolCallDelta {
  /** Disambiguates parallel tool calls; stable across a call's deltas. */
  index: number
  /** Present on the first delta of each call only. */
  id?: string
  type?: 'function'
  function?: {
    /** Present on the first delta of each call only. */
    name?: string
    /** Argument JSON fragment (concatenate across deltas). */
    arguments?: string
  }
}

/**
 * Wire token accounting. `prompt_tokens` INCLUDES cache hits on the
 * OpenAI-compat convention; the translator subtracts them out to keep the
 * harness convention of disjoint counts.
 */
export interface WireUsage {
  prompt_tokens: number
  completion_tokens: number
  prompt_cache_hit_tokens?: number
  prompt_tokens_details?: { cached_tokens?: number }
  completion_tokens_details?: { reasoning_tokens?: number }
}

/** Non-2xx error body. */
export interface WireError {
  error?: { message?: string; type?: string; code?: string }
}
