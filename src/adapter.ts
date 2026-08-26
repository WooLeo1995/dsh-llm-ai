/**
 * models.dev-cataloged implementation of the Harness LLM seam's adapter
 * contract. Each registration reads the currently resolved profiles, so a
 * configuration change reaches the next request without a restart; model
 * descriptors come from the route catalogs those profiles materialized.
 *
 * The wire runtime is the harness-owned `openai-completions` dialect: one
 * `stream()` call makes exactly one provider request — a direct fetch against
 * the route's resolved `baseURL`, with the SSE response framed by
 * `eventsource-parser` and translated into harness stream chunks. Image input
 * resolves durable attachments into transient base64 data URLs at request
 * time, gated on the model's resolved modalities and the attachments seam,
 * with over-budget history offloaded to the fixed placeholder first. The
 * credential resolves once per stream call through the plugin-owned hook,
 * after the image gate and before the request. Failures
 * classify into stable codes; the configured idle timeout bounds each
 * outstanding provider read without counting consumer think time.
 *
 * @module dsh-llm-ai/adapter
 */

import {
  attributionHeaders,
  contentHasImage,
  CONTEXT_WINDOW_EXCEEDED_CODE,
  isContextWindowExceededError,
  isQuotaExceededError,
  LlmAdapter,
  LlmError,
  ProviderRequestId,
  QUOTA_EXCEEDED_CODE,
  ReasoningEffortId,
} from '@deepseek-ai/dsh-llm'
import type {
  GenerateOptions,
  LlmModelInfo,
  LlmProviderInfo,
  LlmResolvedModelInfo,
  ResolvedRetryPolicy,
  StreamChunk,
} from '@deepseek-ai/dsh-llm'
import type { AttachmentStore } from '@deepseek-ai/dsh-attachment'
import { idleWatchdog, timeoutOf } from '@deepseek-ai/dsh-timeout'
import { REASONING_LEVELS } from './catalog.ts'
import type { ReasoningLevel, ResolvedModel } from './catalog.ts'
import type { ResolvedLlmAiProfile } from './config.ts'
import { dispatchReasoning, resolveDialect, serializeRequest, serializeRequestWithImages } from './serialize.ts'
import type { ThinkingDispatch, WireDialect } from './serialize.ts'
import { parseSse } from './sse.ts'
import { translate } from './translate.ts'
import type { WireError } from './types.ts'

/** A value that may be supplied synchronously or asynchronously. */
type MaybePromise<T> = T | Promise<T>

/** Constructor options for {@link LlmAiAdapter}: the operation-local hooks the plugin owns. */
export interface LlmAiAdapterOptions {
  /** Current validated profiles by provider route; called once per operation. */
  profiles: () => ReadonlyMap<string, ResolvedLlmAiProfile>
  /**
   * Resolve the credential for one route's request; the one resolution point
   * for the credential plane, owned by the plugin because only it can see the
   * optional credentials seam. Called once per stream call after the image
   * gate, so a refusal there never resolves a credential: the plugin resolves
   * a named reference through `ctx.credentials` when one is mounted (falling
   * back to the trusted environment otherwise), format-checks every resolved
   * key, and fails a reference that resolves to nothing with
   * `MISSING_CREDENTIAL` instead of authenticating with an unrelated ambient
   * key. `undefined` — a profile naming no reference at all — sends no
   * authorization header.
   */
  resolveApiKey: (profile: ResolvedLlmAiProfile) => MaybePromise<string | undefined>
  /**
   * Resolve the durable attachment service for one request's image input.
   * Read per call, never captured at construction, so Cordis load order
   * cannot freeze optional availability: a composition that loads the
   * attachment store after this plugin still serves the next image request,
   * and one without it keeps every text-only call working. Absence rejects
   * image input while text-only requests never resolve the service.
   */
  resolveAttachments?: () => AttachmentStore | undefined
}

/** The profile for one route, or the not-owned failure. */
function profileOf(
  profiles: ReadonlyMap<string, ResolvedLlmAiProfile>,
  provider: string,
): ResolvedLlmAiProfile {
  const profile = profiles.get(provider)
  if (profile === undefined) {
    throw new LlmError(`llm-ai adapter does not own provider "${provider}"`, 'NO_ADAPTER')
  }
  return profile
}

/** The resolved descriptor for one exact route/model pair, or the unknown-model failure. */
function modelOf(profile: ResolvedLlmAiProfile, model: string): ResolvedModel {
  const resolved = profile.models.find(entry => entry.id === model)
  if (resolved === undefined) {
    throw new LlmError(`llm-ai provider "${profile.provider}" has no configured model "${model}"`, 'UNKNOWN_MODEL')
  }
  return resolved
}

/**
 * The reasoning block one resolved model reports, or nothing at all.
 *
 * A model with no offered efforts — a registry non-reasoning model, or one a
 * declaration stripped — reports no reasoning metadata, which is the seam's
 * way of saying the capability is unavailable. The route's configured default
 * level rides along only when this model offers it: describing what a model
 * can do must not fail because a deployment asked it for something it cannot.
 */
function reasoningInfo(
  model: ResolvedModel,
  defaultLevel: ReasoningLevel | undefined,
): Pick<LlmResolvedModelInfo, 'reasoning'> | Record<string, never> {
  if (model.reasoningEfforts.size === 0) return {}
  const offered = REASONING_LEVELS.filter(level => model.reasoningEfforts.has(level))
  return {
    reasoning: {
      efforts: offered.map(level => ({
        id: ReasoningEffortId(level),
        name: `${level.charAt(0).toUpperCase()}${level.slice(1)}`,
      })),
      ...defaultLevel !== undefined && model.reasoningEfforts.has(defaultLevel)
        ? { defaultEffort: ReasoningEffortId(defaultLevel) }
        : {},
    },
  }
}

/** Capability-owned code the idle watchdog stamps onto its timeout reason. */
const STREAM_IDLE_TIMEOUT_CODE = 'LLM_AI_STREAM_IDLE_TIMEOUT'

/**
 * Parse a `Retry-After` header value: delta-seconds or an HTTP-date.
 * @param value - the raw header value, or `null` when absent.
 * @returns the positive delay in milliseconds, or `undefined` when the value is absent, malformed, or past.
 */
function providerRetryAfterMs(value: string | null): number | undefined {
  if (value === null) return undefined
  if (/^\d+$/.test(value)) {
    const delay = Number(value) * 1_000
    return Number.isFinite(delay) && delay > 0 ? delay : undefined
  }
  const delay = Date.parse(value) - Date.now()
  return Number.isFinite(delay) && delay > 0 ? delay : undefined
}

/** The provider-issued request id of one response, when it carried one. */
function requestId(headers: Headers): ReturnType<typeof ProviderRequestId> | undefined {
  const value = headers.get('x-request-id')
  return value === null || value.length === 0 ? undefined : ProviderRequestId(value)
}

/**
 * Map an HTTP status to a stable LlmError code.
 * @param status - status of a non-2xx provider response.
 * @param error - parsed provider error body, when available.
 * @returns the normalized harness error code.
 */
export function httpErrorCode(status: number, error?: WireError['error']): string {
  if (status === 401 || status === 403) return 'AUTH'
  if (status === 413) return 'INVALID_REQUEST'
  const detail = [error?.code, error?.type, error?.message].filter(Boolean).join(' ')
  if (isQuotaExceededError(detail)) return QUOTA_EXCEEDED_CODE
  if (status === 429) return 'RATE_LIMIT'
  if (status === 400) {
    if (isContextWindowExceededError(detail)) return CONTEXT_WINDOW_EXCEEDED_CODE
    return 'INVALID_REQUEST'
  }
  if (status >= 500) return 'SERVER'
  return `HTTP_${status}`
}

/**
 * Multi-provider adapter over the models.dev catalog. Each operation reads
 * the current profiles; descriptors come from the route catalogs those
 * profiles materialized at resolution.
 */
export class LlmAiAdapter extends LlmAdapter {
  constructor(private readonly config: LlmAiAdapterOptions) {
    super()
  }

  override providerInfo(provider: string): LlmProviderInfo {
    // The configured name, not the route key: `displayName` exists so a
    // deployment can label a route, and a label only the configuration surface
    // reads would leave every selector showing the raw key.
    return { id: provider, name: this.config.profiles().get(provider)?.displayName ?? provider }
  }

  override providerRetryPolicy(provider: string): ResolvedRetryPolicy | undefined {
    return this.config.profiles().get(provider)?.retryPolicy
  }

  override listModels(provider: string): Promise<readonly LlmModelInfo[]> {
    return Promise.resolve().then(() => {
      const profile = profileOf(this.config.profiles(), provider)
      return profile.models.map(model => ({
        provider,
        id: model.id,
        name: model.name,
        inputModalities: [...model.input],
      }))
    })
  }

  override resolveModel(
    provider: string,
    model: string,
    _signal?: AbortSignal,
  ): Promise<LlmResolvedModelInfo> {
    return Promise.resolve().then(() => {
      const profile = profileOf(this.config.profiles(), provider)
      const resolved = modelOf(profile, model)
      // Only a cap the deployment configured is a request default; the
      // registry's `maxTokens` sizes the model and stops there.
      const configuredMaxTokens = profile.configuredMaxTokens.get(model)
      return {
        provider,
        id: model,
        name: resolved.name,
        inputModalities: [...resolved.input],
        context: { contextWindow: resolved.contextWindow },
        ...configuredMaxTokens === undefined ? {} : { defaultMaxTokens: configuredMaxTokens },
        ...reasoningInfo(resolved, profile.reasoning),
      }
    })
  }

  async * stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    // One resolution per stream call: the route profile and the model it
    // serves freeze here for this whole request, so an in-flight stream never
    // observes a configuration change and the next call re-resolves.
    const profiles = this.config.profiles()
    const profile = profileOf(profiles, options.provider)
    const model = modelOf(profile, options.model)
    // Reasoning dispatch refuses an unselectable level before any network
    // I/O, naming the route and model.
    const thinking = dispatchReasoning(profile, model, options.reasoningEffort)
    // Compat resolution runs beside it, per field: the model entry's switches
    // win over the route profile's, then the protocol defaults answer.
    const dialect = resolveDialect(model, profile.compat)
    // Image gate, still before any credential, attachment, or network I/O: a
    // model whose resolved modalities omit image refuses the input outright,
    // and an image-capable model needs the attachments seam on this call.
    // Text-only requests resolve neither the service nor the bytes.
    const hasImages = options.messages.some(message => contentHasImage(message.content))
    let attachments: AttachmentStore | undefined
    if (hasImages) {
      if (!model.input.includes('image')) {
        throw new LlmError(
          `llm-ai: provider "${profile.provider}" model "${model.id}" does not accept image input`,
          'UNSUPPORTED_CONTENT',
        )
      }
      attachments = this.config.resolveAttachments?.()
      if (attachments === undefined) {
        throw new LlmError('llm-ai: image input requires the durable attachment service', 'UNSUPPORTED_CONTENT')
      }
    }
    const apiKey = await this.config.resolveApiKey(profile)
    const consumer = new AbortController()
    const upstream = options.signal === undefined
      ? consumer.signal
      : AbortSignal.any([options.signal, consumer.signal])
    using watchdog = idleWatchdog(upstream, profile.streamIdleTimeoutMs, STREAM_IDLE_TIMEOUT_CODE)
    const iterator = this.request(
      options,
      profile,
      model,
      thinking,
      dialect,
      apiKey,
      attachments,
      watchdog.signal,
      () => { watchdog.pulse() },
    )[Symbol.asyncIterator]()
    let exhausted = false
    try {
      while (true) {
        const result = await watchdog.next(iterator)
        if (result.done) {
          exhausted = true
          return
        }
        yield result.value
      }
    } catch (error: unknown) {
      if (timeoutOf(watchdog.signal, STREAM_IDLE_TIMEOUT_CODE) !== undefined) {
        throw new LlmError(
          `llm-ai: provider "${profile.provider}" stream idle timeout after ${profile.streamIdleTimeoutMs}ms`,
          'TIMEOUT',
          { cause: error },
        )
      }
      if (options.signal?.aborted) {
        throw new LlmError(`llm-ai: provider "${profile.provider}" request aborted by caller`, 'ABORTED', { cause: error })
      }
      if (error instanceof LlmError) throw error
      throw new LlmError(
        `llm-ai: provider "${profile.provider}" stream from ${model.baseURL} failed`,
        'TRANSPORT',
        { cause: error },
      )
    } finally {
      consumer.abort('llm-ai stream consumer stopped')
      // Prompt teardown of a suspended request generator; the aborted signal
      // already owns the transport, so this return only runs generator
      // cleanup (the stream-iterator machinery swallows cancel rejections).
      if (!exhausted && iterator.return !== undefined) {
        await iterator.return()
      }
    }
  }

  /**
   * The one provider request of one stream call, as a chunk iterator: fetch,
   * status classification, and SSE translation. Transport teardown observes
   * `signal`, which fuses the caller's abort with the idle watchdog and also
   * cancels attachment reads on the image path.
   */
  private async * request(
    options: GenerateOptions,
    profile: ResolvedLlmAiProfile,
    model: ResolvedModel,
    thinking: ThinkingDispatch,
    dialect: WireDialect,
    apiKey: string | undefined,
    attachments: AttachmentStore | undefined,
    signal: AbortSignal,
    onComment: () => void,
  ): AsyncIterable<StreamChunk> {
    const url = `${model.baseURL}/chat/completions`
    const body = JSON.stringify(attachments === undefined
      ? serializeRequest(options, profile, model, thinking, dialect)
      : await serializeRequestWithImages(options, profile, model, thinking, {
        attachments,
        maxRequestImageBytes: profile.maxRequestImageBytes,
        signal,
      }, dialect))
    const headers = {
      ...profile.headers,
      ...attributionHeaders(),
      ...apiKey === undefined ? {} : { authorization: `Bearer ${apiKey}` },
      'content-type': 'application/json',
      'accept': 'text/event-stream',
    }

    let response: Response
    try {
      response = await fetch(url, { method: 'POST', headers, body, signal })
    } catch (error: unknown) {
      // The outer stream distinguishes caller cancellation and watchdog expiry.
      if (signal.aborted) throw error
      // fetch wraps every transport failure (DNS, refused connection, TLS,
      // proxy) in a bare `TypeError: fetch failed` whose actionable detail
      // lives on `cause`. Wrapping with the endpoint and chaining the cause
      // lets `errorChain` render the full diagnosis at every reporting boundary.
      throw new LlmError(`llm-ai: provider "${profile.provider}" request to ${url} failed`, 'TRANSPORT', { cause: error })
    }

    if (!response.ok) {
      let message = `llm-ai: provider "${profile.provider}" HTTP ${response.status}`
      let providerError: WireError['error']
      try {
        providerError = (await response.json() as WireError).error
      } catch (_malformedErrorBody) {
        // Only error-body JSON parsing lands here: the HTTP status still
        // identifies the failure, so a malformed gateway body must not mask it.
      }
      if (providerError?.message) message = providerError.message
      const delay = providerRetryAfterMs(response.headers.get('retry-after'))
      const id = requestId(response.headers)
      throw new LlmError(message, httpErrorCode(response.status, providerError), {
        status: response.status,
        ...delay === undefined ? {} : { providerRetryAfterMs: delay },
        ...id === undefined ? {} : { requestId: id },
      })
    }
    if (!response.body) {
      throw new LlmError(`llm-ai: provider "${profile.provider}" returned no response body`, 'EMPTY_RESPONSE')
    }

    yield* translate(parseSse(response.body, onComment))
  }
}
