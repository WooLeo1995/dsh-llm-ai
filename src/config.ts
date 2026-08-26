/**
 * Configuration schema and provider-profile validation for the models.dev
 * adapter. Profiles are a dict keyed by provider route, so the composition
 * base and a user-settings layer merge per provider and the route set is
 * structural.
 *
 * A route key is not required to name a registry provider. When it does, that
 * provider's endpoint and model catalog are the profile's defaults and the
 * profile overrides them field by field; when it does not, the profile is the
 * whole provider declaration. Resolution therefore ends in one materialized
 * route catalog per route, while the configuration key that made a route
 * unserviceable can still be named in the failure.
 *
 * @module dsh-llm-ai/config
 */

import z from '@deepseek-ai/schemastery'
import { credentialRef } from '@deepseek-ai/dsh-credentials'
import type { CredentialRef } from '@deepseek-ai/dsh-credentials'
import { resolveRetryPolicy, RetryPolicySchema } from '@deepseek-ai/dsh-llm'
import type { ModelModality, ResolvedRetryPolicy, RetryPolicyConfig } from '@deepseek-ai/dsh-llm'
import { MAX_TIMER_DELAY_MS } from '@deepseek-ai/dsh-timeout'
import { COMPAT_FIELDS, MODALITIES, REASONING_LEVELS, resolveRouteModels } from './catalog.ts'
import type { CompatProfile, ModelOverride, ModelProfile, ReasoningEfforts, ReasoningLevel, ResolvedModel } from './catalog.ts'
import { supportedProtocols } from './provider.ts'
import type { RegistryCatalog } from './modelsdev.ts'

/** Default maximum idle interval while an adapter stream read is outstanding. */
export const DEFAULT_STREAM_IDLE_TIMEOUT_MS = 300_000

/**
 * Default request-level bound on base64-encoded image payload. Every image in
 * history is re-encoded into every request body, so an unbounded conversation
 * eventually exceeds a provider or gateway request-size cap and the session
 * can never complete another request. The 20MiB default admits four images at
 * the attachment store's 3.5MiB raw-image default after base64 expansion and
 * reserves request capacity for system prompts, history, tools, and JSON.
 * Deployments behind stricter gateways lower it per route.
 */
export const DEFAULT_MAX_REQUEST_IMAGE_BYTES = 20 * 1024 * 1024

/** Context capacity assumed for a configured model entry neither it nor the registry sizes. */
export const DEFAULT_CONTEXT_WINDOW = 262_144

/** Output capability assumed for a configured model entry neither it nor the registry sizes. */
export const DEFAULT_MAX_TOKENS = 32_768

/**
 * Modalities assumed for a model neither its entry nor the registry declares.
 * Text is the floor the one supported protocol certainly carries, so this is
 * the absence of a declaration rather than a guess at the endpoint. Unlike an
 * entry's list, this one may not be empty — nothing sits below it to answer
 * instead.
 */
export const DEFAULT_INPUT: readonly ModelModality[] = ['text']

/** Configuration for one provider route; the `providers` dict key IS the route. */
export interface LlmAiProviderProfile {
  /** Credential reference (environment-variable name) resolved per request through `ctx.credentials`. */
  apiKeyEnv?: string
  /** Name shown by configuration surfaces; defaults to the route key. */
  displayName?: string
  /**
   * Wire protocol every model on this route speaks. The one supported value
   * is `openai-completions`; a route naming anything else is refused at load.
   */
  api?: string
  /** Endpoint for this route's models; defaults to the registry provider's base URL. */
  baseURL?: string
  /**
   * This route's model catalog. Omission serves the registry catalog for the
   * route unchanged; an explicit list replaces it, each entry defaulting its
   * unset fields from the registry model of the same id.
   */
  models?: ModelProfile[]
  /**
   * Registry-catalog customizations by model id: each entry reshapes that one
   * model with the same fields a {@link models} entry takes, while the rest
   * of the catalog keeps serving untouched. Only meaningful on a registry
   * route with no `models` list — `models` already replaces the catalog, so
   * an override beside it, on a route the registry does not describe, or
   * naming a model the registry does not list is refused rather than skipped.
   */
  modelOverrides?: Record<string, ModelOverride>
  /**
   * Wire-compatibility switches defaulting every model on this route; each
   * model's own `compat` overrides per field. Activation and resolution
   * ordering land with the wire runtime.
   */
  compat?: CompatProfile
  /**
   * Context capacity for a configured model entry that neither it nor the
   * registry sizes (default 262,144). A guess by construction, so a
   * deployment whose gateway serves smaller models corrects it here. A
   * registry model served untouched is never guessed: it carries the capacity
   * the registry states or the route refuses.
   */
  defaultContextWindow?: number
  /** Output capability for a configured model entry that neither it nor the registry sizes (default 32,768). */
  defaultMaxTokens?: number
  /**
   * Request modalities for a model that neither its entry nor the registry
   * declares (default `[text]`). A fallback like the capacities above, not an
   * override: a registry model keeps the modalities the registry records for
   * it, and this value never narrows one.
   */
  defaultInput?: ModelModality[]
  /** Provider request headers; Harness attribution wins reserved names. */
  headers?: Record<string, string>
  /** Route-default reasoning level, offered as the model's default when it supports the level. */
  reasoning?: ReasoningLevel
  /** HTTP timeout in milliseconds. */
  timeoutMs?: number
  /** Maximum provider idle time while one stream read is outstanding. */
  streamIdleTimeoutMs?: number
  /**
   * Maximum base64-encoded image payload per request. When a request's
   * accumulated images exceed it, the oldest images are replaced by text
   * placeholders until the request fits, so a long session keeps completing
   * requests instead of being rejected by a request-size cap.
   */
  maxRequestImageBytes?: number
  /** Provider-owned model-request retry policy; omission uses normal mode with five retries. */
  retryPolicy?: RetryPolicyConfig
}

/** Validated profile with its route stamped and every adapter-owned default resolved. */
export interface ResolvedLlmAiProfile {
  /** Harness route key (the configuration dict key) and models.dev provider id. */
  provider: string
  /** Resolved display name for selectors and configuration surfaces. */
  displayName: string
  /** Validated credential reference, when one is configured. */
  apiKeyEnv?: CredentialRef
  /** The wire protocol; always the one protocol this adapter serves. */
  api: string
  /** Endpoint override, when the profile stated one; otherwise the registry's. */
  baseURL?: string
  /** Deployment-declared compat switches, when any; activation lands with the wire runtime. */
  compat?: CompatProfile
  /** Provider request headers, detached. */
  headers?: Record<string, string>
  /** Route-default reasoning level, when configured. */
  reasoning?: ReasoningLevel
  /** HTTP timeout in milliseconds, when configured. */
  timeoutMs?: number
  /** Positive finite provider-idle interval after defaulting. */
  streamIdleTimeoutMs: number
  /** Positive request-level base64 image payload bound after defaulting. */
  maxRequestImageBytes: number
  /** Immutable retry policy captured with this provider route. */
  retryPolicy: ResolvedRetryPolicy
  /** The route's materialized model catalog, in configuration order. */
  models: readonly ResolvedModel[]
  /**
   * Per-request output caps this profile explicitly configured, by model id.
   * The seam materializes one only into a request that names no cap of its
   * own, so a registry capability must not appear here.
   */
  configuredMaxTokens: ReadonlyMap<string, number>
}

/** Plugin configuration: the provider routes this instance owns, plus the catalog fetch knobs. */
export interface Config {
  /**
   * Provider routes, keyed by models.dev provider id (or any route key a
   * hand-declared gateway uses). An empty (or omitted) dict is the dormant
   * settings-driven posture: the adapter mounts with no routes and registers
   * them the moment a settings section supplies profiles.
   */
  providers?: Record<string, LlmAiProviderProfile>
  /** models.dev registry endpoint override, for a self-hosted mirror or tests. */
  catalogUrl?: string
  /** Disk cache file for the registry, overriding the default under the harness home. */
  catalogCachePath?: string
}

/**
 * Keys are the offered levels, values their wire spellings. A valueless key
 * (`off:`) survives validation because schemastery passes nullable data
 * through before any member schema runs — `z.const(null)` only controls the
 * error for non-null wrong values and what a configuration UI renders. Only
 * resolution decides which levels may leave the value empty, so the
 * diagnostic can name the route and model. The assertion narrows
 * schemastery's `Dict`, which types every literal key as required; dict
 * validation checks only present keys, so the runtime value is a partial record.
 */
const reasoningEfforts = z.dict(
  z.union([z.string(), z.const(null)]),
  z.union([...REASONING_LEVELS]),
) as unknown as z<ReasoningEfforts>

const compatProfile: z<CompatProfile> = z.object({
  maxTokensField: z.union(['max_completion_tokens', 'max_tokens']),
  supportsDeveloperRole: z.boolean(),
  thinkingFormat: z.union(['openai', 'deepseek', 'openrouter']),
})

/** The fields a `models` entry and a `modelOverrides` value share; only the id's home differs. */
const modelFields = {
  name: z.string(),
  contextWindow: z.number().step(1).min(1),
  maxTokens: z.number().step(1).min(1),
  // No explicit default, unlike the route's `defaultInput`: schemastery
  // materializes `[]` for an absent array, and resolution reads that as "no
  // answer here" so the registry entry below still applies.
  input: z.array(z.union([...MODALITIES])),
  // The union, not a bare dict: schemastery materializes an absent dict as
  // `{}`, and absent must stay distinguishable — it means "inherit the
  // registry's capability", while `false` disables reasoning.
  reasoningEfforts: z.union([z.const(false), reasoningEfforts]),
  compat: compatProfile,
}

const modelProfile: z<ModelProfile> = z.object({
  id: z.string().required(),
  ...modelFields,
})

/** A {@link modelProfile} whose id lives in the `modelOverrides` dict key. */
const modelOverride: z<ModelOverride> = z.object(modelFields)

const profile = z.object({
  apiKeyEnv: z.string().role('credential-ref'),
  displayName: z.string(),
  api: z.union([...supportedProtocols()]),
  baseURL: z.string(),
  models: z.array(modelProfile),
  modelOverrides: z.dict(modelOverride),
  compat: compatProfile,
  defaultContextWindow: z.number().step(1).min(1).default(DEFAULT_CONTEXT_WINDOW),
  defaultMaxTokens: z.number().step(1).min(1).default(DEFAULT_MAX_TOKENS),
  defaultInput: z.array(z.union([...MODALITIES])).default([...DEFAULT_INPUT]),
  headers: z.dict(z.string()),
  reasoning: z.union([...REASONING_LEVELS]),
  timeoutMs: z.natural(),
  streamIdleTimeoutMs: z.number().min(Number.MIN_VALUE).max(MAX_TIMER_DELAY_MS).default(DEFAULT_STREAM_IDLE_TIMEOUT_MS),
  maxRequestImageBytes: z.number().step(1).min(1).default(DEFAULT_MAX_REQUEST_IMAGE_BYTES),
  retryPolicy: RetryPolicySchema,
})

/** Runtime schema for {@link Config}. */
export const Config: z<Config> = z.object({
  providers: z.dict(profile).default({}),
  catalogUrl: z.string(),
  catalogCachePath: z.string(),
})

/**
 * Reject a compat key the owned surface does not declare. Runs on the route
 * profile because schemastery passes unknown object keys through, so a
 * misspelled or not-yet-owned switch would otherwise ride the resolved
 * profile looking applied. Valueless keys and per-model rules land with the
 * compat activation work.
 */
function assertOwnedCompatFields(provider: string, compat: CompatProfile | undefined): void {
  for (const field of Object.keys(compat ?? {})) {
    if (!(COMPAT_FIELDS as readonly string[]).includes(field)) {
      throw new Error(
        `llm-ai: provider "${provider}" sets compat "${field}", which no wire protocol declares; the`
        + ` configurable switches are ${COMPAT_FIELDS.join(', ')}`,
      )
    }
  }
}

/**
 * Validate profiles against the loaded registry and return a detached
 * route-keyed map suitable for per-request reads. This is the one explicit
 * resolve step, so an omitted dict resolves to the empty (dormant) route set
 * here rather than through a hidden fallback, and each route's models are
 * materialized once.
 * @param providers - configured provider profiles keyed by route.
 * @param catalog - the registry catalog loaded at plugin start.
 * @returns validated profiles in configuration order.
 */
export function resolveProfiles(
  providers: Readonly<Record<string, LlmAiProviderProfile>> | undefined,
  catalog: RegistryCatalog,
): Map<string, ResolvedLlmAiProfile> {
  if (Array.isArray(providers)) {
    throw new Error('llm-ai: providers is a dict keyed by provider route, not an array of profiles')
  }
  const entries = Object.entries(providers ?? {})
  const resolved = new Map<string, ResolvedLlmAiProfile>()
  for (const [provider, source] of entries) {
    if (provider.length === 0) throw new Error('llm-ai: provider names must be non-empty')
    if (source.baseURL !== undefined && source.baseURL.length === 0) {
      throw new Error(`llm-ai: provider "${provider}" has an empty baseURL`)
    }
    if (source.displayName !== undefined && source.displayName.length === 0) {
      throw new Error(`llm-ai: provider "${provider}" has an empty displayName`)
    }
    const streamIdleTimeoutMs = source.streamIdleTimeoutMs ?? DEFAULT_STREAM_IDLE_TIMEOUT_MS
    if (!Number.isFinite(streamIdleTimeoutMs)
      || streamIdleTimeoutMs <= 0
      || streamIdleTimeoutMs > MAX_TIMER_DELAY_MS) {
      throw new Error(
        `llm-ai: provider "${provider}" streamIdleTimeoutMs must be a positive finite number no greater than ${MAX_TIMER_DELAY_MS}`,
      )
    }
    const maxRequestImageBytes = source.maxRequestImageBytes ?? DEFAULT_MAX_REQUEST_IMAGE_BYTES
    if (!Number.isInteger(maxRequestImageBytes) || maxRequestImageBytes <= 0) {
      throw new Error(`llm-ai: provider "${provider}" maxRequestImageBytes must be a positive integer`)
    }
    // Detached from the configuration object because the resolved catalog is
    // published to concurrent readers. The schema's explicit default covers
    // an absent key, so an empty list here is always one someone typed — and
    // unlike an entry's, nothing below it can answer instead.
    const defaultInput = [...source.defaultInput ?? DEFAULT_INPUT]
    if (defaultInput.length === 0) {
      throw new Error(`llm-ai: provider "${provider}" defaultInput must name at least one modality`)
    }
    // The route key, not the registry provider's own name: the directory has
    // always shown route keys, and a registry route must not silently rename
    // itself on every configuration surface just because it gained a profile.
    const displayName = source.displayName ?? provider
    assertOwnedCompatFields(provider, source.compat)
    // The schema's union refuses this on the settings path; the raw entry
    // config skips the schema, so the same refusal stands here: a route
    // naming a protocol this build cannot serve must fail where it is written.
    if (source.api !== undefined && !supportedProtocols().includes(source.api)) {
      throw new Error(
        `llm-ai: provider "${provider}" names api "${source.api}", which this adapter does not serve;`
        + ` supported protocols are ${supportedProtocols().join(', ')}`,
      )
    }
    const route = resolveRouteModels({
      provider,
      registry: catalog.provider(provider),
      ...source.baseURL === undefined ? {} : { baseURL: source.baseURL },
      ...source.models === undefined ? {} : { models: source.models },
      ...source.modelOverrides === undefined ? {} : { modelOverrides: source.modelOverrides },
      defaultInput,
      defaultContextWindow: source.defaultContextWindow ?? DEFAULT_CONTEXT_WINDOW,
      defaultMaxTokens: source.defaultMaxTokens ?? DEFAULT_MAX_TOKENS,
    })
    const { apiKeyEnv, retryPolicy, models: _models, displayName: _displayName, ...rest } = source
    resolved.set(provider, {
      ...rest,
      provider,
      displayName,
      api: 'openai-completions',
      ...apiKeyEnv === undefined ? {} : { apiKeyEnv: credentialRef(apiKeyEnv) },
      streamIdleTimeoutMs,
      maxRequestImageBytes,
      retryPolicy: resolveRetryPolicy(retryPolicy, `llm-ai: provider "${provider}" retryPolicy`),
      ...rest.headers === undefined ? {} : { headers: { ...rest.headers } },
      ...rest.compat === undefined ? {} : { compat: { ...rest.compat } },
      models: route.models,
      configuredMaxTokens: route.configuredMaxTokens,
    })
  }
  return resolved
}

/**
 * Reject a section this adapter could not serve. Registered as the settings
 * namespace's validator, so an unserviceable profile is refused where it is
 * *written* — `settings.update` answers with the offending route and model
 * named — instead of being stored and then quietly disabling every route in
 * the namespace. It stays a validator rather than a schema transform because
 * the schema is also the shape a configuration surface renders and the value
 * an absent section resolves to; wrapping it would break both.
 * @param config - the resolved section to check.
 * @param catalog - the registry catalog loaded at plugin start.
 * @throws Error naming the route and model that cannot be served.
 */
export function assertServiceable(config: Config, catalog: RegistryCatalog): void {
  resolveProfiles(config.providers, catalog)
}
