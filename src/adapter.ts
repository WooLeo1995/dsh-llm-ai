/**
 * models.dev-cataloged implementation of the Harness LLM seam's adapter
 * contract. Each registration reads the currently resolved profiles, so a
 * configuration change reaches the next request without a restart; model
 * descriptors come from the route catalogs those profiles materialized.
 *
 * The wire runtime itself — one `stream()` call making exactly one
 * `openai-completions` request — is the next change on this package; until
 * it lands, a streaming call fails as a terminal error chunk naming the
 * route and model rather than pretending to serve.
 *
 * @module dsh-llm-ai/adapter
 */

import { LlmAdapter, LlmError, ReasoningEffortId } from '@deepseek-ai/dsh-llm'
import type {
  GenerateOptions,
  LlmModelInfo,
  LlmProviderInfo,
  LlmResolvedModelInfo,
  ResolvedRetryPolicy,
  StreamChunk,
} from '@deepseek-ai/dsh-llm'
import { REASONING_LEVELS } from './catalog.ts'
import type { ReasoningLevel, ResolvedModel } from './catalog.ts'
import type { ResolvedLlmAiProfile } from './config.ts'

/** Constructor options for {@link LlmAiAdapter}: the profile resolution hook the plugin owns. */
export interface LlmAiAdapterOptions {
  /** Current validated profiles by provider route; called once per operation. */
  profiles: () => ReadonlyMap<string, ResolvedLlmAiProfile>
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

  override async * stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    // The openai-completions wire runtime (SSE parsing, chunk translation,
    // error classification, credential resolution) is the next change on this
    // package; the skeleton refuses streaming calls rather than pretending.
    throw new LlmError(
      `llm-ai: provider "${options.provider}" model "${options.model}" cannot stream: the`
        + ' openai-completions wire runtime is not implemented yet',
      'NOT_IMPLEMENTED',
    )
  }
}
