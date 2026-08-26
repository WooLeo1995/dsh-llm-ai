/**
 * Materialization of one provider route's model catalog. The models.dev
 * registry supplies defaults keyed by model id, and a profile's own model
 * entries override them field by field, so a route naming a registry provider
 * stays configuration-free while a route the registry has never heard of is
 * fully describable from `settings.yaml`.
 *
 * Every fact the adapter cannot guess is decided here rather than at request
 * time: an unserviceable route fails while its configuration is being
 * resolved, which is the earliest point that can name the offending key.
 *
 * @module dsh-llm-ai/catalog
 */

import type { ModelModality } from '@deepseek-ai/dsh-llm'
import { DEFAULT_REASONING_EFFORTS } from './provider.ts'
import { supportedProtocols } from './provider.ts'
import type { RegistryModel, RegistryProvider } from './modelsdev.ts'

/**
 * Every request modality a profile may declare. The `Record` key type is a
 * drift gate against the seam's modality vocabulary: a change there fails
 * compilation here naming the drifted key, instead of silently narrowing
 * what a profile may declare.
 */
const MODALITY_GATE: Record<ModelModality, true> = {
  text: true,
  image: true,
}

/** Every request modality a profile may declare. */
export const MODALITIES = Object.keys(MODALITY_GATE) as readonly ModelModality[]

/**
 * Every reasoning level a profile may declare, in escalation order. The
 * `Record` key type is a drift gate against the seam's effort vocabulary: a
 * change here fails compilation naming the drifted level instead of silently
 * narrowing what a profile may offer.
 */
const REASONING_LEVEL_GATE: Record<'off' | 'low' | 'medium' | 'high' | 'xhigh' | 'max', true> = {
  off: true,
  low: true,
  medium: true,
  high: true,
  xhigh: true,
  max: true,
}

/** Every reasoning level a profile may declare, in escalation order. */
export const REASONING_LEVELS = Object.keys(REASONING_LEVEL_GATE) as readonly (keyof typeof REASONING_LEVEL_GATE)[]

/** One selectable reasoning level. */
export type ReasoningLevel = keyof typeof REASONING_LEVEL_GATE

/**
 * Selectable reasoning efforts for one model: each key is a level the model
 * offers (and selectors show), and its value is the wire spelling the request
 * sends for it. `off` alone may leave its value empty — "supported, send
 * nothing" — because for most endpoints not thinking is the parameter's
 * absence; every other declared level must name a wire value. A level absent
 * from the dict is not offered.
 */
export type ReasoningEfforts = Partial<Record<ReasoningLevel, string | null>>

/**
 * Every output-cap field spelling an OpenAI-compatible endpoint may read. The
 * `Record` key type is the single authority for the schema union,
 * {@link CompatProfile}, and the wire dialect, so a spelling added here
 * reaches every surface at once.
 */
const MAX_TOKENS_FIELD_GATE: Record<'max_completion_tokens' | 'max_tokens', true> = {
  max_completion_tokens: true,
  max_tokens: true,
}

/** Every output-cap field spelling a compat profile may name, for the schema union. */
export const MAX_TOKENS_FIELDS = Object.keys(MAX_TOKENS_FIELD_GATE) as readonly (keyof typeof MAX_TOKENS_FIELD_GATE)[]

/** One output-cap field spelling an OpenAI-compatible endpoint may read. */
export type MaxTokensField = keyof typeof MAX_TOKENS_FIELD_GATE

/**
 * Every reasoning-parameter dialect a request may serialize thinking under.
 * The `Record` key type is the single authority for the schema union,
 * {@link CompatProfile}, and the wire dialect, the same role the other gates
 * in this module play.
 */
const THINKING_FORMAT_GATE: Record<'openai' | 'deepseek' | 'openrouter', true> = {
  openai: true,
  deepseek: true,
  openrouter: true,
}

/** Every reasoning dialect a compat profile may name, for the schema union. */
export const THINKING_FORMATS = Object.keys(THINKING_FORMAT_GATE) as readonly (keyof typeof THINKING_FORMAT_GATE)[]

/** One reasoning-parameter dialect a request may serialize thinking under. */
export type ThinkingFormat = keyof typeof THINKING_FORMAT_GATE

/**
 * Wire-compatibility switches for OpenAI-compatible gateways whose URLs say
 * nothing about their dialects. Resolution runs per field from a model entry
 * over its route's profile to the protocol default; models.dev records no
 * wire-dialect facts, so no registry layer sits in that chain.
 * `maxTokensField` names the output-cap field the endpoint reads,
 * `supportsDeveloperRole` whether it accepts the `developer` role for the
 * system prompt, and `thinkingFormat` the reasoning-parameter dialect it
 * expects.
 */
export interface CompatProfile {
  /** Which output-cap field the endpoint reads. */
  maxTokensField?: MaxTokensField
  /** Whether the endpoint accepts the `developer` role for the system prompt. */
  supportsDeveloperRole?: boolean
  /** Reasoning-parameter dialect the endpoint expects. */
  thinkingFormat?: ThinkingFormat
}

/** Every compat field a profile may set, for the schema union and diagnostics. */
export const COMPAT_FIELDS = ['maxTokensField', 'supportsDeveloperRole', 'thinkingFormat'] as const

/**
 * Reject a compat key the owned surface does not declare, and one written
 * with no value at all. schemastery passes unknown object keys and nullable
 * values through, so a misspelled or pi-ai-era switch (`supportsTemperature`,
 * `openRouterRouting`) and a valueless `compat:` key would otherwise ride the
 * resolved profile looking applied — the silent drop this surface refuses
 * everywhere else. The diagnostic names the site and lists the offered set,
 * so a porter of an old configuration sees what IS offered instead of hunting
 * a key that never applied.
 * @param provider - provider route key, for the diagnostic.
 * @param site - the configuration site, for the diagnostic (`route` or `model "<id>"`).
 * @param compat - the configured switches, when any.
 */
export function assertOwnedCompatFields(provider: string, site: string, compat: CompatProfile | undefined): void {
  for (const [field, value] of Object.entries(compat ?? {})) {
    // The name is judged before the value, so a misspelled key written bare
    // is refused for being that name rather than for being empty.
    if (!(COMPAT_FIELDS as readonly string[]).includes(field)) {
      invalid(provider, `${site} sets compat "${field}", which no wire protocol declares; the configurable`
        + ` switches are ${COMPAT_FIELDS.join(', ')}`)
    }
    // A valueless key (`maxTokensField:`) survives schemastery, which passes
    // nullable data through before any member schema runs. Carrying it forward
    // would write nothing over the next layer's answer, leaving the switch
    // written but not applied.
    if (value == null) {
      invalid(provider, `${site} sets compat "${field}" with no value; give it one, or remove the key to`
        + ' leave the field to the next layer in the model → route → protocol-default chain')
    }
  }
}

/** One configured model entry: an id plus the registry fields it overrides. */
export interface ModelProfile {
  /** Model id sent to the provider. */
  id: string
  /** Display name for selectors; defaults to the registry name, then the id. */
  name?: string
  /** Maximum combined request and response context in tokens. */
  contextWindow?: number
  /**
   * Maximum output tokens. Configuring one also makes it this model's
   * per-request default; a value inherited from the registry, or the route's
   * fallback, is the model's capability and never becomes a request default
   * on its own.
   */
  maxTokens?: number
  /**
   * Request modalities this model accepts. Absent — or empty, which describes
   * a model that accepts nothing and so states no answer either — keeps the
   * registry entry's modalities, then the route's `defaultInput`.
   */
  input?: ModelModality[]
  /**
   * Selectable reasoning efforts. Absent inherits the registry entry's
   * capability (a hand-declared model has none and does not reason); `false`
   * declares a non-reasoning model, which is how a profile strips reasoning
   * from a registry model its gateway cannot serve; a non-empty dict declares
   * the offered levels and their wire spellings.
   */
  reasoningEfforts?: false | ReasoningEfforts
  /** Wire-compatibility switches for this model, winning over the route's per field. */
  compat?: CompatProfile
}

/**
 * Customization of one registry model, keyed by its id in the route's
 * `modelOverrides` dict — the same fields a `models` entry may set, with the
 * id living in the key. Unlike a `models` list, overrides leave the rest of
 * the catalog serving untouched, which is what makes "correct one model,
 * keep the other thirty-seven" a three-line edit.
 */
export type ModelOverride = Omit<ModelProfile, 'id'>

/** One materialized model, as the adapter serves it. */
export interface ResolvedModel {
  /** Model id sent to the provider. */
  id: string
  /** Display name for selectors. */
  name: string
  /** Provider route key the model serves. */
  provider: string
  /** The wire protocol; always the one protocol this adapter serves. */
  api: string
  /** Endpoint the model's requests go to. */
  baseURL: string
  /** Request modalities the model accepts. */
  input: readonly ModelModality[]
  /** Maximum combined request and response context in tokens. */
  contextWindow: number
  /** Maximum output tokens, when some layer stated one. */
  maxTokens: number | undefined
  /**
   * Selectable reasoning efforts by level; empty means the model does not
   * reason.
   */
  reasoningEfforts: ReadonlyMap<ReasoningLevel, string | null>
  /**
   * Wire-compatibility switches the model's own entry declared, when any.
   * Each field wins over the route profile's; unset fields fall through to
   * the route, then the protocol default.
   */
  compat?: CompatProfile
}

/** The route-level facts model materialization reads. */
export interface RouteCatalogRequest {
  /** Provider route key, stamped onto every materialized model. */
  provider: string
  /** The registry's entry for this route, when it describes one. */
  registry: RegistryProvider | undefined
  /** Endpoint override; absent defers to the registry provider's base URL. */
  baseURL?: string
  /** Configured catalog; absent means the whole registry catalog for this route. */
  models?: readonly ModelProfile[]
  /** Registry-catalog customizations by model id; only meaningful while `models` is absent. */
  modelOverrides?: Readonly<Record<string, ModelOverride>>
  /** Context capacity for a configured entry that states none and the registry cannot size. */
  defaultContextWindow: number
  /** Output capability for a configured entry that states none and the registry cannot size. */
  defaultMaxTokens: number
  /** Modalities for a model neither the entry nor the registry declares. */
  defaultInput: readonly ModelModality[]
}

/** One route's materialized catalog, plus the request caps its profile chose. */
export interface RouteCatalog {
  /** The materialized models in configuration order. */
  models: readonly ResolvedModel[]
  /**
   * Per-request output caps this profile explicitly configured, by model id.
   * Separate from {@link ResolvedModel.maxTokens} because the two answer
   * different questions: the model field is the output *capability* some
   * layer stated, while the seam's `defaultMaxTokens` is a cap the deployment
   * chose to send on requests that name none. Materializing a capability as a
   * request default would start capping every request at a number nobody
   * picked, so only an explicit configuration lands here.
   */
  configuredMaxTokens: ReadonlyMap<string, number>
}

/** Report a route the deployment cannot serve, naming the settings key at fault. */
function invalid(provider: string, detail: string): never {
  throw new Error(`llm-ai: provider "${provider}" ${detail}`)
}

/**
 * One entry's modality list, or `undefined` when it states no answer. Absent
 * and empty mean the same thing — `[]` describes a model that accepts nothing
 * and could serve no request — which is what makes an entry naming a registry
 * model without declaring modalities keep the registry's.
 */
function declaredInput(configured: readonly ModelModality[] | undefined): readonly ModelModality[] | undefined {
  return configured === undefined || configured.length === 0 ? undefined : [...configured]
}

/**
 * Resolve one model's reasoning efforts from its declaration.
 *
 * An absent declaration inherits the registry entry's capability — `true`
 * gets the protocol's default level set, `false` gets none — because a bare
 * capability flag with no spellings is exactly what the default set is for.
 * A declared dict pins every level explicitly: declared levels carry their
 * wire spelling and undeclared canonical levels are simply not offered. A
 * declared `off` with no value stays in the map as `null` ("supported, send
 * nothing") while `off` with a value sends that value.
 */
function resolveModelReasoning(
  provider: string,
  entry: Pick<ModelProfile, 'id' | 'reasoningEfforts'>,
  base: RegistryModel | undefined,
): ReadonlyMap<ReasoningLevel, string | null> {
  const efforts = entry.reasoningEfforts
  if (efforts === undefined) {
    if (base?.reasoning !== true) return new Map()
    return new Map(Object.entries(DEFAULT_REASONING_EFFORTS) as [ReasoningLevel, string | null][])
  }
  if (efforts === false) return new Map()
  // A YAML `reasoningEfforts:` left valueless arrives as null through the
  // schema union — outside the field's declared type, hence the widening —
  // while an explicit `{}` arrives as an empty dict. Both declare nothing,
  // and neither is a spelling of "inherit" or "disable".
  if ((efforts as unknown) === null || Object.keys(efforts).length === 0) {
    invalid(provider, `model "${entry.id}" has an empty reasoningEfforts; declare the offered levels, set`
      + ' false for a non-reasoning model, or omit the field to keep the registry\'s capability')
  }
  const declared = REASONING_LEVELS.flatMap((level) => {
    const wire = efforts[level]
    return wire === undefined ? [] : [[level, wire] as const]
  })
  for (const [level, wire] of declared) {
    if (wire === null) {
      if (level !== 'off') {
        invalid(provider, `model "${entry.id}" reasoningEfforts.${level} needs the wire value the request`
          + ' should send; only "off" may leave it empty')
      }
    } else if (wire.length === 0) {
      invalid(provider, `model "${entry.id}" reasoningEfforts.${level} must not be an empty string`)
    }
  }
  if (!declared.some(([level]) => level !== 'off')) {
    invalid(provider, `model "${entry.id}" reasoningEfforts offers no level beyond "off"; declare a`
      + ' reasoning level, or set reasoningEfforts to false for a non-reasoning model')
  }
  return new Map(declared)
}

/**
 * Materialize one route's catalog by merging the registry defaults under the
 * configured entries. A route with no configured `models` serves the registry
 * catalog for its provider unchanged (with `modelOverrides` reshaping
 * individual entries), which is what keeps an existing
 * `providers: { deepseek: { apiKeyEnv: … } }` profile working untouched.
 * @param request - the route-level catalog facts.
 * @returns the materialized models and the explicitly configured request caps.
 */
export function resolveRouteModels(request: RouteCatalogRequest): RouteCatalog {
  const { provider } = request
  const registry = request.registry
  if (registry?.withheldProtocol !== undefined) {
    invalid(provider, `needs wire protocol "${registry.withheldProtocol}", which this adapter does not serve;`
      + ` supported protocols are ${supportedProtocols().join(', ')}`)
  }
  const defaults = registry?.models ?? new Map<string, RegistryModel>()
  // An absent `models` key and an empty one are the same request: the config
  // schema materializes `[]` for the absent case, and an empty catalog could
  // serve no request anyway, so both mean "serve the registry catalog".
  const configured = request.models ?? []
  const overrides = request.modelOverrides ?? {}
  // Every miss is refused, never skipped: an override that lands nowhere is a
  // typo someone would otherwise hunt for in a silently unchanged model.
  for (const [id, override] of Object.entries(overrides)) {
    if (id.length === 0) invalid(provider, 'has a modelOverrides entry with an empty model id')
    if (defaults.size === 0) {
      invalid(provider, `sets modelOverrides for "${id}", but the registry does not describe this route;`
        + ' a declared route spells every model out in its models list')
    }
    if (configured.length > 0) {
      invalid(provider, `sets modelOverrides for "${id}" beside a models list; models already replaces the`
        + ' served catalog, so declare the fields on its entries')
    }
    if (!defaults.has(id)) {
      invalid(provider, `modelOverrides names "${id}", which the registry does not describe`)
    }
    // The id lives in the dict key; a value carrying its own would quietly
    // rename the model it meant to customize. The static shape already omits
    // it — this guards the schema boundary, which passes unknown keys through.
    if ('id' in override) {
      invalid(provider, `modelOverrides entry "${id}" sets "id", which is the dict key`)
    }
  }
  // An override becomes the registry entry's configuration, so everything a
  // models entry may declare — capacities, efforts — resolves through the
  // same path with the same diagnostics and request-default semantics.
  const entries: readonly ModelProfile[] = configured.length > 0
    ? configured
    : [...defaults.values()].map(model => ({ id: model.id, ...overrides[model.id] }))
  if (entries.length === 0) {
    invalid(provider, 'resolves no models; the registry does not describe this route, so its models must'
      + ' be listed in configuration')
  }
  const baseURL = request.baseURL ?? registry?.baseUrl
  const seen = new Set<string>()
  const configuredMaxTokens = new Map<string, number>()
  const models = entries.map((entry) => {
    if (entry.id.length === 0) invalid(provider, 'has a model with an empty id')
    if (seen.has(entry.id)) invalid(provider, `lists model "${entry.id}" more than once`)
    seen.add(entry.id)
    assertOwnedCompatFields(provider, `model "${entry.id}"`, entry.compat)
    const base = defaults.get(entry.id)
    if (baseURL === undefined) {
      invalid(provider, `model "${entry.id}" needs a baseURL; the registry does not describe this route's`
        + ' endpoint, so set the route\'s baseURL')
    }
    // Capacities fall back to the route's own defaults only for an entry the
    // configuration named: a registry model served untouched must carry the
    // capacity the registry states, and a context window nobody states is
    // refused rather than guessed — a fabricated window would let a request
    // fill a context the endpoint does not have. Numeric ranges are the
    // schema's to enforce on every path that reaches here.
    const configuredEntry = configured.length > 0 || Object.hasOwn(overrides, entry.id)
    const contextWindow = entry.contextWindow ?? base?.contextWindow
      ?? (configuredEntry ? request.defaultContextWindow : undefined)
    if (contextWindow === undefined) {
      invalid(provider, `model "${entry.id}" has no context window in the models.dev registry; declare one`
        + ' on the route\'s models entry or modelOverrides for it')
    }
    const maxTokens = entry.maxTokens ?? base?.maxTokens
      ?? (configuredEntry ? request.defaultMaxTokens : undefined)
    // Only a value the profile named is a deployment choice; the registry's is
    // the model's capability and stays out of request defaults.
    if (entry.maxTokens !== undefined) configuredMaxTokens.set(entry.id, entry.maxTokens)
    const model: ResolvedModel = {
      id: entry.id,
      name: entry.name ?? base?.name ?? entry.id,
      provider,
      api: 'openai-completions',
      baseURL,
      input: declaredInput(entry.input) ?? base?.input ?? [...request.defaultInput],
      contextWindow,
      maxTokens,
      reasoningEfforts: resolveModelReasoning(provider, entry, base),
      ...entry.compat === undefined ? {} : { compat: { ...entry.compat } },
    }
    return model
  })
  return { models, configuredMaxTokens }
}
