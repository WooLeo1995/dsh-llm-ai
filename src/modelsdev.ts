/**
 * models.dev catalog loader: the single provider/model metadata source this
 * adapter reads. The community-maintained [models.dev](https://models.dev)
 * registry (`api.json`) supplies provider base URLs and display names plus
 * per-model context windows, output caps, input modalities, reasoning
 * capability, and the selectable reasoning levels `reasoning_options`
 * states — everything a route needs except the credential, which the
 * profile names by reference.
 *
 * The registry is fetched once at plugin load and cached to disk under the
 * harness home, so a restarted harness serves the last good snapshot when the
 * network is unavailable. It is not live: an update to models.dev reaches the
 * next plugin load, not the running process. A fetch failure is loud only
 * when nothing is cached — the deployment asked for this catalog, and serving
 * nothing would be a silent degradation.
 *
 * @module dsh-llm-ai/modelsdev
 */

import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { dirname } from 'node:path'
import { dshHomePath } from '@deepseek-ai/dsh-home-paths'
import type { ModelModality } from '@deepseek-ai/dsh-llm'
import { isReasoningLevel } from './catalog.ts'
import type { ReasoningLevel } from './catalog.ts'
import { withheldProtocol } from './provider.ts'

/** Default registry endpoint. */
export const DEFAULT_MODELS_DEV_URL = 'https://models.dev/api.json'

/** Default disk cache location under the harness home.
 * @returns the cache file path under `$DSH_HOME/storages`.
 */
export function defaultCachePath(): string {
  return dshHomePath('storages', 'models-dev-cache.json')
}

/** One provider entry of `api.json` (the fields this loader reads). */
export interface ModelsDevProvider {
  id?: unknown
  name?: unknown
  /** API-key environment variable names; presence means the provider takes a key. */
  env?: unknown
  /** Base URL for the provider's API. */
  api?: unknown
  /** Models served by this provider, keyed by model id. */
  models?: Record<string, ModelsDevModel>
}

/** One model entry of `api.json` (the fields this loader reads). */
export interface ModelsDevModel {
  name?: unknown
  reasoning?: unknown
  /** Structured reasoning selectors the registry states, when it states any. */
  reasoning_options?: unknown
  modalities?: { input?: unknown; output?: unknown }
  limit?: { context?: unknown; output?: unknown; input?: unknown }
}

/** Loader options; every field is injectable so tests avoid the network. */
export interface ModelsDevOptions {
  /** Registry endpoint; defaults to {@link DEFAULT_MODELS_DEV_URL}. */
  url?: string
  /** Disk cache file; defaults to {@link defaultCachePath}. */
  cachePath?: string
  /** Fetch implementation, for tests and offline mirrors. */
  fetchImpl?: typeof fetch
}

/**
 * One registry model, with the fields this adapter serves already detached
 * from the raw entry. `contextWindow` and `maxTokens` stay `undefined` when
 * the registry states no capacity: the context window is a refusal at
 * resolution (a model that cannot be sized cannot be served honestly), while
 * a missing output cap is an absent fact the adapter reports as absent.
 */
export interface RegistryModel {
  /** Model id sent to the provider. */
  id: string
  /** Display name from the registry, or the id when it states none. */
  name: string
  /** Whether the registry marks the model as reasoning-capable. */
  reasoning: boolean
  /**
   * The selectable reasoning levels the registry's `reasoning_options`
   * states, mapped to canonical level names keyed to their wire spellings
   * (each regular level spells as itself; the registry's `none` becomes a
   * valueless `off`). Absent when the entry states no servable set — a
   * `toggle`, empty or absent options, or values that all fall outside the
   * canonical vocabulary — leaving the protocol's default set to answer.
   */
  reasoningLevels?: ReadonlyMap<ReasoningLevel, string | null>
  /** Request modalities the registry records, floored at text. */
  input: readonly ModelModality[]
  /** Maximum combined request and response context, when the registry states one. */
  contextWindow?: number
  /** Maximum output tokens, when the registry states a cap. */
  maxTokens?: number
}

/** One registry provider, precomputed for the adapter's synchronous reads. */
export interface RegistryProvider {
  /** Provider id; also the route key. */
  id: string
  /** Registry display name, or the id when it states none. */
  displayName: string
  /** Base URL the registry records, when one exists. */
  baseUrl?: string
  /**
   * The wire protocol this provider needs that the adapter cannot serve, when
   * it is one of the withheld families. The id stays in the configurable
   * provider directory; route registration refuses naming the protocol.
   */
  withheldProtocol?: string
  /** Registry models by id; the empty-id entry is never mapped. */
  models: ReadonlyMap<string, RegistryModel>
}

/** The parsed registry, answering provider facts by route key. */
export interface RegistryCatalog {
  /** Every provider entry in the snapshot, in registry order. */
  providers(): readonly RegistryProvider[]
  /** One provider's facts, when the snapshot describes it. */
  provider(provider: string): RegistryProvider | undefined
}

/** A positive integer field, or `undefined` when absent or unusable. */
function capacity(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isInteger(value) && value > 0 ? value : undefined
}

/** A non-empty string field, or `undefined`. */
function label(value: unknown): string | undefined {
  return typeof value === 'string' && value.length > 0 ? value : undefined
}

/**
 * The reasoning level set one registry entry's `reasoning_options` states.
 *
 * Only an `effort` entry carrying canonical level names yields a set; each
 * regular level spells as itself on the wire, while the registry's `none`
 * becomes a valueless `off` (selectable off; the wire sends the dialect's
 * disabled spelling). Every other shape — `toggle`, an empty or absent array,
 * a non-array field, or values that all fall outside the canonical
 * vocabulary — states no set, so `undefined` hands the protocol's default
 * back to resolution. Filtering is never fatal: a registry entry carrying
 * unknown values must not fail the route or the directory.
 * @param raw - the entry's `reasoning_options` field, as the registry wrote it.
 * @returns the mapped level set, or `undefined` when the entry states none.
 */
function reasoningOptions(raw: unknown): ReadonlyMap<ReasoningLevel, string | null> | undefined {
  if (!Array.isArray(raw)) return undefined
  const effort = raw.find(
    (entry): entry is { values?: unknown } =>
      typeof entry === 'object' && entry !== null && (entry as { type?: unknown }).type === 'effort',
  )
  if (effort === undefined) return undefined
  const levels = new Map<ReasoningLevel, string | null>()
  for (const value of Array.isArray(effort.values) ? effort.values : []) {
    if (value === 'none') levels.set('off', null)
    else if (typeof value === 'string' && isReasoningLevel(value)) levels.set(value, value)
  }
  return levels.size > 0 ? levels : undefined
}

/** Map one models.dev model entry to a {@link RegistryModel}. */
function toRegistryModel(id: string, raw: ModelsDevModel): RegistryModel {
  const input = raw.modalities?.input
  const hasImage = Array.isArray(input) && input.includes('image')
  const contextWindow = capacity(raw.limit?.context)
  const maxTokens = capacity(raw.limit?.output)
  const reasoningLevels = reasoningOptions(raw.reasoning_options)
  return {
    id,
    name: label(raw.name) ?? id,
    reasoning: raw.reasoning === true,
    input: hasImage ? ['text', 'image'] : ['text'],
    ...contextWindow === undefined ? {} : { contextWindow },
    ...maxTokens === undefined ? {} : { maxTokens },
    ...reasoningLevels === undefined ? {} : { reasoningLevels },
  }
}

/**
 * Precompute the adapter-facing catalog for one models.dev snapshot.
 *
 * Mapping is faithful: models are mapped for every provider, including the
 * ones with no base URL (the registry's native-SDK entries — openai, groq,
 * and friends carry no `api` value) and the withheld protocol families.
 * Whether a route can serve them is policy, decided at resolution where the
 * diagnostic can name the missing endpoint or protocol: a provider without a
 * base URL still serves its models over a route that declares `baseURL`.
 * @param providers - parsed `api.json` provider entries keyed by provider id.
 * @returns the registry catalog.
 */
export function catalogFromSnapshot(
  providers: Readonly<Record<string, ModelsDevProvider>>,
): RegistryCatalog {
  const byId = new Map<string, RegistryProvider>()
  for (const id of Object.keys(providers)) {
    const raw = providers[id]
    const baseUrl = label(raw?.api)
    const withheld = withheldProtocol(id)
    const models = new Map<string, RegistryModel>()
    for (const [modelId, model] of Object.entries(raw?.models ?? {})) {
      if (modelId.length === 0) continue
      models.set(modelId, toRegistryModel(modelId, model))
    }
    byId.set(id, {
      id,
      displayName: label(raw?.name) ?? id,
      ...baseUrl === undefined ? {} : { baseUrl },
      ...withheld === undefined ? {} : { withheldProtocol: withheld },
      models,
    })
  }
  return {
    providers: () => [...byId.values()],
    provider: (provider) => byId.get(provider),
  }
}

/**
 * Load the models.dev registry: a fresh fetch, falling back to the disk cache
 * when the network fails.
 *
 * The cache is read first so an offline deployment keeps working, but a
 * successful fetch still wins — the freshness trade is that a restart after a
 * long offline period serves the last snapshot, not the live registry.
 * @param options - loader options.
 * @returns the parsed provider entries keyed by provider id.
 * @throws Error when the fetch fails and no cache exists.
 */
export async function loadModelsDevRegistry(options: ModelsDevOptions = {}): Promise<
  Record<string, ModelsDevProvider>
> {
  const url = options.url ?? DEFAULT_MODELS_DEV_URL
  const cachePath = options.cachePath ?? defaultCachePath()
  const fetchImpl = options.fetchImpl ?? fetch
  let cached: Record<string, ModelsDevProvider> | undefined
  try {
    cached = JSON.parse(await readFile(cachePath, 'utf8'))
  } catch {
    // Absent or unreadable cache; fall through to the network.
  }
  if (cached !== undefined && !isRegistry(cached)) cached = undefined
  try {
    const response = await fetchImpl(url, { headers: { accept: 'application/json' } })
    if (!response.ok) throw new Error(`models.dev answered ${response.status}`)
    const parsed: unknown = await response.json()
    if (!isRegistry(parsed)) {
      throw new Error('models.dev api.json did not answer a provider-keyed registry')
    }
    await writeCache(cachePath, parsed)
    return parsed
  } catch (error) {
    if (cached !== undefined) return cached
    throw new Error(`llm-ai: could not load the models.dev catalog from ${url}: ${message(error)}`)
  }
}

/** Whether a parsed value looks like a provider-keyed registry. */
function isRegistry(value: unknown): value is Record<string, ModelsDevProvider> {
  if (typeof value !== 'object' || value === null) return false
  for (const entry of Object.values(value)) {
    if (typeof entry === 'object' && entry !== null && 'models' in entry) return true
  }
  return Object.keys(value as object).length === 0
}

/** Best-effort cache write; a failure must not fail the fetch. */
async function writeCache(cachePath: string, value: unknown): Promise<void> {
  try {
    await mkdir(dirname(cachePath), { recursive: true })
    await writeFile(cachePath, JSON.stringify(value), 'utf8')
  } catch {
    // A cache that cannot be written costs nothing; the registry already loaded.
  }
}

/** Error message without exposing a non-string throw. */
function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

/** Load the registry and return the adapter-facing catalog.
 * @param options - loader options.
 * @returns the parsed registry as a catalog.
 */
export async function loadRegistryCatalog(options: ModelsDevOptions = {}): Promise<RegistryCatalog> {
  return catalogFromSnapshot(await loadModelsDevRegistry(options))
}
