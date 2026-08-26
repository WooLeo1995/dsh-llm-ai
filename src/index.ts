/**
 * models.dev-cataloged multi-provider LLM adapter plugin. One plugin instance
 * owns a dict of provider routes keyed by models.dev provider id; a route
 * naming a registry provider inherits that provider's endpoint and model
 * catalog as defaults, and a route the registry does not describe is declared
 * outright. The registry snapshot is fetched (or read from the disk cache)
 * once at plugin load, so a change is a plugin reload, never a hot swap.
 * Each stream call resolves the route's credential reference — through
 * `ctx.credentials` when that seam is mounted, the trusted environment
 * otherwise — so a rotated key reaches the next request, and a reference that
 * resolves to nothing fails that request instead of authenticating with an
 * unrelated ambient key.
 *
 * ```yaml
 * - id: llm
 *   name: '@deepseek-ai/dsh-llm-ai'
 *   config:
 *     providers:
 *       # Registry route: everything but the credential comes from models.dev.
 *       deepseek:
 *         apiKeyEnv: DEEPSEEK_API_KEY
 *         retryPolicy:
 *           mode: normal
 *           maxRetries: 2
 *       # Registry route with the catalog narrowed and one capacity corrected.
 *       openrouter:
 *         apiKeyEnv: OPENROUTER_API_KEY
 *         models:
 *           - id: deepseek/deepseek-chat
 *             maxTokens: 16384
 *       # Hand-declared route: the registry ships nothing under this key.
 *       acme-gateway:
 *         displayName: Acme Gateway
 *         apiKeyEnv: ACME_GATEWAY_API_KEY
 *         baseURL: https://gateway.acme.example/v1
 *         models:
 *           - id: acme-large
 *             name: Acme Large
 *             contextWindow: 65536
 *             maxTokens: 4096
 *           - id: acme-think
 *             name: Acme Think
 *             contextWindow: 262144
 *             maxTokens: 32768
 *             # key = selectable level, value = wire spelling; only off may
 *             # leave the value empty (supported, send nothing).
 *             reasoningEfforts:
 *               off:
 *               high: high
 *               max: ultra
 * ```
 *
 * @module @deepseek-ai/dsh-llm-ai
 */

import type { Context } from '@deepseek-ai/cordis'
import { launchEnvironmentOf } from '@deepseek-ai/dsh-launch-environment'
import { assertUsableApiKey, LlmError } from '@deepseek-ai/dsh-llm'
import type { AdapterRegistrationHandle, DirectoryRegistrationHandle, LlmConfigurableProvider } from '@deepseek-ai/dsh-llm'
import { deepEqualJson, installSettingsSection, settingsNamespace } from '@deepseek-ai/dsh-settings'
import { LlmAiAdapter } from './adapter.ts'
import { assertServiceable, Config, resolveProfiles } from './config.ts'
import type { Config as ConfigType, ResolvedLlmAiProfile } from './config.ts'
import { loadRegistryCatalog } from './modelsdev.ts'
import type { RegistryCatalog } from './modelsdev.ts'

export { LlmAiAdapter } from './adapter.ts'
export type { LlmAiAdapterOptions } from './adapter.ts'
export { Config } from './config.ts'
export type {
  LlmAiProviderProfile,
  ResolvedLlmAiProfile,
} from './config.ts'
export type {
  CompatProfile,
  ModelOverride,
  ModelProfile,
  ReasoningEfforts,
  ReasoningLevel,
  ResolvedModel,
} from './catalog.ts'
export { supportedProtocols } from './provider.ts'
export { loadModelsDevRegistry, catalogFromSnapshot } from './modelsdev.ts'
export type { ModelsDevOptions, RegistryCatalog, RegistryModel, RegistryProvider } from './modelsdev.ts'

export const name = 'llm-ai'
export const inject = ['llm']

const NS = settingsNamespace('llm-ai')

/**
 * The registry captures these per route; a change here must re-register.
 * Sorted by provider so a settings document that merely reorders its keys is
 * not mistaken for a route change.
 */
function registrationFacts(profiles: ReadonlyMap<string, ResolvedLlmAiProfile>): unknown {
  return [...profiles.entries()]
    // `displayName` rides along because the registry hands it to every selector
    // through `providerInfo()`: a rename that did not re-register would leave
    // the old label showing until some unrelated fact happened to change.
    .map(([provider, profile]) => ({
      provider,
      displayName: profile.displayName,
      retryPolicy: profile.retryPolicy,
    }))
    .sort((left, right) => left.provider.localeCompare(right.provider))
}

/**
 * The configurable-provider directory: every registry provider this snapshot
 * describes — including the ones this adapter cannot serve, so configuration
 * surfaces can explain what is missing — plus every route the current
 * profiles declare. A hand-declared route has no registry entry, so without
 * this union it would have no settings address and configuration surfaces
 * could neither show nor edit it.
 *
 * The profile half is unconditional, which is what keeps a route already
 * stored against a withheld provider editable and deletable rather than
 * stranded in the settings document with nothing on the page to remove it.
 * @param catalog - the loaded registry catalog.
 * @param profiles - the currently resolved provider profiles.
 * @returns the directory entries in registry order, declared routes last.
 */
function directoryEntries(
  catalog: RegistryCatalog,
  profiles: ReadonlyMap<string, ResolvedLlmAiProfile>,
): LlmConfigurableProvider[] {
  const registry = new Set(catalog.providers().map(provider => provider.id))
  const entries = new Map<string, LlmConfigurableProvider>()
  const declare = (provider: string, displayName: string): void => {
    entries.set(provider, {
      provider,
      displayName,
      settingsNs: NS,
      settingsPath: ['providers', provider],
      // Membership of the registry, not of the settings document: narrowing a
      // shipped provider's models stores a profile too, and that route is
      // still one models.dev knows.
      declared: !registry.has(provider),
    })
  }
  for (const provider of catalog.providers()) declare(provider.id, provider.displayName)
  for (const [provider, profile] of profiles) declare(provider, profile.displayName)
  return [...entries.values()]
}

/** Register one models.dev-cataloged adapter for all configured provider routes. */
export async function apply(ctx: Context, config: ConfigType): Promise<void> {
  // The catalog is a composition fact read once at load: switching its source
  // or URL requires a reload, never a hot swap, because profile resolution
  // reads the registry synchronously. A models.dev fetch failure is loud only
  // when nothing is cached; the deployment asked for this catalog, and
  // serving nothing would be silent degradation.
  const catalog = await loadRegistryCatalog({
    ...config.catalogUrl === undefined ? {} : { url: config.catalogUrl },
    ...config.catalogCachePath === undefined ? {} : { cachePath: config.catalogCachePath },
  })
  let current: () => ConfigType = () => config
  let lastRaw: ConfigType | undefined
  let memoized: ReadonlyMap<string, ResolvedLlmAiProfile> | undefined
  /**
   * The resolved profiles for the current configuration, memoized by the raw
   * snapshot's identity — which is also what makes the adapter's own reads
   * stable across operations that observe no change.
   *
   * No fallback for an unserviceable snapshot lives here: the section schema
   * resolves the whole profile set, so a write that could not be served is
   * refused where it is written, and the settings seam keeps a namespace's
   * last good value for a stored section that fails. Anything reaching this
   * point has already resolved once.
   */
  const profiles = (): ReadonlyMap<string, ResolvedLlmAiProfile> => {
    const raw = current()
    if (raw === lastRaw && memoized !== undefined) return memoized
    const next = resolveProfiles(raw.providers, catalog)
    lastRaw = raw
    memoized = next
    return next
  }
  profiles()

  /**
   * Resolve one route's credential for one stream call. A profile naming no
   * reference at all defers to ambient discovery (this runtime sends no
   * authorization header); once one is named, a miss must fail loud — handing
   * the request no key would let a gateway authenticate it against an
   * unrelated ambient account. The credentials seam is resolved per call for
   * the same reason the attachments seam is: Cordis load order must not
   * freeze optional availability, and when one is mounted its own layers
   * already include the trusted environment, so no second read of the
   * environment happens beside it.
   */
  const resolveApiKey = async (profile: ResolvedLlmAiProfile): Promise<string | undefined> => {
    const ref = profile.apiKeyEnv
    if (ref === undefined) return undefined
    const credentials = ctx.get('credentials')
    const hit = credentials !== undefined
      ? (await credentials.resolve(ref))?.value
      // Without the seam there is no managed store to rank against, so the
      // trusted environment is the whole credential plane.
      : launchEnvironmentOf(ctx).get(ref)?.value
    // Trimmed and header-carryable before it can reach a request; a refusal
    // names the route and the reference, never any part of the key.
    if (hit !== undefined && hit.length > 0) {
      return assertUsableApiKey(hit, `llm-ai provider route "${profile.provider}"`, ref)
    }
    throw new LlmError(
      `llm-ai: no credential for provider route "${profile.provider}"; its profile names apiKeyEnv ${ref},`
      + ` which is not set — store ${ref} through the credentials service (the web Models page writes it)`
      + ' or export it in the launching environment, or remove apiKeyEnv from the providers entry'
      + ' (cordis.yml or the llm-ai: settings section) if this route should authenticate without a'
      + ' bearer token',
      'MISSING_CREDENTIAL',
    )
  }

  const adapter = new LlmAiAdapter({
    profiles,
    resolveApiKey,
    // Resolved per call rather than injected: the attachment service is an
    // optional composition fact, so a composition loading it after this
    // plugin still serves image requests and one omitting it keeps text-only
    // routes whole.
    resolveAttachments: () => ctx.get('attachments'),
  })
  // The full registry is configurable from the moment the plugin mounts —
  // dormant or not — so configuration surfaces can offer every models.dev
  // provider before any route exists. Hand-declared routes join it as
  // profiles appear, and leave with them.
  let directory: DirectoryRegistrationHandle | undefined
  let directoryFacts: unknown
  const ensureDirectory = (): void => {
    const entries = directoryEntries(catalog, profiles())
    // An empty registry and an empty profile set declare nothing; the seam
    // refuses an empty initial registration, and nothing to declare is the
    // dormant posture rather than a failure.
    if (entries.length === 0) return
    if (deepEqualJson(entries, directoryFacts)) return
    // Atomic replace, never dispose-then-register: a route another adapter
    // family already declares would otherwise leave this plugin's whole
    // directory withdrawn and the Models page empty. The candidate set is
    // validated first, so a collision keeps the previous entries serving and
    // only costs a diagnostic.
    if (directory === undefined) {
      directory = ctx.llm.registerConfigurableProviders(entries)
    } else {
      directory.replace(entries)
    }
    directoryFacts = entries
  }
  ensureDirectory()
  // Route effects bind to this apply fiber via the stable `ctx` reference,
  // even when a swap runs inside the scoped settings callback below. A bare
  // mount (zero routes) is the dormant posture: nothing registers until a
  // settings section supplies profiles, and routes drop when it empties.
  let registration: AdapterRegistrationHandle | undefined
  let registeredFacts: unknown
  const ensureRegistrationFacts = (): void => {
    const facts = registrationFacts(profiles())
    if (deepEqualJson(facts, registeredFacts)) return
    // The registry captures the route set and each route's retry policy at
    // registration, so a change to either must re-register. The swap is
    // atomic (same adapter instance, validated before anything moves): a
    // conflicting route leaves the previous routes serving requests, and
    // `registeredFacts` only advances once the registry actually holds the
    // new set — so returning to a working configuration always re-applies.
    const routes = [...profiles().keys()]
    if (registration === undefined) {
      // Dormant bare mount: nothing is registered until a section supplies
      // profiles, and an empty section keeps it that way.
      if (routes.length === 0) {
        registeredFacts = facts
        return
      }
      registration = ctx.llm.registerAdapter(routes, adapter)
    } else {
      registration.replace(routes)
    }
    registeredFacts = facts
  }
  ensureRegistrationFacts()

  installSettingsSection(ctx, NS, Config, config, {
    // Refuse an unserviceable section where it is written: without this a
    // schema-valid profile the adapter cannot serve would be stored and then
    // silently disable every route in this namespace.
    validate: section => assertServiceable(section, catalog),
    setSource: (source) => {
      current = source
    },
    onChange: () => {
      // Named here rather than left to the settings watcher: `assertServiceable`
      // cannot see the llm registry, so a profile claiming a route another
      // adapter family owns is stored successfully and only fails at this swap.
      // Without its own diagnostic that refusal reaches the operator as a
      // generic "settings: watcher failed", naming neither the route nor why it
      // is not serving. The previous routes keep serving either way.
      try {
        ensureRegistrationFacts()
      } catch (error) {
        ctx.logger.error('llm-ai: keeping the previously registered routes after a refused update')
        ctx.logger.error(error)
      }
      // The directory follows the resolved profiles independently of the
      // route swap above: a profile that registers cleanly still deserves its
      // settings address even when another profile's route refused. A refused
      // directory swap is contained here for the same reason the registry's
      // is: the previous entries keep serving, and `directoryFacts` stays put
      // so returning to a working configuration re-applies.
      try {
        ensureDirectory()
      } catch (error) {
        ctx.logger.error('llm-ai: keeping the previous configurable-provider directory after a refused update')
        ctx.logger.error(error)
      }
    },
  })
}
