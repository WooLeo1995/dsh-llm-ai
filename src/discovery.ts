/**
 * Answering "which models can this provider serve?" for a configuration
 * surface's endpoint-interrogation action: `GET <baseURL>/models` with bearer
 * auth, read into candidate models the surface may offer for adoption.
 *
 * Nothing here is a configuration write. The request carries a draft the user
 * is still editing — a typed endpoint, an optional typed key — and the reply is
 * candidate metadata; the settings document remains the only thing that
 * decides what a route serves, so a surface that adopts a candidate writes the
 * declaration itself.
 *
 * Only the `openai-completions` protocol is interrogated: its listing is the
 * one shape a gateway, a self-hosted server, and the official endpoints all
 * agree on, which is the case this action exists for. Every other protocol
 * answers `DISCOVERY_UNSUPPORTED` so the surface falls back to hand-entry
 * rather than guessing a response shape.
 *
 * @module dsh-llm-ai/discovery
 */

import { attributionHeaders, INVALID_CREDENTIAL_CODE, LlmError, normalizeApiKey } from '@deepseek-ai/dsh-llm'
import type { LlmDiscoveredModel, LlmModelDiscoveryRequest } from '@deepseek-ai/dsh-llm'
import { DEFAULT_PROTOCOL, supportedProtocols } from './provider.ts'

/**
 * Endpoint replies larger than this are refused. The endpoint is whatever URL
 * the user typed, so the ceiling holds on the bytes actually read rather than
 * on the length the server claims — and a truncated model listing is not
 * parseable, so overflow rejects instead of truncating.
 */
const MAX_RESPONSE_BYTES = 4 * 1024 * 1024

/** One entry of an OpenAI-compatible `GET /models` reply. */
interface ListingEntry {
  id?: unknown
  /** Common gateway extensions; absent from the official listings. */
  name?: unknown
  display_name?: unknown
  context_window?: unknown
  context_length?: unknown
  max_tokens?: unknown
  max_output_tokens?: unknown
}

/** A positive integer field of a listing entry, or `undefined` when absent or unusable. */
function capacity(...candidates: readonly unknown[]): number | undefined {
  for (const candidate of candidates) {
    if (typeof candidate === 'number' && Number.isInteger(candidate) && candidate > 0) return candidate
  }
  return undefined
}

/** A non-empty string field of a listing entry, or `undefined`. */
function label(...candidates: readonly unknown[]): string | undefined {
  for (const candidate of candidates) {
    if (typeof candidate === 'string' && candidate.length > 0) return candidate
  }
  return undefined
}

/**
 * Join the endpoint base with the listing path. The base is treated as a
 * prefix rather than a URL to resolve against, so a deployment path such as
 * `https://gateway.example/openai/v1` keeps its segments instead of losing
 * them to `URL` resolution.
 */
function listingUrl(baseURL: string): string {
  return `${baseURL.replace(/\/+$/, '')}/models`
}

/**
 * Read a reply body, refusing one that outgrows the ceiling. A declared length
 * is checked first so an honest server is turned away without transferring
 * anything; the accumulated total is what actually enforces the bound, because
 * a server that under-declares (or streams) tells us nothing up front.
 */
async function readBounded(response: Response, url: string): Promise<string> {
  const oversized = (): LlmError =>
    new LlmError(`${url} answered with more than ${MAX_RESPONSE_BYTES} bytes`, 'DISCOVERY_FAILED')
  const declared = Number(response.headers.get('content-length') ?? Number.NaN)
  if (Number.isFinite(declared) && declared > MAX_RESPONSE_BYTES) {
    await response.body?.cancel()
    throw oversized()
  }
  /* v8 ignore next -- fetch always exposes a body stream on a 2xx Response; the null guard is defensive. */
  if (response.body === null) return ''
  const reader = response.body.getReader()
  const chunks: Uint8Array[] = []
  let total = 0
  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      total += value.byteLength
      if (total > MAX_RESPONSE_BYTES) throw oversized()
      chunks.push(value)
    }
  } finally {
    /* v8 ignore next 4 -- cancel() after a completed or abandoned read settles without rejecting; unobserved best-effort cleanup. */
    await reader.cancel().catch(() => {
      // Cancel after a drained read, or after this function walked away from
      // an oversized one, is cleanup; the reply is already decided either way.
    })
  }
  const body = new Uint8Array(total)
  let offset = 0
  for (const chunk of chunks) {
    body.set(chunk, offset)
    offset += chunk.byteLength
  }
  return new TextDecoder().decode(body)
}

/**
 * Read one OpenAI-compatible listing reply. Entries without a usable id are
 * skipped rather than failing the whole interrogation: a single malformed row
 * should not deny the user the rest of a working endpoint's catalog.
 */
function readListing(body: unknown): LlmDiscoveredModel[] {
  const data = (body as { data?: unknown } | null)?.data
  if (!Array.isArray(data)) {
    throw new LlmError(
      'the endpoint\'s model listing has no "data" array; enter this provider\'s models by hand',
      'DISCOVERY_FAILED',
    )
  }
  const models: LlmDiscoveredModel[] = []
  for (const raw of data) {
    const entry = raw as ListingEntry | null
    const id = label(entry?.id)
    if (id === undefined) continue
    const name = label(entry?.name, entry?.display_name)
    const contextWindow = capacity(entry?.context_window, entry?.context_length)
    const maxTokens = capacity(entry?.max_output_tokens, entry?.max_tokens)
    models.push({
      id,
      ...name === undefined ? {} : { name },
      ...contextWindow === undefined ? {} : { contextWindow },
      ...maxTokens === undefined ? {} : { maxTokens },
    })
  }
  return models
}

/**
 * Accept one probe key, or refuse it before the header is built. Without this
 * the `fetch` below would throw a ByteString `TypeError` that this function's
 * catch reports as `could not reach <url>` — blaming the network for a local,
 * deterministic fault.
 * @param raw - the key typed into the draft.
 * @returns the trimmed, usable key.
 */
function usableProbeKey(raw: string): string {
  const checked = normalizeApiKey(raw)
  if (checked.ok) return checked.value
  throw new LlmError(
    checked.reason === 'empty'
      ? 'this provider\'s API key is blank; enter it on the Models page, or clear it to probe unauthenticated'
      : 'this provider\'s API key contains characters no HTTP header can carry; paste the raw key only',
    INVALID_CREDENTIAL_CODE,
  )
}

/** What the plugin answers about the routes it stores, for one interrogation. */
export interface DiscoveryDeps {
  /**
   * One named route's stored facts. The endpoint is the one a request to that
   * route would use, supplied when the draft carries none; `undefined` when
   * neither a stored profile nor the registry describes the route.
   */
  route: (provider: string) => { baseURL?: string; apiKeyEnv?: string } | undefined
  /**
   * Resolve one named route's credential exactly as a request resolves it.
   * Asked only past the endpoint and protocol checks and only when the draft
   * carries no key of its own, so a refused draft costs no credential lookup
   * and no diagnostic about a credential it never needed. A refusal this
   * throws (`MISSING_CREDENTIAL`, `INVALID_CREDENTIAL`) is folded into the
   * discovery wording rather than surfacing as a stream failure; `undefined`
   * — a route that names no reference, or no route at all — interrogates
   * unauthenticated.
   */
  resolveStoredApiKey: (provider: string) => Promise<string | undefined>
}

/**
 * Fold a stored route's credential refusal into the interrogation's wording.
 *
 * The request path's resolver words its refusals for stream calls (provider
 * routes, requests, retry policy); surfaced raw from an interrogation they
 * would read as stream failures. The interrogation re-presents them as its own
 * discovery failure, keeping every name a fix needs — the endpoint, the route,
 * the reference — and never any part of the key. Anything else propagates
 * untouched.
 * @param url - the listing URL the interrogation had reached.
 * @param provider - the stored route whose credential was being resolved.
 * @param apiKeyEnv - the reference that route's profile names.
 * @param error - what the resolver threw.
 * @returns the folded discovery failure.
 */
function storedCredentialRefusal(
  url: string,
  provider: string,
  apiKeyEnv: string,
  error: unknown,
): LlmError {
  if (!(error instanceof LlmError)
    || (error.code !== 'MISSING_CREDENTIAL' && error.code !== INVALID_CREDENTIAL_CODE)) {
    throw error
  }
  const fix = error.code === 'MISSING_CREDENTIAL'
    ? 'is not set — store the credential, or supply a key with the interrogation request'
    : 'resolves to a value no HTTP header can carry — store the raw key only, or supply a key with the interrogation request'
  return new LlmError(
    `could not interrogate ${url}: route "${provider}" names apiKeyEnv ${apiKeyEnv}, which ${fix},`
    + ' or remove apiKeyEnv to interrogate unauthenticated',
    'DISCOVERY_FAILED',
    { cause: error },
  )
}

/**
 * Interrogate one provider endpoint for the models it advertises.
 * @param request - the endpoint, protocol, and one-shot credential to use.
 * @param deps - what the plugin answers about the routes it stores.
 * @returns the advertised models in endpoint order; candidates for adoption,
 *   nothing more — no settings or credential is written.
 * @throws LlmError when the protocol has no readable listing, the endpoint
 *   refuses or fails the request, the reply is not a model listing, or the
 *   stored route's credential cannot be resolved.
 */
export async function discoverModels(
  request: LlmModelDiscoveryRequest,
  deps: DiscoveryDeps,
): Promise<readonly LlmDiscoveredModel[]> {
  // An empty route name is the create case the seam forwards only beside an
  // endpoint; it names no stored route, exactly like none being sent at all.
  const provider = request.provider === undefined || request.provider.length === 0
    ? undefined
    : request.provider
  const stored = provider === undefined ? undefined : deps.route(provider)
  // A form that cleared the field sends '', which says the same thing as one
  // that never had it: the stored route's endpoint is the answer then.
  const baseURL = label(request.baseURL) ?? stored?.baseURL
  if (baseURL === undefined) {
    throw new LlmError(
      `llm-ai has no endpoint for ${provider === undefined ? 'this draft' : `provider "${provider}"`};`
      + " set a baseURL, or enter this provider's models by hand",
      'DISCOVERY_FAILED',
    )
  }
  // A draft that has not chosen a protocol yet is asked as the protocol
  // table's first entry — in v1 the only one, and the one every servable route
  // defaults to — because the alternative, refusing until the field is filled,
  // would withhold the action from the case it exists for.
  const api = request.api ?? DEFAULT_PROTOCOL
  if (!supportedProtocols().includes(api)) {
    throw new LlmError(
      `protocol "${api}" has no model listing this build can read; enter this provider's models by hand`,
      'DISCOVERY_UNSUPPORTED',
    )
  }
  const url = listingUrl(baseURL)
  // A key typed into the draft wins: it is the one the user is testing, and it
  // may be the replacement for exactly the stored key that is failing.
  let apiKey: string | undefined
  if (request.apiKey !== undefined) {
    apiKey = usableProbeKey(request.apiKey)
  } else if (provider !== undefined) {
    try {
      apiKey = await deps.resolveStoredApiKey(provider)
    } catch (error: unknown) {
      throw storedCredentialRefusal(url, provider, stored?.apiKeyEnv ?? '', error)
    }
  }
  let response: Response
  try {
    response = await fetch(url, {
      method: 'GET',
      headers: {
        accept: 'application/json',
        ...apiKey === undefined ? {} : { authorization: `Bearer ${apiKey}` },
        ...attributionHeaders(),
      },
      ...request.signal === undefined ? {} : { signal: request.signal },
    })
  } catch (error: unknown) {
    if (request.signal?.aborted) {
      throw new LlmError('model discovery aborted by caller', 'ABORTED', { cause: error })
    }
    throw new LlmError(`could not reach ${url}`, 'DISCOVERY_FAILED', { cause: error })
  }
  if (!response.ok) {
    throw new LlmError(
      `${url} answered ${response.status}${response.status === 401 || response.status === 403 ? '; check the API key' : ''}`,
      'DISCOVERY_FAILED',
    )
  }
  let text: string
  try {
    text = await readBounded(response, url)
  } catch (error: unknown) {
    // Cancellation during the body read rejects with the abort reason, which
    // may be any value; the caller gets the same coded failure it would have
    // for a cancellation before the request went out.
    if (request.signal?.aborted) {
      throw new LlmError('model discovery aborted by caller', 'ABORTED', { cause: error })
    }
    throw error
  }
  let body: unknown
  try {
    body = JSON.parse(text)
  } catch (error: unknown) {
    throw new LlmError(`${url} did not answer with JSON`, 'DISCOVERY_FAILED', { cause: error })
  }
  return readListing(body)
}
