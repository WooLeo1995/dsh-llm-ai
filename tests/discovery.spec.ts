/**
 * Endpoint interrogation at the `ctx.llm` seam: `GET <baseURL>/models` with
 * bearer auth against loopback gateways, candidate models with disclosed
 * capacities, stored-route credential resolution with typed keys winning,
 * refusal refusals folded into discovery wording, the received-bytes ceiling,
 * cancellation, protocol gating, and the nothing-is-stored posture. Loopback
 * only; no network.
 */

import { createServer } from 'node:http'
import type { IncomingMessage, Server, ServerResponse } from 'node:http'
import { readFile, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import LlmRuntime, { LlmError, userAgent } from '@deepseek-ai/dsh-llm'
import FileSettingsProvider from '@deepseek-ai/dsh-settings-file'
import { settingsNamespace } from '@deepseek-ai/dsh-settings'
import * as LlmAi from '../src/index.ts'
import { discoverModels } from '../src/discovery.ts'
import type { DiscoveryDeps } from '../src/discovery.ts'
import { fixtureRegistry, home, registryServer } from './registry.ts'

const NS = settingsNamespace('llm-ai')

const cleanups: Array<() => Promise<void>> = []
const servers: Server[] = []

afterEach(async () => {
  vi.unstubAllEnvs()
  vi.unstubAllGlobals()
  while (cleanups.length > 0) await cleanups.pop()!()
  await Promise.all(servers.splice(0).map(server => new Promise<void>(resolve => {
    server.closeAllConnections()
    server.close(() => { resolve() })
  })))
})

/** Boot the runtime plus this plugin against a loopback registry snapshot. */
async function boot(registry: unknown = fixtureRegistry(), config: LlmAi.Config = {}): Promise<Context> {
  const server = await registryServer(registry)
  cleanups.push(server.close)
  const ctx = new Context()
  cleanups.push(async () => {
    await ctx.fiber.dispose()
  })
  await ctx.plugin(LlmRuntime)
  await ctx.plugin(LlmAi, {
    catalogUrl: server.url,
    catalogCachePath: join(await home('dsh-ai-discovery-'), 'cache.json'),
    ...config,
  })
  return ctx
}

interface ListingServer {
  url: string
  paths: string[]
  headers: IncomingMessage['headers'][]
}

/**
 * A stand-in gateway that answers one scripted `GET /models`. `chunks` writes
 * without a declared length, which is how a real streamed reply arrives.
 */
async function listingServer(behavior: {
  status?: number
  body?: string
  chunks?: string[]
}): Promise<ListingServer> {
  const paths: string[] = []
  const headers: IncomingMessage['headers'][] = []
  const server = createServer((request: IncomingMessage, response: ServerResponse) => {
    paths.push(request.url ?? '')
    headers.push(request.headers)
    if (behavior.chunks !== undefined) {
      // No declared length: the ceiling has to hold on what is read.
      response.writeHead(behavior.status ?? 200, { 'content-type': 'application/json' })
      for (const chunk of behavior.chunks) response.write(chunk)
      response.end()
      return
    }
    const body = behavior.body ?? '{}'
    response.writeHead(behavior.status ?? 200, {
      'content-type': 'application/json',
      'content-length': String(Buffer.byteLength(body)),
    })
    response.end(body)
  })
  servers.push(server)
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const address = server.address()
  if (address === null || typeof address === 'string') throw new Error('loopback server has no port')
  return { url: `http://127.0.0.1:${address.port}`, paths, headers }
}

/** A listing body wrapping the given rows in the `data` array. */
const listing = (...entries: unknown[]): string => JSON.stringify({ data: entries })

/** Direct-call deps answering nothing about any stored route. */
const noStored: DiscoveryDeps = {
  route: () => undefined,
  resolveStoredApiKey: async () => undefined,
}

describe('draft interrogation', () => {
  it('reads a listing, keeping ids, names, and disclosed capacities; the typed key rides along', async () => {
    const server = await listingServer({
      body: listing(
        { id: 'acme-large', name: 'Acme Large', context_length: 65_536, max_output_tokens: 4_096 },
        { id: 'acme-small' },
      ),
    })
    const ctx = await boot()

    const models = await ctx.llm.discoverModels(NS, { baseURL: `${server.url}/v1`, apiKey: 'probe-key' })

    expect(models).toEqual([
      { id: 'acme-large', name: 'Acme Large', contextWindow: 65_536, maxTokens: 4_096 },
      { id: 'acme-small' },
    ])
    expect(server.paths).toEqual(['/v1/models'])
    expect(server.headers[0]?.authorization).toBe('Bearer probe-key')
    expect(server.headers[0]?.['user-agent']).toBe(userAgent())
  })

  it('reads the gateway spellings of the name and capacity fields', async () => {
    const server = await listingServer({
      body: listing({ id: 'acme-vision', display_name: 'Acme Vision', context_window: 131_072, max_tokens: 8_192 }),
    })
    const ctx = await boot()

    expect(await ctx.llm.discoverModels(NS, { baseURL: server.url })).toEqual([
      { id: 'acme-vision', name: 'Acme Vision', contextWindow: 131_072, maxTokens: 8_192 },
    ])
  })

  it('keeps a deployment path instead of resolving it away', async () => {
    const server = await listingServer({ body: listing({ id: 'm' }) })
    const ctx = await boot()

    await ctx.llm.discoverModels(NS, { baseURL: `${server.url}/openai/v1/` })

    expect(server.paths).toEqual(['/openai/v1/models'])
  })

  it('offers no credential when the draft names none', async () => {
    const server = await listingServer({ body: listing({ id: 'm' }) })
    const ctx = await boot()

    await ctx.llm.discoverModels(NS, { baseURL: server.url })

    expect(server.headers[0]?.authorization).toBeUndefined()
  })

  it('drops unusable rows rather than failing the whole listing', async () => {
    const server = await listingServer({
      body: listing(
        { id: 'good' },
        { id: '' },
        { name: 'no id at all' },
        null,
        { id: 'good' },
        { id: 'zero-capacity', context_length: 0, max_tokens: -1 },
      ),
    })
    const ctx = await boot()

    expect(await ctx.llm.discoverModels(NS, { baseURL: server.url })).toEqual([
      { id: 'good' },
      { id: 'zero-capacity' },
    ])
  })

  it('points at the credential for a rejected one, and only then', async () => {
    const ctx = await boot()

    for (const status of [401, 403]) {
      const refused = await listingServer({ status, body: '{"error":"nope"}' })
      await expect(ctx.llm.discoverModels(NS, { baseURL: refused.url, apiKey: 'wrong' }))
        .rejects.toThrow(new RegExp(`answered ${status}; check the API key`))
    }

    // A server fault is not a credential problem, so it must not send the user
    // off to re-check a key that is fine.
    const broken = await listingServer({ status: 500, body: '{"error":"boom"}' })
    await expect(ctx.llm.discoverModels(NS, { baseURL: broken.url, apiKey: 'fine' }))
      .rejects.toThrow(/answered 500$/)
  })

  it('reports a reply that is not a model listing', async () => {
    const ctx = await boot()

    const shaped = await listingServer({ body: '{"models":[]}' })
    await expect(ctx.llm.discoverModels(NS, { baseURL: shaped.url }))
      .rejects.toThrow(/no "data" array; enter this provider's models by hand/)

    const broken = await listingServer({ body: 'not json at all' })
    await expect(ctx.llm.discoverModels(NS, { baseURL: broken.url }))
      .rejects.toThrow(/did not answer with JSON/)
  })

  it('refuses an oversized reply, whether its length is declared or streamed', async () => {
    const ctx = await boot()
    // Just over the four-megabyte ceiling, as one padded model row.
    const oversized = `{"data":[{"id":"m","pad":"${'x'.repeat(4 * 1024 * 1024)}"}]}`

    const declared = await listingServer({ body: oversized })
    await expect(ctx.llm.discoverModels(NS, { baseURL: declared.url }))
      .rejects.toThrow(/answered with more than 4194304 bytes/)

    // A streamed reply declares no length, so the ceiling has to hold on the
    // body the harness actually read.
    const streamed = await listingServer({ chunks: ['{"data":[{"id":"m","pad":"', 'x'.repeat(4 * 1024 * 1024), '"}]}'] })
    await expect(ctx.llm.discoverModels(NS, { baseURL: streamed.url }))
      .rejects.toThrow(/answered with more than 4194304 bytes/)
  })

  it('reports an unreachable endpoint instead of an empty catalog', async () => {
    const ctx = await boot()
    // Port 9 is the discard service: nothing accepts a connection there.
    await expect(ctx.llm.discoverModels(NS, { baseURL: 'http://127.0.0.1:9/v1' }))
      .rejects.toMatchObject({ code: 'DISCOVERY_FAILED' })
  })

  it('reports cancellation during the body read as an abort, not a raw reason', async () => {
    const ctx = await boot()
    const controller = new AbortController()
    const bodyRead = Promise.withResolvers<undefined>()
    vi.stubGlobal('fetch', async (_url: string | URL, init?: RequestInit) => {
      const signal = init?.signal
      if (signal === undefined || signal === null) throw new Error('expected a discovery signal')
      return new Response(new ReadableStream<Uint8Array>({
        pull(stream) {
          bodyRead.resolve(undefined)
          return new Promise<void>((resolve) => {
            signal.addEventListener('abort', () => {
              stream.error(signal.reason)
              resolve()
            }, { once: true })
          })
        },
      }))
    })
    const probe = ctx.llm.discoverModels(NS, {
      baseURL: 'https://slow.example/v1',
      signal: controller.signal,
    })
    await bodyRead.promise
    controller.abort('test cancellation')

    await expect(probe).rejects.toMatchObject({ code: 'ABORTED' })
  })

  it('honors caller cancellation', async () => {
    const ctx = await boot()
    const aborted = AbortSignal.abort('test cancellation')
    await expect(ctx.llm.discoverModels(NS, {
      baseURL: 'http://127.0.0.1:9/v1',
      signal: aborted,
    })).rejects.toMatchObject({ code: 'ABORTED' })
  })
})

describe('protocol gating', () => {
  it.each(['anthropic-messages', 'openai-codex-responses', 'made-up-protocol'])(
    'says it cannot interrogate %s rather than guessing a shape',
    async (api) => {
      const server = await listingServer({ body: listing({ id: 'm' }) })
      const ctx = await boot()

      await expect(ctx.llm.discoverModels(NS, { baseURL: server.url, api }))
        .rejects.toMatchObject({ code: 'DISCOVERY_UNSUPPORTED' })
      // The refusal happens before any request: nothing was asked.
      expect(server.paths).toEqual([])
    },
  )
})

describe('stored routes', () => {
  it('resolves a stored reference; a typed key wins; an unstored route probes unauthenticated', async () => {
    const server = await listingServer({ body: listing({ id: 'm' }) })
    vi.stubEnv('ACME_GATEWAY_KEY', 'stored-key')
    const ctx = await boot(fixtureRegistry(), {
      providers: {
        'acme-gateway': { apiKeyEnv: 'ACME_GATEWAY_KEY', baseURL: server.url, models: [{ id: 'acme-large' }] },
      },
    })

    // What the Models page sends after a key is saved: the form holds the
    // redacted descriptor, so the draft names the route and the endpoint and
    // no credential at all. Interrogating unauthenticated would answer 401 and
    // read as a wrong key.
    await ctx.llm.discoverModels(NS, { provider: 'acme-gateway', baseURL: server.url })
    // A key typed into the form is the one being tested — possibly the
    // replacement for the stored one — so it wins.
    await ctx.llm.discoverModels(NS, { provider: 'acme-gateway', baseURL: server.url, apiKey: 'typed' })
    // A route no profile declares yet is the create case: nothing is stored.
    await ctx.llm.discoverModels(NS, { provider: 'not-declared-yet', baseURL: server.url })

    expect(server.headers.map(headers => headers.authorization))
      .toEqual(['Bearer stored-key', 'Bearer typed', undefined])
  })

  it('uses the stored route endpoint when the draft carries none, including a cleared field', async () => {
    const server = await listingServer({ body: listing({ id: 'm' }) })
    const ctx = await boot(fixtureRegistry(), {
      providers: {
        'acme-gateway': { baseURL: server.url, models: [{ id: 'acme-large' }] },
      },
    })

    await ctx.llm.discoverModels(NS, { provider: 'acme-gateway' })
    // A form that cleared the field sends '', which says the same thing as one
    // that never had it.
    await ctx.llm.discoverModels(NS, { provider: 'acme-gateway', baseURL: '' })

    expect(server.paths).toEqual(['/models', '/models'])
    // The route names no credential, so both interrogations are unauthenticated.
    expect(server.headers.map(headers => headers.authorization)).toEqual([undefined, undefined])
  })

  it('resolves a registry route through its stored profile, and a registry provider no profile stores through the registry endpoint', async () => {
    const server = await listingServer({ body: listing({ id: 'deepseek-chat' }, { id: 'vision-large' }) })
    const registry: unknown = {
      ...fixtureRegistry(),
      deepseek: {
        id: 'deepseek',
        name: 'DeepSeek',
        env: ['DEEPSEEK_API_KEY'],
        api: server.url,
        models: { 'deepseek-chat': { name: 'DeepSeek Chat', limit: { context: 65_536, output: 4_096 } } },
      },
      visionai: {
        id: 'visionai',
        name: 'Vision AI',
        env: ['VISIONAI_API_KEY'],
        api: server.url,
        models: { 'vision-large': { name: 'Vision Large', limit: { context: 131_072, output: 16_384 } } },
      },
    }
    vi.stubEnv('AI_REGISTRY_KEY', 'registry-key')
    const ctx = await boot(registry, { providers: { deepseek: { apiKeyEnv: 'AI_REGISTRY_KEY' } } })

    // A registry route whose profile declares no baseURL: the registry's
    // endpoint is the one a request would use, and its stored reference
    // resolves through the same path.
    await ctx.llm.discoverModels(NS, { provider: 'deepseek' })
    expect(server.headers[0]?.authorization).toBe('Bearer registry-key')

    // The create draft a surface offers from the directory: no profile yet,
    // so the registry endpoint answers and nothing authenticates.
    await ctx.llm.discoverModels(NS, { provider: 'visionai' })
    expect(server.paths).toEqual(['/models', '/models'])
    expect(server.headers[1]?.authorization).toBeUndefined()
  })

  it('folds a stored reference that resolves to nothing into the discovery failure', async () => {
    const server = await listingServer({ body: listing({ id: 'm' }) })
    vi.stubEnv('ACME_MISSING_KEY', '')
    const ctx = await boot(fixtureRegistry(), {
      providers: {
        'acme-gateway': { apiKeyEnv: 'ACME_MISSING_KEY', baseURL: server.url, models: [{ id: 'acme-large' }] },
      },
    })

    await expect(ctx.llm.discoverModels(NS, { provider: 'acme-gateway', baseURL: server.url }))
      .rejects.toMatchObject({
        code: 'DISCOVERY_FAILED',
        message: expect.stringMatching(/acme-gateway.*ACME_MISSING_KEY/s),
      })
    // The refusal happened before the network: the gateway was never asked.
    expect(server.paths).toEqual([])
  })

  it('folds an unusable stored key the same way, never echoing it', async () => {
    const server = await listingServer({ body: listing({ id: 'm' }) })
    vi.stubEnv('ACME_BAD_KEY', 'pl4nted-sécret key')
    const ctx = await boot(fixtureRegistry(), {
      providers: {
        'acme-gateway': { apiKeyEnv: 'ACME_BAD_KEY', baseURL: server.url, models: [{ id: 'acme-large' }] },
      },
    })

    const failure = await ctx.llm.discoverModels(NS, { provider: 'acme-gateway', baseURL: server.url })
      .then(() => undefined, (error: LlmError) => error)
    expect(failure).toMatchObject({ code: 'DISCOVERY_FAILED' })
    expect(failure?.message).toMatch(/ACME_BAD_KEY/)
    expect(failure?.message).toMatch(/no HTTP header can carry/)
    // The refusal may name where to fix, never what was found there.
    expect(failure?.message).not.toContain('pl4nted')
    expect(failure?.message).not.toContain('sécret')
    expect(server.paths).toEqual([])
  })

  it('lets a typed key bypass the stored resolution entirely', async () => {
    const server = await listingServer({ body: listing({ id: 'm' }) })
    vi.stubEnv('ACME_MISSING_KEY', '')
    const ctx = await boot(fixtureRegistry(), {
      providers: {
        'acme-gateway': { apiKeyEnv: 'ACME_MISSING_KEY', baseURL: server.url, models: [{ id: 'acme-large' }] },
      },
    })

    // The stored reference would fold into a refusal, but the draft carries
    // the key under test and the stored one is never asked for.
    await expect(ctx.llm.discoverModels(NS, {
      provider: 'acme-gateway',
      baseURL: server.url,
      apiKey: 'typed-instead',
    })).resolves.toEqual([{ id: 'm' }])
    expect(server.headers[0]?.authorization).toBe('Bearer typed-instead')
  })

  it('names the route when neither a profile nor the registry describes it', async () => {
    const ctx = await boot()

    const refusal = ctx.llm.discoverModels(NS, { provider: 'ghost' })
    await expect(refusal).rejects.toMatchObject({ code: 'DISCOVERY_FAILED' })
    await expect(refusal).rejects.toThrow(/no endpoint for provider "ghost"/)
  })
})

describe('the offer', () => {
  it('serves its own namespace and refuses one it does not serve', async () => {
    const server = await listingServer({ body: listing({ id: 'deepseek-chat' }) })
    const registry: unknown = {
      ...fixtureRegistry(),
      deepseek: {
        id: 'deepseek',
        name: 'DeepSeek',
        env: ['DEEPSEEK_API_KEY'],
        api: server.url,
        models: { 'deepseek-chat': { name: 'DeepSeek Chat', limit: { context: 65_536, output: 4_096 } } },
      },
    }
    const ctx = await boot(registry)

    await expect(ctx.llm.discoverModels(NS, { provider: 'deepseek' })).resolves.toHaveLength(1)
    await expect(ctx.llm.discoverModels('llm-deepseek', { baseURL: server.url }))
      .rejects.toMatchObject({ code: 'NO_DISCOVERY' })
    await expect(ctx.llm.discoverModels(NS, { baseURL: '' }))
      .rejects.toMatchObject({ code: 'INVALID_DISCOVERY' })
  })

  it('withdraws the offer when the plugin unloads', async () => {
    const server = await listingServer({ body: listing({ id: 'm' }) })
    const registryServerHandle = await registryServer({
      ...fixtureRegistry(),
      deepseek: {
        id: 'deepseek',
        name: 'DeepSeek',
        env: ['DEEPSEEK_API_KEY'],
        api: server.url,
        models: { 'deepseek-chat': { name: 'DeepSeek Chat', limit: { context: 65_536, output: 4_096 } } },
      },
    })
    cleanups.push(registryServerHandle.close)
    const ctx = new Context()
    cleanups.push(async () => {
      await ctx.fiber.dispose()
    })
    await ctx.plugin(LlmRuntime)
    const fiber = await ctx.plugin(LlmAi, {
      catalogUrl: registryServerHandle.url,
      catalogCachePath: join(await home('dsh-ai-discovery-unload-'), 'cache.json'),
    })
    await expect(ctx.llm.discoverModels(NS, { provider: 'deepseek' })).resolves.toHaveLength(1)

    await fiber.dispose()

    await expect(ctx.llm.discoverModels(NS, { provider: 'deepseek' }))
      .rejects.toMatchObject({ code: 'NO_DISCOVERY' })
  })

  it('offers candidates for adoption only: a declaration flow stores nothing', async () => {
    const server = await listingServer({ body: listing({ id: 'deepseek-chat' }) })
    const registry: unknown = {
      ...fixtureRegistry(),
      deepseek: {
        id: 'deepseek',
        name: 'DeepSeek',
        env: ['DEEPSEEK_API_KEY'],
        api: server.url,
        models: { 'deepseek-chat': { name: 'DeepSeek Chat', limit: { context: 65_536, output: 4_096 } } },
      },
    }
    const registryServerHandle = await registryServer(registry)
    cleanups.push(registryServerHandle.close)
    const dir = await home('dsh-ai-discovery-settings-')
    const settingsPath = join(dir, 'settings.yaml')
    await writeFile(settingsPath, '')
    const ctx = new Context()
    cleanups.push(async () => {
      await ctx.fiber.dispose()
    })
    await ctx.plugin(LlmRuntime)
    await ctx.plugin(FileSettingsProvider, { path: settingsPath, watch: false })
    await ctx.plugin(LlmAi, {
      catalogUrl: registryServerHandle.url,
      catalogCachePath: join(dir, 'cache.json'),
    })

    await ctx.llm.discoverModels(NS, { provider: 'deepseek' })

    // The reply is candidate metadata: the settings document still decides
    // what a route serves, and nothing was registered or written.
    expect(await readFile(settingsPath, 'utf8')).toBe('')
    expect(ctx.llm.listProviders()).toEqual([])
  })
})

describe('probe key format and direct folds', () => {
  it('reports an illegal probe key as a credential fault, not an unreachable endpoint', async () => {
    await expect(discoverModels({
      baseURL: 'https://acme.test',
      apiKey: 'sk-\u{1F600}',
    }, noStored)).rejects.toMatchObject({ code: 'INVALID_CREDENTIAL' })
  })

  it('reports a blank probe key as a credential fault too', async () => {
    // The Models page omits `apiKey` for a cleared field rather than sending
    // '', so this pins the contract for every other caller: a supplied key is
    // judged, and only an absent one interrogates unauthenticated.
    await expect(discoverModels({
      baseURL: 'https://acme.test',
      apiKey: '',
    }, noStored)).rejects.toMatchObject({ code: 'INVALID_CREDENTIAL' })
  })

  it('leaves a probe with no key unauthenticated, with or without an empty route name', async () => {
    const requests: RequestInit[] = []
    vi.stubGlobal('fetch', async (_url: string | URL, init?: RequestInit) => {
      requests.push(init ?? {})
      return new Response(JSON.stringify({ data: [] }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      })
    })

    await discoverModels({ baseURL: 'https://acme.test' }, noStored)
    // An empty route name is the create case the seam itself forwards only
    // beside an endpoint; it names no stored route either.
    await discoverModels({ provider: '', baseURL: 'https://acme.test' }, noStored)

    expect(requests).toHaveLength(2)
    for (const init of requests) {
      expect(new Headers(init.headers).has('authorization')).toBe(false)
    }
  })

  it('propagates a resolver failure that is not a credential refusal untouched', async () => {
    const deps: DiscoveryDeps = {
      route: () => ({ baseURL: 'https://acme.test' }),
      resolveStoredApiKey: () => Promise.reject(new Error('credentials seam unavailable')),
    }
    await expect(discoverModels({ provider: 'acme-gateway', baseURL: 'https://acme.test' }, deps))
      .rejects.toThrow(/credentials seam unavailable/)

    const coded: DiscoveryDeps = {
      ...deps,
      resolveStoredApiKey: () => Promise.reject(new LlmError('route suspended', 'TRANSPORT')),
    }
    await expect(discoverModels({ provider: 'acme-gateway', baseURL: 'https://acme.test' }, coded))
      .rejects.toMatchObject({ code: 'TRANSPORT' })
  })

  it('says where a draft with no endpoint at all must get its models', async () => {
    await expect(discoverModels({}, noStored)).rejects.toThrow(/set a baseURL/)
    // Only reachable by calling the module directly: the seam refuses a
    // request naming neither a route nor an endpoint before this runs.
    await expect(discoverModels({ provider: 'acme-gateway' }, noStored))
      .rejects.toThrow(/no endpoint for provider "acme-gateway"/)
  })
})
