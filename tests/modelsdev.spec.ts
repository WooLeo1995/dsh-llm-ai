import { readFile, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { describe, expect, it, vi } from 'vitest'
import { catalogFromSnapshot, defaultCachePath, loadModelsDevRegistry } from '../src/modelsdev.ts'
import { fixtureRegistry, home, provider } from './registry.ts'

/** A fetch replying with one JSON body. */
function okFetch(body: unknown): typeof fetch {
  return vi.fn(async () => new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  })) as unknown as typeof fetch
}

/** A fetch that never reaches a registry. */
const deadFetch = vi.fn(async () => { throw new Error('network down') }) as unknown as typeof fetch

describe('registry loading', () => {
  it('uses the injected fetch with a JSON accept header and writes the cache', async () => {
    const dir = await home('dsh-ai-loader-')
    const cachePath = join(dir, 'cache.json')
    const registry = { openai: provider({ id: 'openai', models: {} }) }
    const fetchImpl = vi.fn(async () => new Response(JSON.stringify(registry), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    }))

    const loaded = await loadModelsDevRegistry({ url: 'https://registry.test/api.json', cachePath, fetchImpl })
    expect(loaded).toEqual(registry)
    expect(fetchImpl).toHaveBeenCalledWith(
      'https://registry.test/api.json',
      expect.objectContaining({ headers: { accept: 'application/json' } }),
    )
    expect(JSON.parse(await readFile(cachePath, 'utf8'))).toEqual(registry)
  })

  it('keeps serving when the cache cannot be written', async () => {
    const dir = await home('dsh-ai-loader-')
    // A regular file as the cache's parent directory: the cache write fails
    // and is swallowed, because the registry already loaded.
    const file = join(dir, 'file.txt')
    await writeFile(file, 'x')
    const registry = { acme: provider() }
    await expect(loadModelsDevRegistry({
      url: 'https://registry.test/api.json',
      cachePath: join(file, 'cache.json'),
      fetchImpl: okFetch(registry),
    })).resolves.toEqual(registry)
  })

  it('serves the disk cache when the network fails, without throwing', async () => {
    const dir = await home('dsh-ai-loader-')
    const cachePath = join(dir, 'cache.json')
    const registry = { cached: provider({ id: 'cached', models: {} }) }
    await writeFile(cachePath, JSON.stringify(registry))

    const loaded = await loadModelsDevRegistry({ url: 'https://registry.test/api.json', cachePath, fetchImpl: deadFetch })
    expect(loaded).toEqual(registry)
  })

  it('fails loud when the network fails and nothing is cached', async () => {
    const dir = await home('dsh-ai-loader-')
    await expect(loadModelsDevRegistry({
      url: 'https://registry.test/api.json',
      cachePath: join(dir, 'absent.json'),
      fetchImpl: deadFetch,
    })).rejects.toThrow(/could not load the models.dev catalog/)
  })

  it('reports a non-error throw without exposing it', async () => {
    const dir = await home('dsh-ai-loader-')
    await expect(loadModelsDevRegistry({
      url: 'https://registry.test/api.json',
      cachePath: join(dir, 'absent.json'),
      fetchImpl: vi.fn(async () => { throw 'boom' }) as unknown as typeof fetch,
    })).rejects.toThrow(/boom/)
  })

  it('fails loud on an HTTP error status', async () => {
    const dir = await home('dsh-ai-loader-')
    await expect(loadModelsDevRegistry({
      url: 'https://registry.test/api.json',
      cachePath: join(dir, 'absent.json'),
      fetchImpl: vi.fn(async () => new Response('nope', { status: 503 })) as unknown as typeof fetch,
    })).rejects.toThrow(/models.dev answered 503/)
  })

  it('rejects a reply that is not a provider-keyed registry', async () => {
    const dir = await home('dsh-ai-loader-')
    await expect(loadModelsDevRegistry({
      url: 'https://registry.test/api.json',
      cachePath: join(dir, 'absent.json'),
      fetchImpl: okFetch({ data: [] }),
    })).rejects.toThrow(/did not answer a provider-keyed registry/)
  })

  it('treats a non-object reply as not a registry', async () => {
    const dir = await home('dsh-ai-loader-')
    await expect(loadModelsDevRegistry({
      url: 'https://registry.test/api.json',
      cachePath: join(dir, 'absent.json'),
      fetchImpl: okFetch(42),
    })).rejects.toThrow(/did not answer a provider-keyed registry/)
  })

  it('accepts an empty registry snapshot', async () => {
    const dir = await home('dsh-ai-loader-')
    await expect(loadModelsDevRegistry({
      url: 'https://registry.test/api.json',
      cachePath: join(dir, 'absent.json'),
      fetchImpl: okFetch({}),
    })).resolves.toEqual({})
  })

  it('treats an unreadable cache as absent and fails loud without the network', async () => {
    const dir = await home('dsh-ai-loader-')
    await writeFile(join(dir, 'cache.json'), 'not json at all')
    await expect(loadModelsDevRegistry({
      url: 'https://registry.test/api.json',
      cachePath: join(dir, 'cache.json'),
      fetchImpl: deadFetch,
    })).rejects.toThrow(/could not load the models.dev catalog/)
  })

  it('treats a cache whose entries are not providers as absent', async () => {
    const dir = await home('dsh-ai-loader-')
    await writeFile(join(dir, 'cache.json'), JSON.stringify({ a: 5 }))
    await expect(loadModelsDevRegistry({
      url: 'https://registry.test/api.json',
      cachePath: join(dir, 'cache.json'),
      fetchImpl: deadFetch,
    })).rejects.toThrow(/could not load the models.dev catalog/)
  })

  it('places the default cache under the harness home storages', () => {
    expect(defaultCachePath()).toContain(join('storages', 'models-dev-cache.json'))
  })

  it('defaults the endpoint and cache path when given neither', async () => {
    const dir = await home('dsh-ai-loader-')
    vi.stubEnv('DSH_HOME', dir)
    try {
      const registry = { acme: provider() }
      // Only the fetch is injected; the endpoint and cache defaults apply, and
      // the cache lands under the stubbed harness home.
      await expect(loadModelsDevRegistry({ fetchImpl: okFetch(registry) })).resolves.toEqual(registry)
      const cached = JSON.parse(
        await readFile(join(dir, 'storages', 'models-dev-cache.json'), 'utf8'),
      )
      expect(cached).toEqual(registry)
    } finally {
      vi.unstubAllEnvs()
    }
  })
})

describe('snapshot mapping', () => {
  it('maps provider and model metadata for servable providers', () => {
    const catalog = catalogFromSnapshot(fixtureRegistry())

    expect(catalog.providers().map(provider => provider.id))
      .toEqual(['deepseek', 'visionai', 'sketchy', 'effortai', 'capless', 'endpointless', 'bare', 'anthropic', 'google'])
    const deepseek = catalog.provider('deepseek')
    expect(deepseek).toMatchObject({
      id: 'deepseek',
      displayName: 'DeepSeek',
      baseUrl: 'https://api.deepseek.example',
    })
    // The empty model id never reaches the catalog.
    expect([...deepseek?.models.keys() ?? []]).toEqual(['deepseek-chat', 'deepseek-reasoner'])
    expect(deepseek?.models.get('deepseek-chat')).toMatchObject({
      id: 'deepseek-chat',
      name: 'DeepSeek Chat',
      reasoning: false,
      input: ['text'],
      contextWindow: 65_536,
      maxTokens: 4096,
    })
    expect(deepseek?.models.get('deepseek-reasoner')?.reasoning).toBe(true)
  })

  it('records image modality from the registry', () => {
    const vision = catalogFromSnapshot(fixtureRegistry()).provider('visionai')?.models.get('vision-large')
    expect(vision?.input).toEqual(['text', 'image'])
  })

  it('maps models for a provider with no base URL; the route must declare one', () => {
    const catalog = catalogFromSnapshot(fixtureRegistry())
    const endpointless = catalog.provider('endpointless')
    expect(endpointless?.displayName).toBe('endpointless')
    expect(endpointless?.baseUrl).toBeUndefined()
    expect([...endpointless?.models.keys() ?? []]).toEqual(['endpointless-mini'])
    expect(catalog.providers().map(provider => provider.id)).toContain('endpointless')
  })

  it('marks the protocol families the one protocol cannot serve', () => {
    const catalog = catalogFromSnapshot(fixtureRegistry())
    expect(catalog.provider('anthropic')).toMatchObject({ withheldProtocol: 'anthropic-messages' })
    expect(catalog.provider('google')).toMatchObject({ withheldProtocol: 'google-generative-ai' })
    // Mapping stays faithful: the models are present, and route registration
    // is where the protocol refusal names the route.
    expect(catalog.provider('anthropic')?.models.size).toBe(1)
  })

  it('floors garbage fields to honest values', () => {
    const sketchy = catalogFromSnapshot(fixtureRegistry()).provider('sketchy')?.models
    // An empty name falls back to the id; non-array modalities floor to text;
    // unusable capacities stay absent.
    const odd = sketchy?.get('sketchy-odd')
    expect(odd?.name).toBe('sketchy-odd')
    expect(odd?.input).toEqual(['text'])
    expect(odd?.contextWindow).toBeUndefined()
    expect(odd?.maxTokens).toBeUndefined()
    // Fractional capacities are unusable, not rounded.
    expect(sketchy?.get('sketchy-fractional')?.contextWindow).toBeUndefined()
  })

  it('maps reasoning_options effort values to a canonical level set', () => {
    const graded = catalogFromSnapshot(fixtureRegistry()).provider('effortai')?.models.get('effort-graded')
    // `none` becomes a valueless off; every other level spells as itself.
    expect(graded?.reasoningLevels).toEqual(new Map([
      ['off', null],
      ['low', 'low'],
      ['high', 'high'],
      ['max', 'max'],
    ]))
  })

  it('filters non-canonical values out of the level set without failing the entry', () => {
    const partial = catalogFromSnapshot(fixtureRegistry()).provider('effortai')?.models.get('effort-partial')
    expect(partial?.reasoningLevels).toEqual(new Map([['medium', 'medium']]))
    expect(partial?.reasoning).toBe(true)
  })

  it('states no level set for toggle, empty, absent, non-array, valueless, and unknown-only options', () => {
    const models = catalogFromSnapshot(fixtureRegistry()).provider('effortai')?.models
    for (const id of ['effort-toggle', 'effort-empty', 'effort-absent', 'effort-garbage', 'effort-valueless', 'effort-unknown']) {
      expect(models?.get(id)?.reasoningLevels).toBeUndefined()
    }
  })

  it('maps a model with no output limit as having no cap', () => {
    const capless = catalogFromSnapshot(fixtureRegistry()).provider('capless')?.models.get('capless-one')
    expect(capless).toMatchObject({ contextWindow: 8192 })
    expect(capless?.maxTokens).toBeUndefined()
  })

  it('maps a provider with no models key to an empty catalog', () => {
    const bare = catalogFromSnapshot(fixtureRegistry()).provider('bare')
    expect(bare?.models.size).toBe(0)
  })

  it('answers undefined for a provider the snapshot does not describe', () => {
    expect(catalogFromSnapshot(fixtureRegistry()).provider('missing')).toBeUndefined()
  })
})
