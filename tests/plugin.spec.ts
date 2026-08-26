import { readFile, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import LlmRuntime, { LlmAdapter } from '@deepseek-ai/dsh-llm'
import FileSettingsProvider from '@deepseek-ai/dsh-settings-file'
import { settingsNamespace } from '@deepseek-ai/dsh-settings'
import * as LlmAi from '../src/index.ts'
import { LlmAiAdapter } from '../src/index.ts'
import { supportedProtocols } from '../src/provider.ts'
import { catalogFromSnapshot } from '../src/modelsdev.ts'
import { resolveProfiles } from '../src/config.ts'
import { fixtureRegistry, home, registryServer } from './registry.ts'
import type { RegistryServer } from './registry.ts'

const NS = settingsNamespace('llm-ai')

/** Minimal foreign adapter: only needs to own a route this plugin then wants. */
class StubAdapter extends LlmAdapter {
  override providerInfo(provider: string) {
    return { id: provider, name: 'Stub' }
  }

  override async * stream(): AsyncIterable<never> {
    throw new Error('stub adapter must never stream')
  }
}

const cleanups: Array<() => Promise<void>> = []

afterEach(async () => {
  while (cleanups.length > 0) await cleanups.pop()!()
})

/** Boot the runtime plus this plugin against a loopback registry snapshot. */
async function harness(
  config: LlmAi.Config = {},
  registry: unknown = fixtureRegistry(),
): Promise<{ ctx: Context; server: RegistryServer }> {
  const server = await registryServer(registry)
  cleanups.push(server.close)
  const ctx = new Context()
  cleanups.push(async () => {
    await ctx.fiber.dispose()
  })
  await ctx.plugin(LlmRuntime)
  await ctx.plugin(LlmAi, {
    catalogUrl: server.url,
    catalogCachePath: join(await home('dsh-ai-plugin-'), 'cache.json'),
    ...config,
  })
  return { ctx, server }
}

/** Boot with a real settings document, as the product composes it. */
async function settingsHarness(config: LlmAi.Config = {}): Promise<Context> {
  const server = await registryServer(fixtureRegistry())
  cleanups.push(server.close)
  const dir = await home('dsh-ai-settings-')
  await writeFile(join(dir, 'settings.yaml'), '')
  const ctx = new Context()
  cleanups.push(async () => {
    await ctx.fiber.dispose()
  })
  await ctx.plugin(LlmRuntime)
  await ctx.plugin(FileSettingsProvider, { path: join(dir, 'settings.yaml'), watch: false })
  await ctx.plugin(LlmAi, {
    catalogUrl: server.url,
    catalogCachePath: join(dir, 'cache.json'),
    ...config,
  })
  return ctx
}

describe('dormant mounting', () => {
  it('registers no routes with providers omitted, keeping the directory configurable', async () => {
    const { ctx } = await harness()

    expect(ctx.llm.listProviders()).toEqual([])
    const directory = ctx.llm.listConfigurableProviders()
    expect(directory).toContainEqual({
      provider: 'deepseek',
      displayName: 'DeepSeek',
      settingsNs: 'llm-ai',
      settingsPath: ['providers', 'deepseek'],
      declared: false,
    })
  })

  it('mounts dormant with an explicitly empty providers dict too', async () => {
    const { ctx } = await harness({ providers: {} })
    expect(ctx.llm.listProviders()).toEqual([])
  })

  it('declares withheld protocol families in the directory rather than hiding them', async () => {
    const { ctx } = await harness()
    const directory = ctx.llm.listConfigurableProviders()
    expect(directory).toContainEqual({
      provider: 'anthropic',
      displayName: 'Anthropic',
      settingsNs: 'llm-ai',
      settingsPath: ['providers', 'anthropic'],
      declared: false,
    })
    expect(directory.find(entry => entry.provider === 'google')).toBeDefined()
  })

  it('mounts bare against an empty registry with no directory entries', async () => {
    const { ctx } = await harness({}, {})
    expect(ctx.llm.listProviders()).toEqual([])
    expect(ctx.llm.listConfigurableProviders()).toEqual([])
  })
})

describe('registry routes', () => {
  it('serves a registry provider from its snapshot with only a credential reference', async () => {
    const { ctx } = await harness({ providers: { deepseek: { apiKeyEnv: 'DEEPSEEK_API_KEY' } } })

    expect(ctx.llm.listProviders()).toEqual([{ id: 'deepseek', name: 'deepseek' }])
    expect(await ctx.llm.listModels('deepseek')).toEqual([
      { provider: 'deepseek', id: 'deepseek-chat', name: 'DeepSeek Chat', inputModalities: ['text'] },
      { provider: 'deepseek', id: 'deepseek-reasoner', name: 'DeepSeek Reasoner', inputModalities: ['text'] },
    ])
  })

  it('answers identity, capacity, and reasoning metadata from the same resolution', async () => {
    const { ctx } = await harness({ providers: { deepseek: {} } })

    expect(await ctx.llm.resolveModelInfo('deepseek', 'deepseek-chat')).toEqual({
      provider: 'deepseek',
      id: 'deepseek-chat',
      name: 'DeepSeek Chat',
      inputModalities: ['text'],
      context: { contextWindow: 65_536 },
    })
    // A registry capability sizes the model; it never becomes a request default.
    expect((await ctx.llm.resolveModelInfo('deepseek', 'deepseek-chat')).defaultMaxTokens).toBeUndefined()

    expect(await ctx.llm.resolveModelInfo('deepseek', 'deepseek-reasoner')).toMatchObject({
      context: { contextWindow: 65_536 },
      reasoning: {
        efforts: [
          { id: 'off', name: 'Off' },
          { id: 'low', name: 'Low' },
          { id: 'medium', name: 'Medium' },
          { id: 'high', name: 'High' },
        ],
      },
    })
  })

  it('keeps image modality from the registry', async () => {
    const { ctx } = await harness({ providers: { visionai: {} } })
    expect((await ctx.llm.resolveModelInfo('visionai', 'vision-large')).inputModalities).toEqual(['text', 'image'])
  })

  it('shows a configured displayName to selectors', async () => {
    const { ctx } = await harness({
      providers: { deepseek: { displayName: 'Prod DeepSeek' } },
    })
    expect(ctx.llm.listProviders()).toEqual([{ id: 'deepseek', name: 'Prod DeepSeek' }])
  })

  it('refuses a route naming a model it does not serve', async () => {
    const { ctx } = await harness({ providers: { deepseek: {} } })
    await expect(ctx.llm.resolveModelInfo('deepseek', 'nope')).rejects.toMatchObject({
      failure: { code: 'UNKNOWN_MODEL' },
    })
  })
})

describe('models list replacement', () => {
  it('replaces the route catalog, defaulting unset fields from the registry entry', async () => {
    const { ctx } = await harness({
      providers: {
        deepseek: {
          models: [
            { id: 'deepseek-chat', maxTokens: 2048 },
            { id: 'brand-new', contextWindow: 999_999, maxTokens: 1234 },
          ],
        },
      },
    })

    expect((await ctx.llm.listModels('deepseek')).map(model => model.id)).toEqual(['deepseek-chat', 'brand-new'])
    const chat = await ctx.llm.resolveModelInfo('deepseek', 'deepseek-chat')
    // The entry declared only maxTokens; context came from the registry entry.
    expect(chat).toMatchObject({ context: { contextWindow: 65_536 }, defaultMaxTokens: 2048 })
    const fresh = await ctx.llm.resolveModelInfo('deepseek', 'brand-new')
    expect(fresh).toMatchObject({ context: { contextWindow: 999_999 }, defaultMaxTokens: 1234 })
  })

  it('applies the route capacity fallbacks only to configured entries', async () => {
    const { ctx } = await harness({
      providers: {
        sketchy: {
          defaultContextWindow: 111_111,
          defaultMaxTokens: 2_222,
          models: [{ id: 'sketchy-unsized' }],
        },
      },
    })
    // sketchy-unsized has no registry context and the entry states none, so
    // the route fallback answers — unlike the bare route, which refuses.
    expect(await ctx.llm.resolveModelInfo('sketchy', 'sketchy-unsized')).toMatchObject({
      context: { contextWindow: 111_111 },
    })
    // A fallback capability is not a request default; only a stated entry value is.
    expect((await ctx.llm.resolveModelInfo('sketchy', 'sketchy-unsized')).defaultMaxTokens).toBeUndefined()
  })

  it('serves a registry model with no output cap as having none', async () => {
    const { ctx } = await harness({ providers: { capless: {} } })
    expect(await ctx.llm.resolveModelInfo('capless', 'capless-one')).toMatchObject({
      context: { contextWindow: 8192 },
    })
    // Nothing anywhere stated an output cap, so none is reported; the route's
    // fallback would be a guess and the model was not configured.
    expect((await ctx.llm.resolveModelInfo('capless', 'capless-one')).defaultMaxTokens).toBeUndefined()
  })

  it('falls back to the route defaultMaxTokens only for a configured entry', async () => {
    const { ctx } = await harness({
      providers: {
        capless: {
          defaultMaxTokens: 5_555,
          models: [{ id: 'capless-one' }],
        },
      },
    })
    // The fallback sizes the model but stays out of request defaults.
    expect((await ctx.llm.resolveModelInfo('capless', 'capless-one')).defaultMaxTokens).toBeUndefined()
  })

  it('refuses a model entry with an unusable capacity at the config boundary', async () => {
    await expect(harness({
      providers: { deepseek: { models: [{ id: 'm', contextWindow: 0 }] } },
    })).rejects.toThrow(/invalid config/)
    await expect(harness({
      providers: { deepseek: { models: [{ id: 'm', maxTokens: 2.5 }] } },
    })).rejects.toThrow(/invalid config/)
  })

  it('refuses an empty or duplicated model id', async () => {
    await expect(harness({
      providers: { deepseek: { models: [{ id: '', contextWindow: 10, maxTokens: 10 }] } },
    })).rejects.toThrow(/model with an empty id/)
    await expect(harness({
      providers: {
        deepseek: { models: [{ id: 'a', contextWindow: 10, maxTokens: 10 }, { id: 'a' }] },
      },
    })).rejects.toThrow(/lists model "a" more than once/)
  })
})

describe('modelOverrides', () => {
  it('reshapes one model while the rest of the catalog keeps serving', async () => {
    const { ctx } = await harness({
      providers: { deepseek: { modelOverrides: { 'deepseek-chat': { maxTokens: 2048 } } } },
    })

    expect((await ctx.llm.listModels('deepseek')).map(model => model.id))
      .toEqual(['deepseek-chat', 'deepseek-reasoner'])
    expect(await ctx.llm.resolveModelInfo('deepseek', 'deepseek-chat')).toMatchObject({
      context: { contextWindow: 65_536 },
      defaultMaxTokens: 2048,
    })
  })

  it('refuses an override naming a model the registry does not describe', async () => {
    await expect(harness({
      providers: { deepseek: { modelOverrides: { ghost: {} } } },
    })).rejects.toThrow(/modelOverrides names "ghost", which the registry does not describe/)
  })

  it('refuses an override beside a models list', async () => {
    await expect(harness({
      providers: {
        deepseek: {
          models: [{ id: 'deepseek-chat' }],
          modelOverrides: { 'deepseek-chat': { maxTokens: 1 } },
        },
      },
    })).rejects.toThrow(/beside a models list/)
  })

  it('refuses an override on a route the registry does not describe', async () => {
    await expect(harness({
      providers: {
        'acme-gateway': {
          baseURL: 'https://gateway.acme.example/v1',
          modelOverrides: { 'acme-large': {} },
        },
      },
    })).rejects.toThrow(/the registry does not describe this route/)
  })

  it('refuses an override carrying its own id', async () => {
    await expect(harness({
      providers: { deepseek: { modelOverrides: { 'deepseek-chat': { id: 'renamed' } as never } } },
    })).rejects.toThrow(/sets "id", which is the dict key/)
  })

  it('refuses an override under an empty model id', async () => {
    await expect(harness({
      providers: { deepseek: { modelOverrides: { '': {} } } },
    })).rejects.toThrow(/empty model id/)
  })
})

describe('hand-declared routes', () => {
  function gateway(overrides: Record<string, unknown> = {}): LlmAi.Config {
    return {
      providers: {
        'acme-gateway': {
          displayName: 'Acme Gateway',
          apiKeyEnv: 'ACME_GATEWAY_API_KEY',
          baseURL: 'https://gateway.acme.example/v1',
          models: [
            { id: 'acme-large', name: 'Acme Large', contextWindow: 65_536, maxTokens: 4096 },
            { id: 'acme-think', name: 'Acme Think', contextWindow: 262_144, maxTokens: 32_768 },
          ],
          ...overrides,
        },
      },
    }
  }

  it('serves a route the registry has never heard of from its own declaration', async () => {
    const { ctx } = await harness(gateway())

    expect(ctx.llm.listProviders()).toEqual([{ id: 'acme-gateway', name: 'Acme Gateway' }])
    expect(await ctx.llm.listModels('acme-gateway')).toEqual([
      { provider: 'acme-gateway', id: 'acme-large', name: 'Acme Large', inputModalities: ['text'] },
      { provider: 'acme-gateway', id: 'acme-think', name: 'Acme Think', inputModalities: ['text'] },
    ])
  })

  it('joins the configurable-provider directory marked declared', async () => {
    const { ctx } = await harness(gateway())
    expect(ctx.llm.listConfigurableProviders()).toContainEqual({
      provider: 'acme-gateway',
      displayName: 'Acme Gateway',
      settingsNs: 'llm-ai',
      settingsPath: ['providers', 'acme-gateway'],
      declared: true,
    })
  })

  it('refuses a declared route with no models', async () => {
    await expect(harness({
      providers: { 'acme-gateway': { baseURL: 'https://gateway.acme.example/v1' } },
    })).rejects.toThrow(/resolves no models/)
  })

  it('refuses a registry provider with no endpoint unless baseURL is declared', async () => {
    await expect(harness({
      providers: { endpointless: {} },
    })).rejects.toThrow(/model "endpointless-mini" needs a baseURL/)
    const { ctx } = await harness({
      providers: { endpointless: { baseURL: 'https://proxy.example/v1' } },
    })
    expect((await ctx.llm.listModels('endpointless')).map(model => model.id)).toEqual(['endpointless-mini'])
  })
})

describe('capacity refusals', () => {
  it('refuses a registry model with no context window, naming route and model', async () => {
    await expect(harness({ providers: { sketchy: {} } })).rejects.toThrow(
      /provider "sketchy" model "sketchy-unsized" has no context window in the models.dev registry/,
    )
  })
})

describe('reasoning declarations', () => {
  it('reshapes the offered levels and keeps off valueless', async () => {
    const { ctx } = await harness({
      providers: {
        deepseek: {
          models: [{ id: 'deepseek-chat', reasoningEfforts: { off: null, high: 'high', max: 'ultra' } }],
        },
      },
    })
    expect(await ctx.llm.resolveModelInfo('deepseek', 'deepseek-chat')).toMatchObject({
      reasoning: {
        efforts: [
          { id: 'off', name: 'Off' },
          { id: 'high', name: 'High' },
          { id: 'max', name: 'Max' },
        ],
      },
    })
  })

  it('strips reasoning with an explicit false', async () => {
    const { ctx } = await harness({
      providers: {
        deepseek: {
          models: [{ id: 'deepseek-reasoner', reasoningEfforts: false }],
        },
      },
    })
    expect((await ctx.llm.resolveModelInfo('deepseek', 'deepseek-reasoner')).reasoning).toBeUndefined()
  })

  it('offers the route-configured default level when the model supports it', async () => {
    const { ctx } = await harness({ providers: { deepseek: { reasoning: 'high' } } })
    expect(await ctx.llm.resolveModelInfo('deepseek', 'deepseek-reasoner')).toMatchObject({
      reasoning: { defaultEffort: 'high' },
    })
    // A model that does not offer the configured level describes none rather
    // than failing: describing capability must not throw on a mis-set knob.
    const declared = await harness({
      providers: {
        deepseek: {
          reasoning: 'xhigh',
          models: [{ id: 'deepseek-chat', reasoningEfforts: { low: 'low' } }],
        },
      },
    })
    expect((await declared.ctx.llm.resolveModelInfo('deepseek', 'deepseek-chat')).reasoning).toMatchObject({
      efforts: [{ id: 'low', name: 'Low' }],
    })
  })

  it('refuses an empty, valueless, or empty-string declaration', async () => {
    await expect(harness({
      providers: { deepseek: { models: [{ id: 'm', reasoningEfforts: {} }] } },
    })).rejects.toThrow(/empty reasoningEfforts/)
    await expect(harness({
      providers: { deepseek: { models: [{ id: 'm', reasoningEfforts: { low: null } }] } },
    })).rejects.toThrow(/reasoningEfforts.low needs the wire value/)
    await expect(harness({
      providers: { deepseek: { models: [{ id: 'm', reasoningEfforts: { high: '' } }] } },
    })).rejects.toThrow(/must not be an empty string/)
    await expect(harness({
      providers: { deepseek: { models: [{ id: 'm', reasoningEfforts: { off: null } }] } },
    })).rejects.toThrow(/offers no level beyond "off"/)
  })
})

describe('protocol table', () => {
  it('exposes exactly one supported protocol', () => {
    expect(supportedProtocols()).toEqual(['openai-completions'])
  })

  it('refuses a route naming a protocol this build does not serve at the config boundary', async () => {
    await expect(harness({
      providers: { deepseek: { api: 'anthropic-messages' } },
    })).rejects.toThrow(/expected "openai-completions"/)
  })

  it('refuses a withheld registry provider naming the protocol it needs', async () => {
    await expect(harness({
      providers: { anthropic: { apiKeyEnv: 'ANTHROPIC_API_KEY' } },
    })).rejects.toThrow(
      /provider "anthropic" needs wire protocol "anthropic-messages", which this adapter does not serve/,
    )
  })
})

describe('registration semantics', () => {
  it('fails DUPLICATE_ADAPTER on a route another adapter owns, leaving it serving', async () => {
    const server = await registryServer(fixtureRegistry())
    cleanups.push(server.close)
    const ctx = new Context()
    cleanups.push(async () => {
      await ctx.fiber.dispose()
    })
    await ctx.plugin(LlmRuntime)
    ctx.llm.registerAdapter(['deepseek'], new StubAdapter())

    await expect(ctx.plugin(LlmAi, {
      catalogUrl: server.url,
      catalogCachePath: join(await home('dsh-ai-plugin-'), 'cache.json'),
      providers: { deepseek: {} },
    })).rejects.toMatchObject({ failure: { code: 'DUPLICATE_ADAPTER' } })

    // All-or-nothing: the refused registration left the stub owning the route.
    expect(ctx.llm.listProviders()).toEqual([{ id: 'deepseek', name: 'Stub' }])
  })

  it('fails loud at load when the catalog cannot be fetched and nothing is cached', async () => {
    const ctx = new Context()
    cleanups.push(async () => {
      await ctx.fiber.dispose()
    })
    await ctx.plugin(LlmRuntime)
    await expect(ctx.plugin(LlmAi, {
      // Port 1 on loopback refuses every connection.
      catalogUrl: 'http://127.0.0.1:1/api.json',
      catalogCachePath: join(await home('dsh-ai-plugin-'), 'absent.json'),
    })).rejects.toThrow(/could not load the models.dev catalog/)
  })

  it('boots from the disk cache when the network is down', async () => {
    const cachePath = join(await home('dsh-ai-plugin-'), 'cache.json')
    await writeFile(cachePath, JSON.stringify(fixtureRegistry()))
    const ctx = new Context()
    cleanups.push(async () => {
      await ctx.fiber.dispose()
    })
    await ctx.plugin(LlmRuntime)
    await ctx.plugin(LlmAi, {
      catalogUrl: 'http://127.0.0.1:1/api.json',
      catalogCachePath: cachePath,
      providers: { deepseek: {} },
    })
    expect((await ctx.llm.listModels('deepseek')).map(model => model.id)).toEqual(['deepseek-chat', 'deepseek-reasoner'])
  })

  it('defaults the catalog endpoint and cache path when the entry names neither', async () => {
    const dir = await home('dsh-ai-defaults-')
    vi.stubEnv('DSH_HOME', dir)
    vi.stubGlobal('fetch', async () => new Response(JSON.stringify(fixtureRegistry()), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    }))
    try {
      const ctx = new Context()
      cleanups.push(async () => {
        await ctx.fiber.dispose()
      })
      await ctx.plugin(LlmRuntime)
      await ctx.plugin(LlmAi, { providers: { deepseek: {} } })
      expect((await ctx.llm.listModels('deepseek')).map(model => model.id))
        .toEqual(['deepseek-chat', 'deepseek-reasoner'])
      // The default cache landed under the stubbed harness home.
      expect(JSON.parse(await readFile(join(dir, 'storages', 'models-dev-cache.json'), 'utf8')))
        .toEqual(fixtureRegistry())
    } finally {
      vi.unstubAllEnvs()
      vi.unstubAllGlobals()
    }
  })
})

describe('settings section', () => {
  it('registers routes live from the llm-ai namespace and drops them when it empties', async () => {
    const ctx = await settingsHarness()

    expect(ctx.llm.listProviders()).toEqual([])
    await ctx.settings.update(NS, { providers: { deepseek: { apiKeyEnv: 'DEEPSEEK_API_KEY' } } })
    expect(ctx.llm.listProviders()).toEqual([{ id: 'deepseek', name: 'deepseek' }])
    expect((await ctx.llm.listModels('deepseek')).map(model => model.id)).toEqual(['deepseek-chat', 'deepseek-reasoner'])

    // Emptying the user layer returns the adapter to its dormant state.
    await ctx.settings.replace(NS, {})
    expect(ctx.llm.listProviders()).toEqual([])
  })

  it('refuses an unserviceable section where it is written, keeping routes serving', async () => {
    const ctx = await settingsHarness({ providers: { deepseek: {} } })

    await expect(ctx.settings.update(NS, { providers: { sketchy: {} } })).rejects.toThrow(
      /has no context window in the models.dev registry/,
    )
    expect(ctx.llm.listProviders()).toEqual([{ id: 'deepseek', name: 'deepseek' }])
  })

  it('keeps the previously registered routes when a settings swap collides with another adapter', async () => {
    const ctx = await settingsHarness({ providers: { deepseek: {} } })
    ctx.llm.registerAdapter(['acme-gateway'], new StubAdapter())

    // The write is schema-valid and serviceable; only the registry swap sees
    // the collision, contains it, and keeps the previous routes serving.
    await ctx.settings.update(NS, {
      providers: {
        deepseek: {},
        'acme-gateway': {
          baseURL: 'https://gateway.acme.example/v1',
          models: [{ id: 'acme-large', contextWindow: 100, maxTokens: 10 }],
        },
      },
    })
    // This plugin's swap was refused whole, so it still owns only deepseek;
    // the stub keeps the route it already had.
    expect(ctx.llm.listProviders()).toEqual([
      { id: 'deepseek', name: 'deepseek' },
      { id: 'acme-gateway', name: 'Stub' },
    ])

    // A later working configuration re-applies over the refused swap.
    await ctx.settings.replace(NS, { providers: { deepseek: {}, visionai: {} } })
    expect(ctx.llm.listProviders()).toEqual(expect.arrayContaining([
      { id: 'deepseek', name: 'deepseek' },
      { id: 'acme-gateway', name: 'Stub' },
      { id: 'visionai', name: 'visionai' },
    ]))
    expect(ctx.llm.listProviders()).toHaveLength(3)
  })

  it('joins a settings-born declared route into the directory', async () => {
    const ctx = await settingsHarness()

    await ctx.settings.update(NS, {
      providers: {
        'acme-gateway': {
          baseURL: 'https://gateway.acme.example/v1',
          models: [{ id: 'acme-large', contextWindow: 100, maxTokens: 10 }],
        },
      },
    })
    expect(ctx.llm.listConfigurableProviders()).toContainEqual({
      provider: 'acme-gateway',
      displayName: 'acme-gateway',
      settingsNs: 'llm-ai',
      settingsPath: ['providers', 'acme-gateway'],
      declared: true,
    })
    // Resetting the user layer withdraws the declared entry again.
    await ctx.settings.replace(NS, {})
    expect(ctx.llm.listConfigurableProviders().find(entry => entry.provider === 'acme-gateway')).toBeUndefined()
  })

  it('keeps the previous configurable-provider directory when a swap collides with another registration', async () => {
    const ctx = await settingsHarness({ providers: { deepseek: {} } })
    ctx.llm.registerConfigurableProviders([{
      provider: 'acme-gateway',
      displayName: 'Foreign',
      settingsNs: 'other-ns',
      settingsPath: ['providers', 'acme-gateway'],
    }])

    // The route registers (no adapter owns it), but the directory swap sees
    // the foreign declaration, contains the refusal, and keeps the previous
    // entries serving.
    await ctx.settings.update(NS, {
      providers: {
        'acme-gateway': {
          baseURL: 'https://gateway.acme.example/v1',
          models: [{ id: 'acme-large', contextWindow: 100, maxTokens: 10 }],
        },
      },
    })
    expect(ctx.llm.listProviders().map(provider => provider.id)).toEqual(['deepseek', 'acme-gateway'])
    expect(ctx.llm.listConfigurableProviders().find(entry => entry.provider === 'acme-gateway'))
      .toMatchObject({ displayName: 'Foreign', settingsNs: 'other-ns' })
  })

  it('re-registers when a displayName changes', async () => {
    const ctx = await settingsHarness({ providers: { deepseek: {} } })
    await ctx.settings.update(NS, { providers: { deepseek: { displayName: 'Renamed' } } })
    expect(ctx.llm.listProviders()).toEqual([{ id: 'deepseek', name: 'Renamed' }])
  })
})

describe('adapter boundary', () => {
  // These run against the exported adapter class directly: the llm runtime
  // guards ownership before the adapter sees the call, so its own refusals
  // are only reachable from a direct registration.
  const catalog = catalogFromSnapshot(fixtureRegistry())

  it('refuses providers and models it does not own', async () => {
    const adapter = new LlmAiAdapter({ profiles: () => new Map(), resolveApiKey: () => undefined })
    expect(adapter.providerInfo('ghost')).toEqual({ id: 'ghost', name: 'ghost' })
    await expect(adapter.listModels('ghost')).rejects.toMatchObject({ failure: { code: 'NO_ADAPTER' } })
    await expect(adapter.resolveModel('ghost', 'm')).rejects.toMatchObject({ failure: { code: 'NO_ADAPTER' } })
  })

  it('refuses an unknown model on an owned route', async () => {
    const profiles = resolveProfiles({ deepseek: {} }, catalog)
    const adapter = new LlmAiAdapter({ profiles: () => profiles, resolveApiKey: () => undefined })
    await expect(adapter.resolveModel('deepseek', 'nope')).rejects.toMatchObject({
      failure: { code: 'UNKNOWN_MODEL' },
    })
  })
})
