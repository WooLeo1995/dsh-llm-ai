import { describe, expect, it } from 'vitest'
import { MAX_TIMER_DELAY_MS } from '@deepseek-ai/dsh-timeout'
import { catalogFromSnapshot } from '../src/modelsdev.ts'
import { Config, resolveProfiles } from '../src/config.ts'
import type { LlmAiProviderProfile } from '../src/config.ts'
import { fixtureRegistry } from './registry.ts'

const catalog = catalogFromSnapshot(fixtureRegistry())

/** Resolve raw profiles against the fixture registry, throwing the refusal verbatim. */
function resolve(providers: Record<string, LlmAiProviderProfile> | undefined) {
  return resolveProfiles(providers, catalog)
}

describe('Config schema', () => {
  it('defaults the providers dict so an absent section is the dormant posture', () => {
    expect(Config({})).toEqual({ providers: {} })
  })

  it('passes the catalog knobs through', () => {
    expect(Config({ catalogUrl: 'https://mirror.test/api.json', catalogCachePath: '/tmp/c.json' })).toMatchObject({
      catalogUrl: 'https://mirror.test/api.json',
      catalogCachePath: '/tmp/c.json',
      providers: {},
    })
  })
})

describe('profile resolution refusals', () => {
  it('refuses an array of profiles naming the dict shape', () => {
    expect(() => resolve([] as unknown as Record<string, LlmAiProviderProfile>))
      .toThrow(/providers is a dict keyed by provider route/)
  })

  it('refuses an empty route key', () => {
    expect(() => resolve({ '': {} })).toThrow(/provider names must be non-empty/)
  })

  it('refuses an empty baseURL', () => {
    expect(() => resolve({ deepseek: { baseURL: '' } })).toThrow(/provider "deepseek" has an empty baseURL/)
  })

  it('refuses an empty displayName', () => {
    expect(() => resolve({ deepseek: { displayName: '' } })).toThrow(/provider "deepseek" has an empty displayName/)
  })

  it('refuses an unusable streamIdleTimeoutMs', () => {
    expect(() => resolve({ deepseek: { streamIdleTimeoutMs: 0 } })).toThrow(/streamIdleTimeoutMs/)
    expect(() => resolve({ deepseek: { streamIdleTimeoutMs: Number.POSITIVE_INFINITY } })).toThrow(/streamIdleTimeoutMs/)
    expect(() => resolve({ deepseek: { streamIdleTimeoutMs: MAX_TIMER_DELAY_MS + 1 } })).toThrow(/streamIdleTimeoutMs/)
  })

  it('refuses an unusable maxRequestImageBytes', () => {
    expect(() => resolve({ deepseek: { maxRequestImageBytes: 0 } })).toThrow(/maxRequestImageBytes/)
    expect(() => resolve({ deepseek: { maxRequestImageBytes: 1.5 } })).toThrow(/maxRequestImageBytes/)
  })

  it('refuses an empty defaultInput', () => {
    expect(() => resolve({ deepseek: { defaultInput: [] } })).toThrow(/defaultInput must name at least one modality/)
  })

  it('refuses a compat key no wire protocol declares', () => {
    expect(() => resolve({ deepseek: { compat: { supportsStore: true } as never } }))
      .toThrow(/sets compat "supportsStore", which no wire protocol declares/)
  })

  it('accepts a compat whose every key the wire protocol declares', () => {
    const compat = { maxTokensField: 'max_tokens', supportsDeveloperRole: true, thinkingFormat: 'openai' } as const
    const profile = resolve({ deepseek: { compat } }).get('deepseek')
    expect(profile?.compat).toEqual(compat)
  })

  it('refuses an api the adapter does not serve', () => {
    expect(() => resolve({ deepseek: { api: 'anthropic-messages' } }))
      .toThrow(/names api "anthropic-messages", which this adapter does not serve/)
  })

  it('refuses an invalid credential reference', () => {
    expect(() => resolve({ deepseek: { apiKeyEnv: 'not a ref!' } })).toThrow()
  })
})

describe('profile resolution results', () => {
  it('resolves an omitted dict to the dormant route set', () => {
    expect([...resolve(undefined).keys()]).toEqual([])
  })

  it('stamps the route defaults and the one protocol on a bare registry route', () => {
    const profile = resolve({ deepseek: { apiKeyEnv: 'DEEPSEEK_API_KEY' } }).get('deepseek')
    expect(profile).toMatchObject({
      provider: 'deepseek',
      displayName: 'deepseek',
      api: 'openai-completions',
      apiKeyEnv: 'DEEPSEEK_API_KEY',
      streamIdleTimeoutMs: 300_000,
      maxRequestImageBytes: 20 * 1024 * 1024,
    })
    expect(profile?.retryPolicy).toMatchObject({ mode: 'normal' })
    expect(profile?.models.map(model => model.id)).toEqual(['deepseek-chat', 'deepseek-reasoner'])
    expect(profile?.baseURL).toBeUndefined()
  })

  it('honors an explicit displayName, baseURL, and headers detachment', () => {
    const headers = { 'x-deploy': 'a' }
    const profile = resolve({
      deepseek: { displayName: 'Prod DeepSeek', baseURL: 'https://proxy.example', headers },
    }).get('deepseek')
    expect(profile?.displayName).toBe('Prod DeepSeek')
    expect(profile?.baseURL).toBe('https://proxy.example')
    headers['x-deploy'] = 'mutated'
    expect(profile?.headers).toEqual({ 'x-deploy': 'a' })
    for (const model of profile?.models ?? []) expect(model.baseURL).toBe('https://proxy.example')
  })

  it('reads a models entry modalities as stated, absent, or empty', () => {
    const profile = resolve({
      visionai: {
        models: [
          // Stated: the entry's list wins.
          { id: 'vision-large', input: ['text'] },
          // Absent: the registry entry's modalities apply.
          { id: 'capless-one', contextWindow: 100 },
          // Empty: the same as absent, then the route default.
          { id: 'sketchy-unsized', input: [], contextWindow: 100 },
        ],
      },
    }).get('visionai')
    const inputs = profile?.models.map(model => [...model.input])
    expect(inputs).toEqual([['text'], ['text'], ['text']])
  })
})
