/**
 * Dynamic configuration and credentials at the `ctx.llm` seam: settings
 * sections merging per provider over the composition base, per-call credential
 * resolution through the credentials seam and the trusted environment, the
 * in-flight snapshot freeze that keeps one stream on the facts it started
 * with, and the refusals that make a mis-set credential visible and safe.
 * Loopback registry and wire servers only; no network.
 */

import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import LlmRuntime, { createUserMessage } from '@deepseek-ai/dsh-llm'
import type { GenerateOptions, LlmFailure, StreamChunk } from '@deepseek-ai/dsh-llm'
import { credentialRef } from '@deepseek-ai/dsh-credentials'
import { LocalCredentialProvider } from '@deepseek-ai/dsh-credentials-local'
import FileSettingsProvider from '@deepseek-ai/dsh-settings-file'
import { settingsNamespace } from '@deepseek-ai/dsh-settings'
import * as LlmAi from '../src/index.ts'
import { fixtureRegistry, registryServer } from './registry.ts'
import { mockServer, textEvents } from './mock-server.ts'

const NS = settingsNamespace('llm-ai')

const cleanups: Array<() => Promise<void>> = []

afterEach(async () => {
  vi.unstubAllEnvs()
  while (cleanups.length > 0) await cleanups.pop()!()
})

/** Boot the runtime, a settings document, optionally the local credential store, and this plugin. */
async function boot(
  dir: string,
  config: LlmAi.Config,
  credentials: 'mounted' | 'absent' = 'mounted',
): Promise<Context> {
  const registry = await registryServer(fixtureRegistry())
  cleanups.push(registry.close)
  const ctx = new Context()
  cleanups.push(async () => {
    await ctx.fiber.dispose()
  })
  await ctx.plugin(LlmRuntime)
  await ctx.plugin(FileSettingsProvider, { path: join(dir, 'settings.yaml'), watch: false })
  if (credentials === 'mounted') {
    await ctx.plugin(LocalCredentialProvider, { path: join(dir, '.credentials.yaml'), watch: false })
  }
  await ctx.plugin(LlmAi, {
    catalogUrl: registry.url,
    catalogCachePath: join(dir, 'cache.json'),
    ...config,
  })
  return ctx
}

const user = (text: string) => createUserMessage({
  content: [{ type: 'text', text }],
  source: { kind: 'user' },
})

/** Collect every chunk of one streamed call. */
async function chunksOf(ctx: Context, options: GenerateOptions): Promise<StreamChunk[]> {
  const collected: StreamChunk[] = []
  for await (const chunk of ctx.llm.stream(options)) collected.push(chunk)
  return collected
}

/** The terminal failure of one streamed call; refuses non-failure finishes. */
async function failureOf(ctx: Context, options: GenerateOptions): Promise<LlmFailure> {
  const collected = await chunksOf(ctx, options)
  const last = collected[collected.length - 1]
  if (last === undefined || last.type !== 'finish') throw new Error('expected a terminal finish chunk')
  if (last.reason.kind !== 'error') throw new Error(`expected an error finish, got ${last.reason.kind}`)
  return last.reason.failure
}

/** Resolve when the predicate holds, failing the test on timeout instead of hanging. */
async function waitFor(predicate: () => boolean): Promise<void> {
  const deadline = Date.now() + 5_000
  while (!predicate()) {
    if (Date.now() > deadline) throw new Error('expected condition was not reached')
    await new Promise(resolve => setTimeout(resolve, 5))
  }
}

/** The deepseek route's default call. */
const call: GenerateOptions = { provider: 'deepseek', model: 'deepseek-chat', messages: [user('hi')] }

/** A throwaway home holding the settings and credentials documents. */
async function freshHome(): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), 'dsh-ai-dynamic-'))
  cleanups.push(() => rm(dir, { recursive: true, force: true }))
  return dir
}

describe('credential resolution order', () => {
  it('resolves a named reference through the credentials seam, rotating per request', async () => {
    const dir = await freshHome()
    await writeFile(join(dir, '.credentials.yaml'), 'AI_ROTATE_KEY: pk-one\n', { mode: 0o600 })
    const server = await mockServer([{ kind: 'sse', events: textEvents }, { kind: 'sse', events: textEvents }])
    const ctx = await boot(dir, {
      providers: { deepseek: { apiKeyEnv: 'AI_ROTATE_KEY', baseURL: server.url } },
    })

    await chunksOf(ctx, call)
    expect(server.headers[0]?.authorization).toBe('Bearer pk-one')

    await ctx.credentials.set(credentialRef('AI_ROTATE_KEY'), 'pk-two')
    await chunksOf(ctx, call)
    expect(server.headers[1]?.authorization).toBe('Bearer pk-two')
  })

  it('trims a padded stored key before it reaches the header', async () => {
    const dir = await freshHome()
    await writeFile(join(dir, '.credentials.yaml'), 'AI_PADDED_KEY: "  padded-key  "\n', { mode: 0o600 })
    const server = await mockServer([{ kind: 'sse', events: textEvents }])
    const ctx = await boot(dir, {
      providers: { deepseek: { apiKeyEnv: 'AI_PADDED_KEY', baseURL: server.url } },
    })

    await chunksOf(ctx, call)
    expect(server.headers[0]?.authorization).toBe('Bearer padded-key')
  })

  it('reads the trusted environment when no seam is mounted', async () => {
    const dir = await freshHome()
    const server = await mockServer([{ kind: 'sse', events: textEvents }])
    const ctx = await boot(dir, {
      providers: { deepseek: { apiKeyEnv: 'AI_AMBIENT_KEY', baseURL: server.url } },
    }, 'absent')
    vi.stubEnv('AI_AMBIENT_KEY', 'ambient-key')

    await chunksOf(ctx, call)
    expect(server.headers[0]?.authorization).toBe('Bearer ambient-key')
  })

  it('resolves the credentials seam per call, so load order does not freeze availability', async () => {
    const dir = await freshHome()
    await writeFile(join(dir, '.credentials.yaml'), 'AI_LATE_KEY: late-key\n', { mode: 0o600 })
    const server = await mockServer([{ kind: 'sse', events: textEvents }])
    const ctx = await boot(dir, {
      providers: { deepseek: { apiKeyEnv: 'AI_LATE_KEY', baseURL: server.url } },
    }, 'absent')
    // The seam mounts after the adapter; the next request still finds it.
    await ctx.plugin(LocalCredentialProvider, { path: join(dir, '.credentials.yaml'), watch: false })

    await chunksOf(ctx, call)
    expect(server.headers[0]?.authorization).toBe('Bearer late-key')
  })

  it('fails MISSING_CREDENTIAL through the seam when the reference resolves to nothing', async () => {
    const dir = await freshHome()
    const server = await mockServer([])
    const ctx = await boot(dir, {
      providers: { deepseek: { apiKeyEnv: 'AI_MISSING_KEY', baseURL: server.url } },
    })
    vi.stubEnv('AI_MISSING_KEY', '')

    const failure = await failureOf(ctx, call)
    expect(failure.code).toBe('MISSING_CREDENTIAL')
    // The guidance names the route, the reference, both credential stores, and
    // both configuration entry points — everything a fix could touch.
    expect(failure.message).toMatch(/provider route "deepseek"/)
    expect(failure.message).toMatch(/AI_MISSING_KEY/)
    expect(failure.message).toMatch(/credentials service/)
    expect(failure.message).toMatch(/launching environment/)
    expect(failure.message).toMatch(/cordis\.yml/)
    expect(failure.message).toMatch(/llm-ai: settings section/)
    // A reference that misses never authenticates the request at all.
    expect(server.requests).toHaveLength(0)

    // Absent, not just empty: the same refusal.
    const absent = await boot(await freshHome(), {
      providers: { deepseek: { apiKeyEnv: 'AI_NEVER_SET_KEY', baseURL: server.url } },
    })
    expect((await failureOf(absent, call)).code).toBe('MISSING_CREDENTIAL')
    expect(server.requests).toHaveLength(0)
  })

  it('fails MISSING_CREDENTIAL from the environment when no seam is mounted and the variable is empty', async () => {
    const dir = await freshHome()
    const server = await mockServer([])
    const ctx = await boot(dir, {
      providers: { deepseek: { apiKeyEnv: 'AI_ENV_EMPTY_KEY', baseURL: server.url } },
    }, 'absent')
    vi.stubEnv('AI_ENV_EMPTY_KEY', '')

    const failure = await failureOf(ctx, call)
    expect(failure.code).toBe('MISSING_CREDENTIAL')
    expect(failure.message).toMatch(/AI_ENV_EMPTY_KEY/)
    expect(server.requests).toHaveLength(0)
  })

  it('fails INVALID_CREDENTIAL naming route and reference, never any part of the key', async () => {
    const dir = await freshHome()
    const server = await mockServer([])
    const ctx = await boot(dir, {
      providers: { deepseek: { apiKeyEnv: 'AI_BAD_KEY', baseURL: server.url } },
    })
    await ctx.credentials.set(credentialRef('AI_BAD_KEY'), 'pl4nted-sécret key')

    const failure = await failureOf(ctx, call)
    expect(failure.code).toBe('INVALID_CREDENTIAL')
    expect(failure.message).toMatch(/provider route "deepseek"/)
    expect(failure.message).toMatch(/AI_BAD_KEY/)
    expect(failure.message).toMatch(/no HTTP header can carry/)
    // The refusal may name where to fix, never what was found there.
    expect(failure.message).not.toContain('pl4nted')
    expect(failure.message).not.toContain('sécret')
    expect(server.requests).toHaveLength(0)

    // Whitespace only is a supplied-but-blank key, not a missing one.
    await ctx.credentials.set(credentialRef('AI_BAD_KEY'), '   ')
    const blank = await failureOf(ctx, call)
    expect(blank.code).toBe('INVALID_CREDENTIAL')
    expect(blank.message).toMatch(/is blank/)
    expect(server.requests).toHaveLength(0)
  })

  it('sends no authorization header for a profile naming no reference, seam mounted or not', async () => {
    const dir = await freshHome()
    const server = await mockServer([{ kind: 'sse', events: textEvents }])
    const ctx = await boot(dir, { providers: { deepseek: { baseURL: server.url } } })

    await chunksOf(ctx, call)
    expect(server.headers[0]).not.toHaveProperty('authorization')
  })

  it('stores only the reference: no literal key in the settings document', async () => {
    const dir = await freshHome()
    const server = await mockServer([{ kind: 'sse', events: textEvents }])
    const ctx = await boot(dir, {})
    await ctx.settings.update(NS, {
      providers: { deepseek: { apiKeyEnv: 'AI_REDACT_KEY', baseURL: server.url } },
    })
    await ctx.credentials.set(credentialRef('AI_REDACT_KEY'), 'pl4nted-secret-value')
    await chunksOf(ctx, call)
    expect(server.headers[0]?.authorization).toBe('Bearer pl4nted-secret-value')

    // The user document carries the reference; only the managed credential
    // store ever holds the value.
    const settings = await readFile(join(dir, 'settings.yaml'), 'utf8')
    expect(settings).toContain('AI_REDACT_KEY')
    expect(settings).not.toContain('pl4nted-secret-value')
    const stored = await readFile(join(dir, '.credentials.yaml'), 'utf8')
    expect(stored).toContain('pl4nted-secret-value')
  })
})

describe('in-flight snapshot freeze', () => {
  it('keeps an in-flight stream on the facts it started with; the next request sees the change', async () => {
    const dir = await freshHome()
    await writeFile(join(dir, '.credentials.yaml'), 'AI_FREEZE_KEY: k-one\n', { mode: 0o600 })
    // Server A answers slowly, so the stream is still open when the settings
    // change lands; server B is where the change points the route.
    const serverA = await mockServer([{ kind: 'sse', events: textEvents, delayMs: 60 }])
    const serverB = await mockServer([{ kind: 'sse', events: textEvents }])
    const ctx = await boot(dir, {
      providers: { deepseek: { apiKeyEnv: 'AI_FREEZE_KEY', baseURL: serverA.url } },
    })

    const inFlight = chunksOf(ctx, call)
    await waitFor(() => serverA.requests.length === 1)
    // Repoint the endpoint and rotate the credential underneath the open stream.
    await ctx.settings.update(NS, {
      providers: { deepseek: { apiKeyEnv: 'AI_FREEZE_KEY', baseURL: serverB.url } },
    })
    await ctx.credentials.set(credentialRef('AI_FREEZE_KEY'), 'k-two')

    // The in-flight stream finishes on the endpoint and key it started with.
    const collected = await inFlight
    expect(collected[collected.length - 1]).toMatchObject({ type: 'finish', reason: { kind: 'stop' } })
    expect(serverA.requests).toHaveLength(1)
    expect(serverA.headers[0]?.authorization).toBe('Bearer k-one')
    expect(serverB.requests).toHaveLength(0)

    // The next request re-resolves both facts.
    await chunksOf(ctx, call)
    expect(serverB.requests).toHaveLength(1)
    expect(serverB.headers[0]?.authorization).toBe('Bearer k-two')
    expect(serverA.requests).toHaveLength(1)
  })
})

describe('per-provider settings merge over the composition base', () => {
  it('overrides one field, repoints a baseURL, and adds a route; reset restores the base', async () => {
    const dir = await freshHome()
    await writeFile(join(dir, '.credentials.yaml'), 'AI_BASE_KEY: base-key\nAI_ALT_KEY: alt-key\n', { mode: 0o600 })
    const serverA = await mockServer([{ kind: 'sse', events: textEvents }, { kind: 'sse', events: textEvents }])
    const serverB = await mockServer([{ kind: 'sse', events: textEvents }, { kind: 'sse', events: textEvents }])
    const ctx = await boot(dir, {
      providers: { deepseek: { apiKeyEnv: 'AI_BASE_KEY', baseURL: serverA.url } },
    })

    // One field overridden: the credential switches while the unstated
    // baseURL stays the composition's.
    await ctx.settings.update(NS, {
      providers: { deepseek: { apiKeyEnv: 'AI_ALT_KEY' } },
    })
    await chunksOf(ctx, call)
    expect(serverA.headers[0]?.authorization).toBe('Bearer alt-key')

    // A second write repoints the endpoint and adds a whole route.
    await ctx.settings.update(NS, {
      providers: {
        deepseek: { baseURL: serverB.url },
        visionai: { apiKeyEnv: 'AI_ALT_KEY', baseURL: serverB.url },
      },
    })
    await chunksOf(ctx, call)
    expect(serverB.requests).toHaveLength(1)
    expect(serverB.headers[0]?.authorization).toBe('Bearer alt-key')
    expect(ctx.llm.listProviders().map(provider => provider.id)).toEqual(['deepseek', 'visionai'])

    // Resetting the user layer re-inherits the composition base and drops the
    // settings-born route.
    await ctx.settings.replace(NS, {})
    expect(ctx.llm.listProviders()).toEqual([{ id: 'deepseek', name: 'deepseek' }])
    await chunksOf(ctx, call)
    expect(serverA.requests).toHaveLength(2)
    expect(serverA.headers[1]?.authorization).toBe('Bearer base-key')
  })

  it('re-registers in place when a captured retry policy changes', async () => {
    const dir = await freshHome()
    const server = await mockServer([{ kind: 'sse', events: textEvents }])
    const ctx = await boot(dir, {
      providers: { deepseek: { apiKeyEnv: 'AI_NEVER_SET_KEY', baseURL: server.url } },
    })

    await ctx.settings.update(NS, {
      providers: {
        deepseek: {
          retryPolicy: { mode: 'always', backoff: { initialDelayMs: 25, maxDelayMs: 100, jitterRatio: 0.2 } },
        },
      },
    })
    expect(ctx.llm.providerRetryPolicy('deepseek')).toEqual({
      mode: 'always',
      initialDelayMs: 25,
      maxDelayMs: 100,
      jitterRatio: 0.2,
    })
    expect(ctx.llm.listProviders()).toEqual([{ id: 'deepseek', name: 'deepseek' }])
  })

  it('refuses a settings write naming the dropped timeoutMs field', async () => {
    const dir = await freshHome()
    const server = await mockServer([{ kind: 'sse', events: textEvents }])
    const ctx = await boot(dir, {})

    await expect(ctx.settings.update(NS, {
      providers: { deepseek: { baseURL: server.url, timeoutMs: 5_000 } as never },
    })).rejects.toThrow(/timeoutMs, which named pi-ai runtime behavior/)
    // The refusal happened at the write: nothing was stored, nothing serves.
    expect(ctx.llm.listProviders()).toEqual([])
  })
})
