/**
 * The invariant companion: a registration shell only. This package owns no
 * runtime event sequence or mutable-data relation beyond the contracts its
 * seam enforces, so installing it reserves the package name and contributes
 * no check; the disposer releases the reservation.
 */
import { Context } from '@deepseek-ai/cordis'
import { afterEach, describe, expect, it } from 'vitest'
import InvariantRegistry from '@deepseek-ai/dsh-invariants'
import * as LlmAiInvariant from '../src/invariant.ts'

const cleanups: Array<() => Promise<void>> = []

afterEach(async () => {
  while (cleanups.length > 0) await cleanups.pop()!()
})

describe('invariant companion', () => {
  it('registers this package with no runtime check and disposes cleanly', async () => {
    const ctx = new Context()
    cleanups.push(async () => {
      await ctx.fiber.dispose()
    })
    await ctx.plugin(InvariantRegistry, { enabled: true })

    expect(LlmAiInvariant.name).toBe('llm-ai-invariant')
    expect(LlmAiInvariant.inject).toEqual(['invariants'])
    const dispose = await LlmAiInvariant.apply(ctx)
    // The registry installs the no-op installer in a child fiber; let it start
    // before releasing the registration.
    await new Promise(resolve => setTimeout(resolve, 0))
    expect(typeof dispose).toBe('function')
    dispose()
  })
})
