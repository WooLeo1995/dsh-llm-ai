/**
 * Direct translation edge cases the seam suite cannot reach: a payload
 * source that violates parseSse's [DONE] guarantee, finishes that arrive
 * beside open blocks, and degenerate tool-call deltas that never carried an
 * id or name.
 */

import { describe, expect, it } from 'vitest'
import { CallId } from '@deepseek-ai/dsh-llm'
import type { StreamChunk } from '@deepseek-ai/dsh-llm'
import { translate } from '../src/translate.ts'

/** Collect every chunk of one direct translation. */
async function chunksOf(payloads: string[]): Promise<StreamChunk[]> {
  const collected: StreamChunk[] = []
  for await (const chunk of translate((async function* () {
    for (const payload of payloads) yield payload
  })())) {
    collected.push(chunk)
  }
  return collected
}

describe('translate', () => {
  it('rejects a payload source that ends without the [DONE] sentinel', async () => {
    await expect(chunksOf(['{"choices":[{"delta":{"content":"x"}}]}'])).rejects.toMatchObject({
      failure: { code: 'STREAM_CLOSED' },
    })
  })

  it('maps a length finish beside open blocks to max-tokens, usage first', async () => {
    expect(await chunksOf([
      '{"choices":[{"delta":{"content":"partial"}}]}',
      '{"choices":[{"delta":{},"finish_reason":"length"}],"usage":{"prompt_tokens":2,"completion_tokens":9}}',
      '[DONE]',
    ])).toEqual([
      { type: 'block-start', index: 0, blockType: 'text' },
      { type: 'text-delta', index: 0, text: 'partial' },
      { type: 'block-end', index: 0, block: { type: 'text', text: 'partial' } },
      { type: 'usage', usage: { inputTokens: 2, outputTokens: 9 } },
      { type: 'finish', reason: { kind: 'max-tokens' } },
    ])
  })

  it('keeps the latest usage when it arrives both attached and trailing', async () => {
    expect(await chunksOf([
      '{"choices":[{"delta":{"content":"x"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}',
      '{"usage":{"prompt_tokens":5,"completion_tokens":6}}',
      '[DONE]',
    ])).toEqual([
      { type: 'block-start', index: 0, blockType: 'text' },
      { type: 'text-delta', index: 0, text: 'x' },
      { type: 'block-end', index: 0, block: { type: 'text', text: 'x' } },
      { type: 'usage', usage: { inputTokens: 5, outputTokens: 6 } },
      { type: 'finish', reason: { kind: 'stop' } },
    ])
  })

  it('closes a tool-call block that never carried an id or name', async () => {
    expect(await chunksOf([
      '{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]}}]}',
      '{"choices":[{"delta":{"tool_calls":[{"index":0}]}}]}',
      '[DONE]',
    ])).toEqual([
      { type: 'block-start', index: 0, blockType: 'tool-call' },
      { type: 'tool-call-delta', index: 0, id: CallId(''), argumentsDelta: '{}' },
      { type: 'tool-call-delta', index: 0, id: CallId(''), argumentsDelta: '' },
      { type: 'block-end', index: 0, block: { type: 'tool-call', id: CallId(''), name: '', arguments: '{}' } },
      { type: 'finish', reason: { kind: 'stop' } },
    ])
  })

  it('reads a finish from a choice that carries no delta at all', async () => {
    // No blocks opened, so the completed finish maps to the retryable class.
    expect((await chunksOf([
      '{"choices":[{"finish_reason":"stop"}]}',
      '[DONE]',
    ]))[0]).toMatchObject({
      type: 'finish',
      reason: { kind: 'error', failure: { code: 'EMPTY_RESPONSE' } },
    })
  })
})
