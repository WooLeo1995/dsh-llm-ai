/**
 * The openai-completions wire runtime at the `ctx.llm` seam: a local mock SSE
 * server drives the adapter through real loopback HTTP, asserting exact chunk
 * sequences, request shapes, error-code classification, timeout-vs-abort,
 * reasoning dispatch, image input with oldest-first offload, and
 * single-request semantics. No test touches the network.
 */

import { join } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import LlmRuntime, {
  CallId,
  createAssistantMessage,
  createMessage,
  createToolResultMessage,
  createUserMessage,
  OFFLOADED_IMAGE_TEXT,
  ReasoningEffortId,
  userAgent,
} from '@deepseek-ai/dsh-llm'
import type { GenerateOptions, ImageBlock, LlmFailure, Message, StreamChunk } from '@deepseek-ai/dsh-llm'
import AttachmentStore, { AttachmentError, AttachmentId } from '@deepseek-ai/dsh-attachment'
import type {
  ImageAttachmentLimits,
  ImageAttachmentRef,
  SaveImageAttachment,
  StoredImageAttachment,
} from '@deepseek-ai/dsh-attachment'
import * as LlmAi from '../src/index.ts'
import { LlmAiAdapter } from '../src/index.ts'
import { resolveProfiles } from '../src/config.ts'
import {
  PROTOCOL_DEFAULT_DIALECT,
  serializeMessagesWithImages,
  serializeRequest,
  serializeRequestWithImages,
} from '../src/serialize.ts'
import type { ThinkingDispatch } from '../src/serialize.ts'
import { catalogFromSnapshot } from '../src/modelsdev.ts'
import { fixtureRegistry, home, registryServer } from './registry.ts'
import { mockServer, textEvents } from './mock-server.ts'
import type { Behavior, MockServer } from './mock-server.ts'

const cleanups: Array<() => Promise<void>> = []

afterEach(async () => {
  vi.unstubAllEnvs()
  vi.useRealTimers()
  while (cleanups.length > 0) await cleanups.pop()!()
})

/** A deepseek route served by one mock wire server URL. */
function route(
  url: string,
  overrides: Partial<Omit<LlmAi.LlmAiProviderProfile, 'baseURL' | 'apiKeyEnv'>> = {},
): LlmAi.LlmAiProviderProfile {
  return { baseURL: url, apiKeyEnv: 'WIRE_KEY', ...overrides }
}

/** Boot the runtime plus this plugin over the loopback registry snapshot. */
async function harness(providers: Record<string, LlmAi.LlmAiProviderProfile>): Promise<Context> {
  vi.stubEnv('WIRE_KEY', 'wire-secret')
  const registry = await registryServer(fixtureRegistry())
  cleanups.push(registry.close)
  const ctx = new Context()
  cleanups.push(async () => {
    await ctx.fiber.dispose()
  })
  await ctx.plugin(LlmRuntime)
  await ctx.plugin(LlmAi, {
    catalogUrl: registry.url,
    catalogCachePath: join(await home('dsh-ai-wire-'), 'cache.json'),
    providers,
  })
  return ctx
}

/** Boot the deepseek route against one mock wire server's script. */
async function wireHarness(
  script: Behavior[],
  profile: (url: string) => LlmAi.LlmAiProviderProfile = url => route(url),
): Promise<{ ctx: Context; server: MockServer }> {
  const server = await mockServer(script)
  const ctx = await harness({ deepseek: profile(server.url) })
  return { ctx, server }
}

const user = (text: string): Message => createUserMessage({
  content: [{ type: 'text', text }],
  source: { kind: 'user' },
})

const systemMessage = (text: string): Message => createMessage({
  role: 'system',
  content: [{ type: 'text', text }],
  source: { kind: 'plugin', plugin: 'wire-test' },
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
  if (last.reason.kind !== 'error' && last.reason.kind !== 'aborted') {
    throw new Error(`expected a failure finish, got ${last.reason.kind}`)
  }
  return last.reason.failure
}

const call: GenerateOptions = {
  provider: 'deepseek',
  model: 'deepseek-chat',
  messages: [user('hi')],
}

describe('text streaming', () => {
  it('translates a complete text generation into the exact chunk sequence, one request', async () => {
    const { ctx, server } = await wireHarness([{ kind: 'sse', events: textEvents }])

    expect(await chunksOf(ctx, {
      ...call,
      system: 'be brief',
      temperature: 0.5,
      maxTokens: 128,
      tools: [{ name: 'lookup', description: 'look it up', parameters: { type: 'object' } }],
    })).toEqual([
      { type: 'block-start', index: 0, blockType: 'text' },
      { type: 'text-delta', index: 0, text: 'hello' },
      { type: 'block-end', index: 0, block: { type: 'text', text: 'hello' } },
      { type: 'usage', usage: { inputTokens: 3, outputTokens: 1 } },
      { type: 'finish', reason: { kind: 'stop' } },
    ])

    expect(server.requests).toHaveLength(1)
    expect(server.requests[0]).toEqual({
      model: 'deepseek-chat',
      messages: [
        { role: 'system', content: 'be brief' },
        { role: 'user', content: 'hi' },
      ],
      stream: true,
      stream_options: { include_usage: true },
      tools: [{
        type: 'function',
        function: { name: 'lookup', description: 'look it up', parameters: { type: 'object' } },
      }],
      temperature: 0.5,
      max_completion_tokens: 128,
    })
    expect(server.headers[0]?.authorization).toBe('Bearer wire-secret')
    expect(server.headers[0]?.['user-agent']).toBe(userAgent())
    expect(server.headers[0]?.['content-type']).toBe('application/json')
    expect(server.headers[0]?.accept).toBe('text/event-stream')
  })

  it('round-trips assistant history, reasoning passback, and tool results', async () => {
    const { ctx, server } = await wireHarness([{ kind: 'sse', events: textEvents }])

    const messages: Message[] = [
      systemMessage('workspace rules'),
      createAssistantMessage({
        content: [
          { type: 'reasoning', text: 'pondering' },
          { type: 'text', text: 'checking' },
          { type: 'tool-call', id: CallId('call-1'), name: 'get_weather', arguments: '{"city":"SF"}' },
        ],
        source: { provider: 'deepseek', model: 'deepseek-chat' },
      }),
      createToolResultMessage({ callId: CallId('call-1'), content: [], isError: false }),
      user('and tomorrow?'),
    ]
    await chunksOf(ctx, { ...call, messages })

    expect(server.requests[0]).toEqual({
      model: 'deepseek-chat',
      messages: [
        { role: 'system', content: 'workspace rules' },
        {
          role: 'assistant',
          content: 'checking',
          reasoning_content: 'pondering',
          tool_calls: [{
            id: 'call-1',
            type: 'function',
            function: { name: 'get_weather', arguments: '{"city":"SF"}' },
          }],
        },
        { role: 'tool', tool_call_id: 'call-1', content: '(no output)' },
        { role: 'user', content: 'and tomorrow?' },
      ],
      stream: true,
      stream_options: { include_usage: true },
    })
  })

  it('omits absent optional request fields and sends a bare assistant turn', async () => {
    const { ctx, server } = await wireHarness([{ kind: 'sse', events: textEvents }])

    const messages: Message[] = [
      createAssistantMessage({ content: [{ type: 'text', text: '' }], source: { provider: 'deepseek', model: 'deepseek-chat' } }),
      user(''),
    ]
    await chunksOf(ctx, { ...call, messages })

    expect(server.requests[0]).toEqual({
      model: 'deepseek-chat',
      messages: [
        { role: 'assistant', content: '' },
        { role: 'user', content: '' },
      ],
      stream: true,
      stream_options: { include_usage: true },
    })
  })

  it('carries tool-result text beside its tool message', async () => {
    const { ctx, server } = await wireHarness([{ kind: 'sse', events: textEvents }])

    const messages: Message[] = [
      createToolResultMessage({
        callId: CallId('call-2'),
        content: [{ type: 'text', text: '22C' }],
        isError: false,
      }),
    ]
    await chunksOf(ctx, { ...call, messages })

    expect(server.requests[0]).toMatchObject({
      messages: [{ role: 'tool', tool_call_id: 'call-2', content: '22C' }],
    })
  })

  it('makes exactly one provider request per stream call, including failure paths', async () => {
    const { ctx, server } = await wireHarness([
      { kind: 'http-error', status: 500, body: '{}' },
      { kind: 'sse', events: textEvents },
      { kind: 'sse', events: ['{"choices":[{"delta":{},"finish_reason":"stop"}]}', '[DONE]'] },
    ])

    await failureOf(ctx, call)
    expect(server.requests).toHaveLength(1)

    await chunksOf(ctx, call)
    expect(server.requests).toHaveLength(2)

    // A degenerate empty completion is retryable classification, not the
    // adapter's own retry: still one request for this call.
    const collected = await chunksOf(ctx, call)
    expect(collected[collected.length - 1]).toMatchObject({
      type: 'finish',
      reason: { kind: 'error', failure: { code: 'EMPTY_RESPONSE' } },
    })
    expect(server.requests).toHaveLength(3)
  })
})

/** One durable image reference with a distinct id, media type, and encoded size. */
function imageRef(seed: string, mediaType: ImageAttachmentRef['mediaType'], bytes: number): ImageAttachmentRef {
  return {
    attachmentId: AttachmentId(`sha256:${seed.repeat(64)}`),
    mediaType,
    bytes,
    width: 1,
    height: 1,
  }
}

const IMAGE_PNG_A = imageRef('a', 'image/png', 3)
const IMAGE_JPEG_B = imageRef('b', 'image/jpeg', 6)
const IMAGE_PNG_C = imageRef('c', 'image/png', 3)

/** An image block carrying one durable reference. */
const imageBlock = (ref: ImageAttachmentRef): ImageBlock => ({ type: 'image', attachment: ref })

/**
 * In-memory attachment store standing in for the durable seam: records every
 * read so tests can prove omitted attachments are never read, and serves the
 * bytes the test staged for each id.
 */
class RecordingAttachmentStore extends AttachmentStore {
  /** Attachment ids read, in request order. */
  readonly reads: string[] = []
  /** Staged bytes by attachment id; unstaged ids read as a fixed sentinel. */
  readonly images = new Map<string, Uint8Array>()
  /** When set, readImage throws it instead of serving bytes. */
  readFailure: unknown

  override readonly imageLimits: ImageAttachmentLimits = {
    maxImageBytes: 3_500_000,
    maxImagesPerMessage: 20,
    maxMessageImageBytes: 14_000_000,
    maxImagePixels: 4_000_000,
    maxImageDimension: 2_048,
    mediaTypes: ['image/png', 'image/jpeg', 'image/webp', 'image/gif'],
  }

  override validateImage(_input: SaveImageAttachment): Promise<void> {
    return Promise.resolve()
  }

  override saveImage(_input: SaveImageAttachment): Promise<ImageAttachmentRef> {
    return Promise.reject(new Error('the wire suite never saves images'))
  }

  override readImage(ref: ImageAttachmentRef, _signal?: AbortSignal): Promise<StoredImageAttachment> {
    this.reads.push(String(ref.attachmentId))
    if (this.readFailure !== undefined) throw this.readFailure
    return Promise.resolve({ ref, data: this.images.get(String(ref.attachmentId)) ?? Uint8Array.of(9) })
  }
}

/** Boot the runtime plus this plugin with the attachment service present. */
async function attachmentHarness(
  script: Behavior[],
  providers: (url: string) => Record<string, LlmAi.LlmAiProviderProfile>,
  order: 'store-first' | 'adapter-first' = 'store-first',
): Promise<{ ctx: Context; server: MockServer; store: RecordingAttachmentStore }> {
  const server = await mockServer(script)
  const registry = await registryServer(fixtureRegistry())
  cleanups.push(registry.close)
  vi.stubEnv('WIRE_KEY', 'wire-secret')
  const ctx = new Context()
  cleanups.push(async () => {
    await ctx.fiber.dispose()
  })
  await ctx.plugin(LlmRuntime)
  const mount = async (): Promise<void> => {
    await ctx.plugin(LlmAi, {
      catalogUrl: registry.url,
      catalogCachePath: join(await home('dsh-ai-image-'), 'cache.json'),
      providers: providers(server.url),
    })
  }
  // The adapter resolves the service per call, so either load order serves
  // image requests; `adapter-first` proves availability is not frozen at mount.
  if (order === 'store-first') {
    await ctx.plugin(RecordingAttachmentStore)
    await mount()
  } else {
    await mount()
    await ctx.plugin(RecordingAttachmentStore)
  }
  return { ctx, server, store: ctx.get('attachments') as RecordingAttachmentStore }
}

/** The visionai route with its registry catalog, against one mock server. */
function visionHarness(
  script: Behavior[],
  profile: (url: string) => LlmAi.LlmAiProviderProfile = url => route(url),
): Promise<{ ctx: Context; server: MockServer; store: RecordingAttachmentStore }> {
  return attachmentHarness(script, url => ({ visionai: profile(url) }))
}

/** The vision-capable fixture route's default call. */
const visionCall: GenerateOptions = { provider: 'visionai', model: 'vision-large', messages: [] }

const imageMessage = (content: Parameters<typeof createUserMessage>[0]['content']): Message => createUserMessage({
  content,
  source: { kind: 'user' },
})

describe('image input and offload', () => {
  it('sends durable images as ordered base64 data URLs preserving text/image order', async () => {
    const { ctx, server, store } = await visionHarness([{ kind: 'sse', events: textEvents }])
    store.images.set(String(IMAGE_PNG_A.attachmentId), Uint8Array.of(1, 2, 3))
    store.images.set(String(IMAGE_JPEG_B.attachmentId), Uint8Array.of(4, 5, 6))

    await chunksOf(ctx, {
      ...visionCall,
      messages: [imageMessage([
        imageBlock(IMAGE_PNG_A),
        { type: 'text', text: 'describe both' },
        imageBlock(IMAGE_JPEG_B),
        // Blocks outside the user-input wire vocabulary ride along in typed
        // content without reaching the wire.
        { type: 'tool-call', id: CallId('call-x'), name: 'noop', arguments: '{}' },
      ])],
    })

    expect(server.requests[0]).toMatchObject({
      model: 'vision-large',
      messages: [{
        role: 'user',
        content: [
          { type: 'image_url', image_url: { url: 'data:image/png;base64,AQID' } },
          { type: 'text', text: 'describe both' },
          { type: 'image_url', image_url: { url: 'data:image/jpeg;base64,BAUG' } },
        ],
      }],
    })
    expect(store.reads).toEqual([String(IMAGE_PNG_A.attachmentId), String(IMAGE_JPEG_B.attachmentId)])
  })

  it('follows tool-result images with their string-only tool messages in one user message', async () => {
    const { ctx, server, store } = await visionHarness([{ kind: 'sse', events: textEvents }])
    store.images.set(String(IMAGE_PNG_A.attachmentId), Uint8Array.of(1, 2, 3))
    store.images.set(String(IMAGE_JPEG_B.attachmentId), Uint8Array.of(4, 5, 6))

    const messages: Message[] = [
      systemMessage('workspace rules'),
      createAssistantMessage({
        content: [{ type: 'tool-call', id: CallId('call-1'), name: 'screenshot', arguments: '{}' }],
        source: { provider: 'visionai', model: 'vision-large' },
      }),
      createToolResultMessage({
        callId: CallId('call-1'),
        content: [{ type: 'text', text: '22C' }, imageBlock(IMAGE_PNG_A)],
        isError: false,
      }),
      createToolResultMessage({
        callId: CallId('call-2'),
        content: [imageBlock(IMAGE_JPEG_B)],
        isError: false,
      }),
      createToolResultMessage({ callId: CallId('call-3'), content: [], isError: false }),
      user('and tomorrow?'),
    ]
    await chunksOf(ctx, { ...visionCall, messages })

    expect(server.requests[0]).toMatchObject({
      messages: [
        { role: 'system', content: 'workspace rules' },
        {
          role: 'assistant',
          content: '',
          tool_calls: [{ id: 'call-1', type: 'function', function: { name: 'screenshot', arguments: '{}' } }],
        },
        { role: 'tool', tool_call_id: 'call-1', content: '22C' },
        { role: 'tool', tool_call_id: 'call-2', content: '(see attached image)' },
        { role: 'tool', tool_call_id: 'call-3', content: '(no output)' },
        {
          role: 'user',
          content: [
            { type: 'text', text: 'Attached image(s) from tool result:' },
            { type: 'image_url', image_url: { url: 'data:image/png;base64,AQID' } },
            { type: 'image_url', image_url: { url: 'data:image/jpeg;base64,BAUG' } },
          ],
        },
        { role: 'user', content: 'and tomorrow?' },
      ],
    })
  })

  it('offloads the oldest images to the fixed placeholder until the request fits, without reading them', async () => {
    // Base64 sizes: A=4, B=8, C=4; the bound of 12 fits only after A leaves.
    const { ctx, server, store } = await visionHarness(
      [{ kind: 'sse', events: textEvents }],
      url => route(url, { maxRequestImageBytes: 12 }),
    )
    store.images.set(String(IMAGE_JPEG_B.attachmentId), Uint8Array.of(4, 5, 6))
    store.images.set(String(IMAGE_PNG_C.attachmentId), Uint8Array.of(1, 2, 3))

    await chunksOf(ctx, {
      ...visionCall,
      messages: [
        imageMessage([imageBlock(IMAGE_PNG_A), imageBlock(IMAGE_JPEG_B)]),
        imageMessage([imageBlock(IMAGE_PNG_C)]),
        user(''),
      ],
    })

    expect(server.requests[0]).toMatchObject({
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: OFFLOADED_IMAGE_TEXT },
            { type: 'image_url', image_url: { url: 'data:image/jpeg;base64,BAUG' } },
          ],
        },
        { role: 'user', content: [{ type: 'image_url', image_url: { url: 'data:image/png;base64,AQID' } }] },
        { role: 'user', content: '' },
      ],
    })
    expect(store.reads).toEqual([String(IMAGE_JPEG_B.attachmentId), String(IMAGE_PNG_C.attachmentId)])
  })

  it('rejects image input when the attachment service is absent, while text-only calls never need it', async () => {
    const server = await mockServer([{ kind: 'sse', events: textEvents }])
    const ctx = await harness({ visionai: { baseURL: server.url } })

    const failure = await failureOf(ctx, {
      ...visionCall,
      messages: [imageMessage([imageBlock(IMAGE_PNG_A)])],
    })
    expect(failure.code).toBe('UNSUPPORTED_CONTENT')
    expect(failure.message).toMatch(/requires the durable attachment service/)
    expect(server.requests).toHaveLength(0)

    await chunksOf(ctx, { ...visionCall, messages: [user('plain text')] })
    expect(server.requests).toHaveLength(1)
  })

  it('surfaces attachment verification failures under the store code, and other read faults as TRANSPORT', async () => {
    const { ctx, server, store } = await visionHarness([{ kind: 'sse', events: textEvents }])

    store.readFailure = new AttachmentError('stored image no longer matches its reference', 'ATTACHMENT_CORRUPT')
    const corrupt = await failureOf(ctx, {
      ...visionCall,
      messages: [imageMessage([imageBlock(IMAGE_PNG_A)])],
    })
    expect(corrupt).toMatchObject({
      code: 'ATTACHMENT_CORRUPT',
      message: 'stored image no longer matches its reference',
    })

    store.readFailure = new Error('disk gone')
    const transport = await failureOf(ctx, {
      ...visionCall,
      messages: [imageMessage([imageBlock(IMAGE_PNG_A)])],
    })
    expect(transport.code).toBe('TRANSPORT')
    expect(server.requests).toHaveLength(0)
  })

  it('refuses image content in a message whose wire role cannot carry it', async () => {
    const { ctx, server } = await visionHarness([{ kind: 'sse', events: textEvents }])

    const failure = await failureOf(ctx, {
      ...visionCall,
      messages: [createMessage({
        role: 'system',
        content: [imageBlock(IMAGE_PNG_A)],
        source: { kind: 'plugin', plugin: 'wire-test' },
      })],
    })
    expect(failure.code).toBe('UNSUPPORTED_CONTENT')
    expect(failure.message).toMatch(/system message/)
    expect(server.requests).toHaveLength(0)
  })

  it('keeps a registry model image-capable when the entry names it without declaring input', async () => {
    const { ctx, server, store } = await visionHarness(
      [{ kind: 'sse', events: textEvents }],
      url => route(url, { models: [{ id: 'vision-large' }] }),
    )
    store.images.set(String(IMAGE_PNG_A.attachmentId), Uint8Array.of(1, 2, 3))

    await chunksOf(ctx, { ...visionCall, messages: [imageMessage([imageBlock(IMAGE_PNG_A)])] })

    const content = (server.requests[0] as { messages: Array<{ content: unknown }> }).messages[0]?.content
    expect(JSON.stringify(content)).toContain('data:image/png;base64,AQID')
  })

  it('admits image input on a hand-declared model through the route default', async () => {
    const { ctx, server, store } = await attachmentHarness(
      [{ kind: 'sse', events: textEvents }],
      url => ({
        newai: {
          baseURL: url,
          defaultInput: ['text', 'image'],
          models: [{ id: 'new-vision', contextWindow: 4_096 }],
        },
      }),
    )
    store.images.set(String(IMAGE_PNG_A.attachmentId), Uint8Array.of(1, 2, 3))

    await chunksOf(ctx, {
      provider: 'newai',
      model: 'new-vision',
      messages: [imageMessage([imageBlock(IMAGE_PNG_A)])],
    })

    const content = (server.requests[0] as { messages: Array<{ content: unknown }> }).messages[0]?.content
    expect(JSON.stringify(content)).toContain('data:image/png;base64,AQID')
  })

  it('resolves the attachment service per call, so load order does not freeze availability', async () => {
    const { ctx, server, store } = await attachmentHarness(
      [{ kind: 'sse', events: textEvents }],
      url => ({ visionai: { baseURL: url } }),
      'adapter-first',
    )
    store.images.set(String(IMAGE_PNG_A.attachmentId), Uint8Array.of(1, 2, 3))

    await chunksOf(ctx, { ...visionCall, messages: [imageMessage([imageBlock(IMAGE_PNG_A)])] })

    const content = (server.requests[0] as { messages: Array<{ content: unknown }> }).messages[0]?.content
    expect(JSON.stringify(content)).toContain('data:image/png;base64,AQID')
  })
})

describe('image serialization units', () => {
  /** Structural stand-in: the direct conversion tests read images and nothing else. */
  const attachmentStore = (): AttachmentStore => ({
    readImage: (ref: ImageAttachmentRef) => Promise.resolve({ ref, data: Uint8Array.of(1, 2, 3) }),
  } as unknown as AttachmentStore)

  const signal = (): AbortSignal => new AbortController().signal

  it('recursively converts nested tool-result content and keeps the empty fallback', async () => {
    const messages = [createUserMessage({
      content: [
        {
          type: 'tool-result',
          toolCallId: CallId('nested'),
          content: [{
            type: 'tool-result',
            toolCallId: CallId('inner'),
            content: [{ type: 'text', text: 'inside' }],
          }],
        },
        { type: 'tool-result', toolCallId: CallId('empty'), content: [] },
      ],
      source: { kind: 'user' },
    })]

    await expect(serializeMessagesWithImages(
      messages,
      attachmentStore(),
      signal(),
      PROTOCOL_DEFAULT_DIALECT,
    )).resolves.toEqual([
      { role: 'tool', tool_call_id: 'nested', content: 'inside' },
      { role: 'tool', tool_call_id: 'empty', content: '(no output)' },
    ])
  })

  it('carries the request system prompt on the image path', async () => {
    const profiles = resolveProfiles({
      visionai: { baseURL: 'https://nowhere.example' },
    }, catalogFromSnapshot(fixtureRegistry()))
    const profile = profiles.get('visionai')!
    const model = profile.models.find(entry => entry.id === 'vision-large')!

    const request = await serializeRequestWithImages(
      {
        provider: 'visionai',
        model: 'vision-large',
        system: 'be brief',
        messages: [imageMessage([imageBlock(IMAGE_PNG_A)])],
      },
      profile,
      model,
      { state: 'none' } satisfies ThinkingDispatch,
      { attachments: attachmentStore(), maxRequestImageBytes: 20 * 1024 * 1024, signal: signal() },
    )

    expect(request.messages).toEqual([
      { role: 'system', content: 'be brief' },
      {
        role: 'user',
        content: [{ type: 'image_url', image_url: { url: 'data:image/png;base64,AQID' } }],
      },
    ])
  })
})

describe('reasoning, tool-call, and usage translation', () => {
  it('yields the exact interleaved chunk sequence with deferred ends, usage, and finish', async () => {
    const events = [
      '{"choices":[{"delta":{"reasoning_content":"think"}}]}',
      '{"choices":[{"delta":{"content":"answer"}}]}',
      '{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-7","type":"function","function":{"name":"search","arguments":"{\\"q\\""}}]}}]}',
      '{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":":1}"}}]}}]}',
      '{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
      '{"usage":{"prompt_tokens":10,"completion_tokens":4,"prompt_tokens_details":{"cached_tokens":3},"completion_tokens_details":{"reasoning_tokens":2}}}',
      '[DONE]',
    ]
    const { ctx } = await wireHarness([{ kind: 'sse', events }])

    expect(await chunksOf(ctx, { ...call, model: 'deepseek-reasoner' })).toEqual([
      { type: 'block-start', index: 0, blockType: 'reasoning' },
      { type: 'reasoning-delta', index: 0, text: 'think' },
      { type: 'block-start', index: 1, blockType: 'text' },
      { type: 'text-delta', index: 1, text: 'answer' },
      { type: 'block-start', index: 2, blockType: 'tool-call' },
      { type: 'tool-call-delta', index: 2, id: CallId('call-7'), name: 'search', argumentsDelta: '{"q"' },
      { type: 'tool-call-delta', index: 2, id: CallId('call-7'), name: 'search', argumentsDelta: ':1}' },
      { type: 'block-end', index: 0, block: { type: 'reasoning', text: 'think' } },
      { type: 'block-end', index: 1, block: { type: 'text', text: 'answer' } },
      { type: 'block-end', index: 2, block: { type: 'tool-call', id: CallId('call-7'), name: 'search', arguments: '{"q":1}' } },
      { type: 'usage', usage: { inputTokens: 7, outputTokens: 4, cacheReadTokens: 3, reasoningTokens: 2 } },
      { type: 'finish', reason: { kind: 'tool-calls' } },
    ])
  })

  it('reads the reasoning delta from either wire field without duplicating', async () => {
    const events = [
      '{"choices":[{"delta":{"reasoning":"via-reason"}}]}',
      '{"choices":[{"delta":{"reasoning_content":"primary","reasoning":"duplicate"}}]}',
      '{"choices":[{"delta":{"content":"do"}}]}',
      '{"choices":[{"delta":{"content":"ne"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"prompt_cache_hit_tokens":1}}',
      '[DONE]',
    ]
    const { ctx } = await wireHarness([{ kind: 'sse', events }])

    expect(await chunksOf(ctx, { ...call, model: 'deepseek-reasoner' })).toEqual([
      { type: 'block-start', index: 0, blockType: 'reasoning' },
      { type: 'reasoning-delta', index: 0, text: 'via-reason' },
      { type: 'reasoning-delta', index: 0, text: 'primary' },
      { type: 'block-start', index: 1, blockType: 'text' },
      { type: 'text-delta', index: 1, text: 'do' },
      { type: 'text-delta', index: 1, text: 'ne' },
      { type: 'block-end', index: 0, block: { type: 'reasoning', text: 'via-reasonprimary' } },
      { type: 'block-end', index: 1, block: { type: 'text', text: 'done' } },
      { type: 'usage', usage: { inputTokens: 0, outputTokens: 1, cacheReadTokens: 1 } },
      { type: 'finish', reason: { kind: 'stop' } },
    ])
  })

  it('maps degenerate completions to a retryable EMPTY_RESPONSE error finish', async () => {
    const { ctx } = await wireHarness([
      { kind: 'sse', events: ['{"choices":[{"delta":{},"finish_reason":"stop"}]}', '[DONE]'] },
      { kind: 'sse', events: ['{"choices":[{"delta":{},"finish_reason":"length"}]}', '[DONE]'] },
      { kind: 'sse', events: ['{"choices":[{"delta":{},"finish_reason":"content_filter"}]}', '[DONE]'] },
      { kind: 'sse', events: ['{}', '[DONE]'] },
    ])

    // A stop (or absent) finish with no opened blocks is degenerate.
    expect((await chunksOf(ctx, call))[0]).toMatchObject({
      type: 'finish',
      reason: { kind: 'error', failure: { code: 'EMPTY_RESPONSE' } },
    })
    // A max-tokens-style empty completion is the same retryable class.
    expect((await chunksOf(ctx, call))[0]).toMatchObject({
      type: 'finish',
      reason: { kind: 'error', failure: { code: 'EMPTY_RESPONSE' } },
    })
    // A finish the provider itself flagged as an error keeps its own code.
    expect((await chunksOf(ctx, call))[0]).toMatchObject({
      type: 'finish',
      reason: { kind: 'error', failure: { code: 'CONTENT_FILTER' } },
    })
    // No choices and no finish at all: the stop default still completes empty.
    expect((await chunksOf(ctx, call))[0]).toMatchObject({
      type: 'finish',
      reason: { kind: 'error', failure: { code: 'EMPTY_RESPONSE' } },
    })
  })
})

describe('reasoning dispatch', () => {
  /** A route whose model declares custom spellings for the dispatch tests. */
  function declared(
    efforts: Record<string, string | null>,
    routeOverrides: Partial<Omit<LlmAi.LlmAiProviderProfile, 'baseURL' | 'apiKeyEnv'>> = {},
  ): (url: string) => LlmAi.LlmAiProviderProfile {
    return url => route(url, { models: [{ id: 'deepseek-chat', reasoningEfforts: efforts }], ...routeOverrides })
  }

  it('sends the declared spelling with the enabled thinking toggle', async () => {
    const { ctx, server } = await wireHarness(
      [{ kind: 'sse', events: textEvents }],
      declared({ off: null, high: 'ultra' }),
    )

    await chunksOf(ctx, { ...call, reasoningEffort: ReasoningEffortId('high') })
    expect(server.requests[0]).toMatchObject({
      thinking: { type: 'enabled' },
      reasoning_effort: 'ultra',
    })
  })

  it('sends the registry-default spelling for a declared level', async () => {
    const { ctx, server } = await wireHarness([{ kind: 'sse', events: textEvents }])

    await chunksOf(ctx, { ...call, model: 'deepseek-reasoner', reasoningEffort: ReasoningEffortId('low') })
    expect(server.requests[0]).toMatchObject({
      thinking: { type: 'enabled' },
      reasoning_effort: 'low',
    })
  })

  it('sends the disabled spelling for valueless off and omits the effort parameter', async () => {
    const { ctx, server } = await wireHarness(
      [{ kind: 'sse', events: textEvents }],
      declared({ off: null, high: 'high' }),
    )

    await chunksOf(ctx, { ...call, reasoningEffort: ReasoningEffortId('off') })
    expect(server.requests[0]).toMatchObject({ thinking: { type: 'disabled' } })
    expect(server.requests[0]).not.toHaveProperty('reasoning_effort')
  })

  it('sends the wire value of a valued off', async () => {
    const { ctx, server } = await wireHarness(
      [{ kind: 'sse', events: textEvents }],
      declared({ off: 'none', high: 'high' }),
    )

    await chunksOf(ctx, { ...call, reasoningEffort: ReasoningEffortId('off') })
    expect(server.requests[0]).toMatchObject({
      thinking: { type: 'enabled' },
      reasoning_effort: 'none',
    })
  })

  it('materializes the route default when the request names no effort', async () => {
    const { ctx, server } = await wireHarness(
      [{ kind: 'sse', events: textEvents }],
      declared({ off: null, high: 'ultra' }, { reasoning: 'high' }),
    )

    await chunksOf(ctx, call)
    expect(server.requests[0]).toMatchObject({
      thinking: { type: 'enabled' },
      reasoning_effort: 'ultra',
    })
  })

  it('refuses an unselectable per-request level before network I/O', async () => {
    const server = await mockServer([{ kind: 'sse', events: textEvents }])
    const ctx = await harness({ deepseek: route(server.url, {
      models: [{ id: 'deepseek-chat', reasoningEfforts: { off: null, high: 'high' } }],
    }) })

    const failure = await failureOf(ctx, { ...call, reasoningEffort: ReasoningEffortId('medium') })
    expect(failure).toMatchObject({ code: 'UNSUPPORTED_REASONING_EFFORT' })
    expect(failure.message).toMatch(/provider "deepseek" model "deepseek-chat"/)
    expect(server.requests).toHaveLength(0)
  })
})

describe('compat switches', () => {
  // models.dev records no wire-dialect facts of its own, so resolution is
  // model → route → protocol default; the registry-inference layer of the
  // pi-ai lineage has nothing to read in this catalog.

  it('applies route-level switches to every model on the route that reads them', async () => {
    const { ctx, server } = await wireHarness(
      [{ kind: 'sse', events: textEvents }, { kind: 'sse', events: textEvents }],
      url => route(url, { compat: { maxTokensField: 'max_tokens', supportsDeveloperRole: true } }),
    )

    await chunksOf(ctx, { ...call, system: 'be brief', maxTokens: 64 })
    expect(server.requests[0]).toMatchObject({
      messages: [{ role: 'developer', content: 'be brief' }, { role: 'user', content: 'hi' }],
      max_tokens: 64,
    })
    expect(server.requests[0]).not.toHaveProperty('max_completion_tokens')

    await chunksOf(ctx, { ...call, model: 'deepseek-reasoner', system: 'be brief', maxTokens: 32 })
    expect(server.requests[1]).toMatchObject({
      messages: [{ role: 'developer', content: 'be brief' }, { role: 'user', content: 'hi' }],
      max_tokens: 32,
    })
    expect(server.requests[1]).not.toHaveProperty('max_completion_tokens')
  })

  it('resolves each field independently: model compat over route compat over protocol default', async () => {
    const { ctx, server } = await wireHarness(
      [{ kind: 'sse', events: textEvents }, { kind: 'sse', events: textEvents }],
      url => route(url, {
        compat: { maxTokensField: 'max_tokens', supportsDeveloperRole: true, thinkingFormat: 'openai' },
        models: [
          // The model restates the protocol defaults over the route's
          // switches: a per-field winner, never all-or-nothing.
          { id: 'deepseek-chat', compat: { maxTokensField: 'max_completion_tokens', supportsDeveloperRole: false } },
          // No compat of its own: every route switch applies.
          { id: 'deepseek-reasoner' },
        ],
      }),
    )

    await chunksOf(ctx, { ...call, system: 'be brief', maxTokens: 64 })
    expect(server.requests[0]).toMatchObject({
      messages: [{ role: 'system', content: 'be brief' }, { role: 'user', content: 'hi' }],
      max_completion_tokens: 64,
    })
    expect(server.requests[0]).not.toHaveProperty('max_tokens')

    await chunksOf(ctx, {
      ...call,
      model: 'deepseek-reasoner',
      system: 'be brief',
      maxTokens: 64,
      reasoningEffort: ReasoningEffortId('low'),
    })
    expect(server.requests[1]).toMatchObject({
      messages: [{ role: 'developer', content: 'be brief' }, { role: 'user', content: 'hi' }],
      max_tokens: 64,
      // The route's thinkingFormat reached a model that set none of its own.
      reasoning_effort: 'low',
    })
    expect(server.requests[1]).not.toHaveProperty('thinking')
  })

  it('keeps the protocol defaults when no layer states a field', async () => {
    const { ctx, server } = await wireHarness([{ kind: 'sse', events: textEvents }])

    await chunksOf(ctx, {
      ...call,
      model: 'deepseek-reasoner',
      system: 'be brief',
      maxTokens: 64,
      reasoningEffort: ReasoningEffortId('low'),
    })
    expect(server.requests[0]).toMatchObject({
      messages: [{ role: 'system', content: 'be brief' }, { role: 'user', content: 'hi' }],
      max_completion_tokens: 64,
      thinking: { type: 'enabled' },
      reasoning_effort: 'low',
    })
  })

  it('serializes the openai thinking format as the bare effort, with valueless off sending nothing', async () => {
    const { ctx, server } = await wireHarness(
      [
        { kind: 'sse', events: textEvents },
        { kind: 'sse', events: textEvents },
        { kind: 'sse', events: textEvents },
      ],
      url => route(url, {
        models: [{
          id: 'deepseek-chat',
          reasoningEfforts: { off: null, high: 'ultra' },
          compat: { thinkingFormat: 'openai' },
        }],
      }),
    )

    await chunksOf(ctx, { ...call, reasoningEffort: ReasoningEffortId('high') })
    expect(server.requests[0]).toMatchObject({ reasoning_effort: 'ultra' })
    expect(server.requests[0]).not.toHaveProperty('thinking')
    expect(server.requests[0]).not.toHaveProperty('reasoning')

    await chunksOf(ctx, { ...call, reasoningEffort: ReasoningEffortId('off') })
    expect(server.requests[1]).not.toHaveProperty('thinking')
    expect(server.requests[1]).not.toHaveProperty('reasoning_effort')
    expect(server.requests[1]).not.toHaveProperty('reasoning')

    await chunksOf(ctx, call)
    expect(server.requests[2]).not.toHaveProperty('reasoning_effort')
  })

  it('serializes the openrouter thinking format as the nested reasoning object, omitting it for valueless off', async () => {
    const { ctx, server } = await wireHarness(
      [
        { kind: 'sse', events: textEvents },
        { kind: 'sse', events: textEvents },
        { kind: 'sse', events: textEvents },
        { kind: 'sse', events: textEvents },
      ],
      url => route(url, {
        compat: { thinkingFormat: 'openrouter' },
        models: [
          { id: 'deepseek-chat', reasoningEfforts: { off: null, high: 'ultra' } },
          { id: 'deepseek-reasoner', reasoningEfforts: { off: 'none', high: 'high' } },
        ],
      }),
    )

    await chunksOf(ctx, { ...call, reasoningEffort: ReasoningEffortId('high') })
    expect(server.requests[0]).toMatchObject({ reasoning: { effort: 'ultra' } })
    expect(server.requests[0]).not.toHaveProperty('thinking')
    expect(server.requests[0]).not.toHaveProperty('reasoning_effort')

    // A valueless off is the object's absence.
    await chunksOf(ctx, { ...call, reasoningEffort: ReasoningEffortId('off') })
    expect(server.requests[1]).not.toHaveProperty('reasoning')
    expect(server.requests[1]).not.toHaveProperty('thinking')

    // Nothing dispatched: the same wire shape as a valueless off.
    await chunksOf(ctx, call)
    expect(server.requests[2]).not.toHaveProperty('reasoning')

    // A valued off carries its spelling into the object.
    await chunksOf(ctx, { ...call, model: 'deepseek-reasoner', reasoningEffort: ReasoningEffortId('off') })
    expect(server.requests[3]).toMatchObject({ reasoning: { effort: 'none' } })
  })

  it('threads the resolved dialect through the image serialization path', async () => {
    const { ctx, server, store } = await visionHarness(
      [{ kind: 'sse', events: textEvents }],
      url => route(url, { compat: { supportsDeveloperRole: true, maxTokensField: 'max_tokens' } }),
    )
    store.images.set(String(IMAGE_PNG_A.attachmentId), Uint8Array.of(1, 2, 3))

    await chunksOf(ctx, {
      ...visionCall,
      system: 'be brief',
      maxTokens: 32,
      messages: [imageMessage([imageBlock(IMAGE_PNG_A)])],
    })
    expect(server.requests[0]).toMatchObject({
      messages: [
        { role: 'developer', content: 'be brief' },
        { role: 'user', content: [{ type: 'image_url', image_url: { url: 'data:image/png;base64,AQID' } }] },
      ],
      max_tokens: 32,
    })
    expect(server.requests[0]).not.toHaveProperty('max_completion_tokens')
  })
})

describe('error classification', () => {
  it('classifies authentication failures', async () => {
    const { ctx } = await wireHarness([
      { kind: 'http-error', status: 401, body: '{"error":{"message":"bad key"}}' },
      { kind: 'http-error', status: 403, body: '{}' },
    ])
    expect(await failureOf(ctx, call)).toMatchObject({ code: 'AUTH', status: 401, message: 'bad key' })
    expect(await failureOf(ctx, call)).toMatchObject({ code: 'AUTH', status: 403 })
  })

  it('classifies quota exhaustion ahead of the 429 rate limit', async () => {
    const { ctx } = await wireHarness([
      { kind: 'http-error', status: 429, body: '{"error":{"code":"insufficient_quota","message":"quota is exhausted"}}' },
      { kind: 'http-error', status: 429, body: '{"error":{"message":"slow down"}}' },
    ])
    expect(await failureOf(ctx, call)).toMatchObject({ code: 'QUOTA', status: 429 })
    expect(await failureOf(ctx, call)).toMatchObject({ code: 'RATE_LIMIT', status: 429 })
  })

  it('classifies context overflow from provider detail text', async () => {
    const { ctx } = await wireHarness([
      { kind: 'http-error', status: 400, body: '{"error":{"code":"context_length_exceeded","message":"This model support maximum context window of 65536 tokens"}}' },
      { kind: 'http-error', status: 400, body: '{"error":{"message":"unknown field"}}' },
      { kind: 'http-error', status: 413, body: 'too big', contentType: 'text/plain' },
    ])
    expect(await failureOf(ctx, call)).toMatchObject({ code: 'CONTEXT_WINDOW_EXCEEDED', status: 400 })
    expect(await failureOf(ctx, call)).toMatchObject({ code: 'INVALID_REQUEST', status: 400 })
    expect(await failureOf(ctx, call)).toMatchObject({ code: 'INVALID_REQUEST', status: 413 })
  })

  it('classifies server and unusual statuses', async () => {
    const { ctx } = await wireHarness([
      { kind: 'http-error', status: 500, body: '{}' },
      { kind: 'http-error', status: 418, body: '{}' },
    ])
    expect(await failureOf(ctx, call)).toMatchObject({ code: 'SERVER', status: 500 })
    expect(await failureOf(ctx, call)).toMatchObject({ code: 'HTTP_418', status: 418 })
  })

  it('keeps the status-line message for bodies without one, JSON or not', async () => {
    const { ctx } = await wireHarness([
      { kind: 'http-error', status: 400, body: '{"error":{"type":"invalid_request_error"}}' },
      { kind: 'http-error', status: 502, body: '<html>bad gateway</html>', contentType: 'text/html' },
    ])
    expect(await failureOf(ctx, call)).toMatchObject({
      code: 'INVALID_REQUEST',
      message: 'llm-ai: provider "deepseek" HTTP 400',
    })
    expect(await failureOf(ctx, call)).toMatchObject({ code: 'SERVER', message: 'llm-ai: provider "deepseek" HTTP 502' })
  })

  it('retains retry-after and provider request id as structured facts', async () => {
    const { ctx } = await wireHarness([
      { kind: 'http-error', status: 429, body: '{}', headers: { 'retry-after': '2', 'x-request-id': 'req-1' } },
      {
        kind: 'http-error',
        status: 429,
        body: '{}',
        headers: { 'retry-after': new Date(Date.now() + 5_000).toUTCString() },
      },
      { kind: 'http-error', status: 429, body: '{}', headers: { 'retry-after': '0' } },
      { kind: 'http-error', status: 429, body: '{}', headers: { 'retry-after': 'soon' } },
      { kind: 'http-error', status: 429, body: '{}', headers: { 'retry-after': new Date(Date.now() - 5_000).toUTCString() } },
      { kind: 'http-error', status: 429, body: '{}', headers: { 'x-request-id': '' } },
    ])

    expect(await failureOf(ctx, call)).toMatchObject({ providerRetryAfterMs: 2_000, requestId: 'req-1' })
    const dated = await failureOf(ctx, call)
    expect(dated.providerRetryAfterMs).toBeGreaterThan(0)
    expect(dated.requestId).toBeUndefined()
    // Zero, non-numeric, past-dated, and empty values all omit the fact.
    for (let index = 0; index < 4; index += 1) {
      const omitted = await failureOf(ctx, call)
      expect(omitted.providerRetryAfterMs).toBeUndefined()
      expect(omitted.requestId).toBeUndefined()
    }
  })

  it('classifies connection refusal as TRANSPORT naming the configured endpoint', async () => {
    const ctx = await harness({ deepseek: route('http://127.0.0.1:1') })

    const failure = await failureOf(ctx, call)
    expect(failure).toMatchObject({
      code: 'TRANSPORT',
      message: 'llm-ai: provider "deepseek" request to http://127.0.0.1:1/chat/completions failed',
    })
  })

  it('classifies an abrupt mid-stream close as TRANSPORT', async () => {
    const { ctx } = await wireHarness([{ kind: 'close-early', events: ['{"choices":[{"delta":{"content":"par"}}]}'] }])

    const failure = await failureOf(ctx, call)
    expect(failure.code).toBe('TRANSPORT')
    expect(failure.message).toMatch(/^llm-ai: provider "deepseek" stream from .* failed$/)
  })

  it('classifies a stream that ends without [DONE] as STREAM_CLOSED', async () => {
    const { ctx } = await wireHarness([{ kind: 'sse', events: ['{"choices":[{"delta":{"content":"cut"}}]}'] }])

    expect(await failureOf(ctx, call)).toMatchObject({ code: 'STREAM_CLOSED' })
  })

  it('classifies an unparsable payload as MALFORMED_RESPONSE', async () => {
    const { ctx } = await wireHarness([{ kind: 'sse', events: ['{"choices":[{"delta":{"content":"a"}}]}', 'not json'] }])

    const collected = await chunksOf(ctx, call)
    expect(collected).toHaveLength(3)
    expect(collected[2]).toMatchObject({
      type: 'finish',
      reason: { kind: 'error', failure: { code: 'MALFORMED_RESPONSE' } },
    })
  })
})

describe('abort and idle timeout', () => {
  it('classifies a pre-aborted signal as an aborted finish without a request', async () => {
    const server = await mockServer([{ kind: 'sse', events: textEvents }])
    const ctx = await harness({ deepseek: route(server.url) })
    const controller = new AbortController()
    controller.abort()

    const collected = await chunksOf(ctx, { ...call, signal: controller.signal })
    expect(collected).toEqual([{
      type: 'finish',
      reason: { kind: 'aborted', failure: expect.objectContaining({ code: 'ABORTED' }) },
    }])
    expect(server.requests).toHaveLength(0)
  })

  it('keeps an earlier caller abort ABORTED while the stream is mid-flight', async () => {
    const { ctx } = await wireHarness([{ kind: 'sse', events: textEvents, delayMs: 50 }])
    const controller = new AbortController()

    const pending = chunksOf(ctx, { ...call, signal: controller.signal })
    setTimeout(() => { controller.abort() }, 30)
    const collected = await pending
    expect(collected).toHaveLength(1)
    expect(collected[0]).toMatchObject({
      type: 'finish',
      reason: { kind: 'aborted', failure: expect.objectContaining({ code: 'ABORTED' }) },
    })
  })

  it('aborts the underlying body when the stream stays idle past its timeout', async () => {
    vi.useFakeTimers()
    let stopped = false
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation((_input, init) => {
      const signal = init?.signal
      const body = new ReadableStream<Uint8Array>({
        start(controller) {
          signal?.addEventListener('abort', () => {
            stopped = true
            controller.error(signal.reason)
          }, { once: true })
        },
      })
      return Promise.resolve(new Response(body, { status: 200 }))
    })
    const profiles = resolveProfiles({
      deepseek: {
        baseURL: 'https://idle.example',
        models: [{ id: 'deepseek-chat', contextWindow: 100 }],
        streamIdleTimeoutMs: 100,
      },
    }, catalogFromSnapshot(fixtureRegistry()))
    const adapter = new LlmAiAdapter({
      profiles: () => profiles,
      resolveApiKey: () => 'key-of-no-concern',
    })
    try {
      const drain = (async () => {
        for await (const _chunk of adapter.stream({ provider: 'deepseek', model: 'deepseek-chat', messages: [] })) { /* drain */ }
      })()
      const rejected = expect(drain).rejects.toMatchObject({ failure: { code: 'TIMEOUT' } })
      await vi.advanceTimersByTimeAsync(0)
      await vi.advanceTimersByTimeAsync(100)
      await rejected
      expect(stopped).toBe(true)
    } finally {
      fetchSpy.mockRestore()
    }
  })

  it('keeps an idle provider read alive through SSE comments', async () => {
    vi.useFakeTimers()
    const encoder = new TextEncoder()
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(() => {
      const body = new ReadableStream<Uint8Array>({
        start(controller) {
          setTimeout(() => { controller.enqueue(encoder.encode(': keep-alive\n\n')) }, 75)
          setTimeout(() => { controller.enqueue(encoder.encode(': keep-alive\n\n')) }, 150)
          setTimeout(() => {
            controller.enqueue(encoder.encode(textEvents.map(event => `data: ${event}\n\n`).join('')))
            controller.close()
          }, 225)
        },
      })
      return Promise.resolve(new Response(body, { status: 200 }))
    })
    const profiles = resolveProfiles({
      deepseek: {
        baseURL: 'https://idle.example',
        models: [{ id: 'deepseek-chat', contextWindow: 100 }],
        streamIdleTimeoutMs: 100,
      },
    }, catalogFromSnapshot(fixtureRegistry()))
    const adapter = new LlmAiAdapter({
      profiles: () => profiles,
      resolveApiKey: () => 'key-of-no-concern',
    })
    try {
      const types: string[] = []
      const drain = (async () => {
        for await (const chunk of adapter.stream({ provider: 'deepseek', model: 'deepseek-chat', messages: [] })) {
          types.push(chunk.type)
        }
      })()
      await vi.advanceTimersByTimeAsync(75)
      await vi.advanceTimersByTimeAsync(75)
      await vi.advanceTimersByTimeAsync(75)
      await expect(drain).resolves.toBeUndefined()
      expect(types).toEqual(['block-start', 'text-delta', 'block-end', 'usage', 'finish'])
    } finally {
      fetchSpy.mockRestore()
    }
  })
})

describe('credentials', () => {
  it('sends no auth header for a profile naming no apiKeyEnv', async () => {
    const server = await mockServer([{ kind: 'sse', events: textEvents }])
    const ctx = await harness({ deepseek: { baseURL: server.url } })

    await chunksOf(ctx, call)
    expect(server.headers[0]).not.toHaveProperty('authorization')
  })

  it('fails MISSING_CREDENTIAL when the referenced variable is unset or empty', async () => {
    const server = await mockServer([{ kind: 'sse', events: textEvents }, { kind: 'sse', events: textEvents }])
    const ctx = await harness({
      deepseek: { baseURL: server.url, apiKeyEnv: 'WIRE_UNSET_KEY' },
    })
    expect(await failureOf(ctx, call)).toMatchObject({ code: 'MISSING_CREDENTIAL' })
    expect(server.requests).toHaveLength(0)

    vi.stubEnv('WIRE_UNSET_KEY', '')
    expect(await failureOf(ctx, call)).toMatchObject({ code: 'MISSING_CREDENTIAL' })
    expect(server.requests).toHaveLength(0)
  })

  it('resolves the credential through the injectable hook', async () => {
    const server = await mockServer([{ kind: 'sse', events: textEvents }])
    const profiles = resolveProfiles({
      deepseek: { baseURL: server.url, apiKeyEnv: 'WIRE_KEY' },
    }, catalogFromSnapshot(fixtureRegistry()))
    const adapter = new LlmAiAdapter({
      profiles: () => profiles,
      resolveApiKey: profile => Promise.resolve(`${profile.provider}-key`),
    })
    for await (const _chunk of adapter.stream(call)) { /* drain */ }
    expect(server.headers[0]?.authorization).toBe('Bearer deepseek-key')
  })

  it('reads the referenced environment variable when no credentials seam is mounted', async () => {
    const server = await mockServer([{ kind: 'sse', events: textEvents }])
    const ctx = await harness({
      deepseek: { baseURL: server.url, apiKeyEnv: 'WIRE_AMBIENT_KEY' },
    })
    vi.stubEnv('WIRE_AMBIENT_KEY', 'ambient-key')

    await chunksOf(ctx, call)
    expect(server.headers[0]?.authorization).toBe('Bearer ambient-key')
  })
})

describe('provider headers and policy', () => {
  it('merges profile headers under harness attribution', async () => {
    const server = await mockServer([{ kind: 'sse', events: textEvents }])
    const ctx = await harness({
      deepseek: route(server.url, { headers: { 'x-deploy': 'a', 'user-agent': 'spoof' } }),
    })

    await chunksOf(ctx, call)
    expect(server.headers[0]?.['x-deploy']).toBe('a')
    expect(server.headers[0]?.['user-agent']).toBe(userAgent())
  })

  it('reports the resolved retry policy through the provider-policy seam', async () => {
    const server = await mockServer([{ kind: 'sse', events: textEvents }])
    const ctx = await harness({
      deepseek: route(server.url, { retryPolicy: { mode: 'normal', maxRetries: 2 } }),
    })

    expect(ctx.llm.providerRetryPolicy('deepseek')).toMatchObject({ mode: 'normal', maxRetries: 2 })
  })
})

describe('adapter boundary', () => {
  // The llm runtime validates the selected effort against reported metadata
  // before dispatch, so the adapter's own refusal of a non-canonical level is
  // only reachable from a direct registration.
  it('refuses a level outside the canonical vocabulary before network I/O', async () => {
    const profiles = resolveProfiles({
      deepseek: { baseURL: 'https://nowhere.example', models: [{ id: 'deepseek-chat', contextWindow: 100 }] },
    }, catalogFromSnapshot(fixtureRegistry()))
    const adapter = new LlmAiAdapter({ profiles: () => profiles, resolveApiKey: () => undefined })

    await expect(async () => {
      for await (const _chunk of adapter.stream({
        provider: 'deepseek',
        model: 'deepseek-chat',
        messages: [],
        reasoningEffort: ReasoningEffortId('gigantic'),
      })) { /* drain */ }
    }).rejects.toMatchObject({ failure: { code: 'UNSUPPORTED_REASONING_EFFORT' } })
  })

  // The llm runtime projects images to deterministic text for models whose
  // reported input modalities omit image before dispatch, so the adapter's
  // own modality refusal is only reachable from a direct registration — the
  // same boundary pattern as the reasoning-vocabulary refusal above.
  const textOnlyRoutes: Array<{
    case: string
    provider: string
    model: string
    profile: Partial<Omit<LlmAi.LlmAiProviderProfile, 'baseURL'>>
  }> = [
    {
      case: 'a registry text-only model',
      provider: 'deepseek',
      model: 'deepseek-chat',
      profile: {},
    },
    {
      case: 'a route-narrowed entry',
      provider: 'visionai',
      model: 'vision-large',
      profile: { models: [{ id: 'vision-large', input: ['text'] }] },
    },
    {
      case: 'an undeclared hand-declared model on the text route default',
      provider: 'privateai',
      model: 'priv-lm',
      profile: { models: [{ id: 'priv-lm', contextWindow: 1_000 }] },
    },
  ]
  for (const route of textOnlyRoutes) {
    it(`refuses image input for ${route.case} before credential, attachment, or network I/O`, async () => {
      const server = await mockServer([{ kind: 'sse', events: textEvents }])
      const resolveApiKey = vi.fn(() => 'never')
      const resolveAttachments = vi.fn(() => ({}) as AttachmentStore)
      const profiles = resolveProfiles({
        [route.provider]: { baseURL: server.url, ...route.profile },
      }, catalogFromSnapshot(fixtureRegistry()))
      const adapter = new LlmAiAdapter({ profiles: () => profiles, resolveApiKey, resolveAttachments })

      await expect(async () => {
        for await (const _chunk of adapter.stream({
          provider: route.provider,
          model: route.model,
          messages: [createUserMessage({ content: [imageBlock(IMAGE_PNG_A)], source: { kind: 'user' } })],
        })) { /* drain */ }
      }).rejects.toMatchObject({
        failure: {
          code: 'UNSUPPORTED_CONTENT',
          message: `llm-ai: provider "${route.provider}" model "${route.model}" does not accept image input`,
        },
      })
      expect(resolveApiKey).not.toHaveBeenCalled()
      expect(resolveAttachments).not.toHaveBeenCalled()
      expect(server.requests).toHaveLength(0)
    })
  }

  it('refuses image input before network I/O when no attachment resolver is configured', async () => {
    const server = await mockServer([{ kind: 'sse', events: textEvents }])
    const profiles = resolveProfiles({
      visionai: { baseURL: server.url },
    }, catalogFromSnapshot(fixtureRegistry()))
    const adapter = new LlmAiAdapter({ profiles: () => profiles, resolveApiKey: () => undefined })

    await expect(async () => {
      for await (const _chunk of adapter.stream({
        provider: 'visionai',
        model: 'vision-large',
        messages: [createUserMessage({ content: [imageBlock(IMAGE_PNG_A)], source: { kind: 'user' } })],
      })) { /* drain */ }
    }).rejects.toMatchObject({ failure: { code: 'UNSUPPORTED_CONTENT' } })
    expect(server.requests).toHaveLength(0)
  })

  it('refuses image content handed to the sync text serialization path', () => {
    const profiles = resolveProfiles({
      visionai: { baseURL: 'https://nowhere.example' },
    }, catalogFromSnapshot(fixtureRegistry()))
    const profile = profiles.get('visionai')!
    const model = profile.models.find(entry => entry.id === 'vision-large')!

    expect(() => serializeRequest(
      {
        provider: 'visionai',
        model: 'vision-large',
        messages: [createUserMessage({ content: [imageBlock(IMAGE_PNG_A)], source: { kind: 'user' } })],
      },
      profile,
      model,
      { state: 'none' } satisfies ThinkingDispatch,
    )).toThrow(/received image content outside the image serialization path/)
  })

  it('throws EMPTY_RESPONSE when a 200 carries no body', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(null, { status: 200 }))
    const profiles = resolveProfiles({
      deepseek: { baseURL: 'https://nowhere.example', models: [{ id: 'deepseek-chat', contextWindow: 100 }] },
    }, catalogFromSnapshot(fixtureRegistry()))
    const adapter = new LlmAiAdapter({ profiles: () => profiles, resolveApiKey: () => undefined })
    try {
      const drain = (async () => {
        for await (const _chunk of adapter.stream({ provider: 'deepseek', model: 'deepseek-chat', messages: [] })) { /* drain */ }
      })()
      await expect(drain).rejects.toMatchObject({ failure: { code: 'EMPTY_RESPONSE' } })
    } finally {
      fetchSpy.mockRestore()
    }
  })

  it('contains a failing transport teardown after an early consumer stop', async () => {
    const encoder = new TextEncoder()
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(() => {
      const body = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(encoder.encode('data: {"choices":[{"delta":{"content":"x"}}]}\n\n'))
        },
        cancel() {
          throw new Error('teardown blew up')
        },
      })
      return Promise.resolve(new Response(body, { status: 200 }))
    })
    const profiles = resolveProfiles({
      deepseek: { baseURL: 'https://nowhere.example', models: [{ id: 'deepseek-chat', contextWindow: 100 }] },
    }, catalogFromSnapshot(fixtureRegistry()))
    const adapter = new LlmAiAdapter({ profiles: () => profiles, resolveApiKey: () => undefined })
    try {
      const seen: string[] = []
      for await (const chunk of adapter.stream({ provider: 'deepseek', model: 'deepseek-chat', messages: [] })) {
        seen.push(chunk.type)
        break
      }
      expect(seen).toEqual(['block-start'])
    } finally {
      fetchSpy.mockRestore()
    }
  })

  it('maps transport rejections to TRANSPORT without losing the cause', async () => {
    const cause = new TypeError('connection refused')
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockRejectedValue(cause)
    const profiles = resolveProfiles({
      deepseek: { baseURL: 'https://unreachable.example', models: [{ id: 'deepseek-chat', contextWindow: 100 }] },
    }, catalogFromSnapshot(fixtureRegistry()))
    const adapter = new LlmAiAdapter({ profiles: () => profiles, resolveApiKey: () => undefined })
    try {
      const drain = (async () => {
        for await (const _chunk of adapter.stream({ provider: 'deepseek', model: 'deepseek-chat', messages: [] })) { /* drain */ }
      })()
      await expect(drain).rejects.toMatchObject({ code: 'TRANSPORT', cause })
    } finally {
      fetchSpy.mockRestore()
    }
  })

  it('renders a non-Error transport rejection without losing its cause', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(() => {
      const failed = Promise.withResolvers<Response>()
      failed.reject('offline')
      return failed.promise
    })
    const profiles = resolveProfiles({
      deepseek: { baseURL: 'https://unreachable.example', models: [{ id: 'deepseek-chat', contextWindow: 100 }] },
    }, catalogFromSnapshot(fixtureRegistry()))
    const adapter = new LlmAiAdapter({ profiles: () => profiles, resolveApiKey: () => undefined })
    try {
      const drain = (async () => {
        for await (const _chunk of adapter.stream({ provider: 'deepseek', model: 'deepseek-chat', messages: [] })) { /* drain */ }
      })()
      await expect(drain).rejects.toMatchObject({
        message: 'llm-ai: provider "deepseek" request to https://unreachable.example/chat/completions failed',
        code: 'TRANSPORT',
        cause: 'offline',
      })
    } finally {
      fetchSpy.mockRestore()
    }
  })
})
