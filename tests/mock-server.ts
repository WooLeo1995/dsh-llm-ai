/**
 * A local chat-completions stand-in for the wire runtime: loopback HTTP
 * replaying scripted SSE or error behaviors per request, recording every
 * request body and header bag. Tests never leave the machine.
 */

import { createServer } from 'node:http'
import type { IncomingMessage, Server, ServerResponse } from 'node:http'
import { afterEach } from 'vitest'

/** One scripted behavior for the next request the mock server receives. */
export type Behavior =
  | { kind: 'sse'; events: string[]; delayMs?: number }
  | { kind: 'http-error'; status: number; body: string; contentType?: string; headers?: Record<string, string> }
  | { kind: 'close-early'; events: string[] }

/** One running mock chat-completions endpoint. */
export interface MockServer {
  /** Base URL to configure as the route's `baseURL`. */
  url: string
  /** Bodies of received requests, in order. */
  requests: unknown[]
  /** Header bags of received requests, in order (parallel to `requests`). */
  headers: IncomingMessage['headers'][]
  /** Remaining scripted behaviors; the server 500s when the script runs dry. */
  script: Behavior[]
  /** Stop the server and wait for the socket to close. */
  close(): Promise<void>
}

const servers: Server[] = []

/** Close every server opened since the last call; run from each spec's afterEach. */
export async function closeMockServers(): Promise<void> {
  await Promise.all(servers.splice(0).map(server => new Promise<void>(resolve => {
    // Pooled keep-alive sockets would otherwise hold the listener open until
    // the client's own idle timeout; tests own both ends, so cut them now.
    server.closeAllConnections()
    server.close(() => { resolve() })
  })))
}

afterEach(async () => {
  await closeMockServers()
})

/** A minimal complete text generation, reused by request-shape assertions. */
export const textEvents = [
  '{"choices":[{"delta":{"role":"assistant","content":null,"reasoning_content":""}}]}',
  '{"choices":[{"delta":{"content":"hello"}}]}',
  '{"choices":[{"delta":{"content":""},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1}}',
  '[DONE]',
]

/** Serve scripted chat-completions behaviors on loopback. */
export async function mockServer(script: Behavior[]): Promise<MockServer> {
  const requests: unknown[] = []
  const headers: IncomingMessage['headers'][] = []
  const server = createServer((request: IncomingMessage, response: ServerResponse) => {
    let body = ''
    request.on('data', (chunk: Buffer) => { body += chunk.toString('utf8') })
    request.on('end', () => {
      requests.push(JSON.parse(body))
      headers.push(request.headers)
      const behavior = script.shift()
      if (!behavior) {
        response.writeHead(500).end('mock script exhausted')
        return
      }
      if (behavior.kind === 'http-error') {
        response.writeHead(behavior.status, {
          'content-type': behavior.contentType ?? 'application/json',
          ...behavior.headers,
        })
        response.end(behavior.body)
        return
      }
      response.writeHead(200, { 'content-type': 'text/event-stream' })
      const write = (index: number): void => {
        if (index >= behavior.events.length) {
          if (behavior.kind === 'sse') response.end()
          else response.destroy() // close-early: drop the socket mid-stream
          return
        }
        response.write(`data: ${behavior.events[index]}\n\n`)
        setTimeout(() => { write(index + 1) }, behavior.kind === 'sse' ? behavior.delayMs ?? 0 : 5)
      }
      write(0)
    })
  })
  servers.push(server)
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const address = server.address()
  if (address === null || typeof address === 'string') throw new Error('loopback server has no port')
  return {
    url: `http://127.0.0.1:${address.port}`,
    requests,
    headers,
    script,
    close: () => new Promise<void>(resolve => {
      server.closeAllConnections()
      server.close(() => { resolve() })
    }),
  }
}
