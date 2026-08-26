/**
 * Shared fixtures: a local models.dev-shaped registry server (loopback HTTP,
 * never the network), a fixture snapshot covering every loader branch, and
 * throwaway homes for cache files.
 */

import { createServer } from 'node:http'
import type { Server } from 'node:http'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach } from 'vitest'
import type { ModelsDevProvider } from '../src/modelsdev.ts'

/** A loopback registry server answering every path with the fixture body. */
export interface RegistryServer {
  /** Base URL to pass as `catalogUrl`. */
  url: string
  /** Stop the server and wait for the socket to close. */
  close(): Promise<void>
}

/** Serve one JSON body on loopback; tests never leave the machine. */
export async function registryServer(body: unknown): Promise<RegistryServer> {
  const server: Server = createServer((_request, response) => {
    response.writeHead(200, { 'content-type': 'application/json' })
    response.end(JSON.stringify(body))
  })
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const address = server.address()
  if (address === null || typeof address === 'string') throw new Error('loopback server has no port')
  return {
    url: `http://127.0.0.1:${address.port}/api.json`,
    close: () => new Promise<void>((resolve, reject) => server.close(error => error === undefined ? resolve() : reject(error))),
  }
}

/** A minimal models.dev-style provider entry. */
export function provider(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    id: 'acme',
    name: 'Acme',
    env: ['ACME_API_KEY'],
    api: 'https://api.acme.example/v1',
    models: {},
    ...overrides,
  }
}

/** A minimal models.dev-style model entry. */
export function model(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    name: 'Acme Large',
    limit: { context: 65_536, output: 4096 },
    ...overrides,
  }
}

/**
 * The fixture snapshot: `deepseek` resolves cleanly bare, `visionai` carries
 * image modality, `sketchy` holds unsized and garbage-field models (its bare
 * route is the capacity refusal), `endpointless` has models but no endpoint,
 * and the withheld families (`anthropic`, `google`) round it out.
 */
export function fixtureRegistry(): Record<string, ModelsDevProvider> {
  return {
    deepseek: provider({
      id: 'deepseek',
      name: 'DeepSeek',
      env: ['DEEPSEEK_API_KEY'],
      api: 'https://api.deepseek.example',
      models: {
        'deepseek-chat': model({ name: 'DeepSeek Chat' }),
        'deepseek-reasoner': model({ name: 'DeepSeek Reasoner', reasoning: true }),
        // An empty model id is skipped by the loader, never served.
        '': model({ name: 'Broken' }),
      },
    }),
    visionai: provider({
      id: 'visionai',
      name: 'Vision AI',
      env: ['VISIONAI_API_KEY'],
      api: 'https://api.visionai.example/v1',
      models: {
        'vision-large': model({
          name: 'Vision Large',
          modalities: { input: ['text', 'image'], output: ['text'] },
          limit: { context: 131_072, output: 16_384 },
        }),
      },
    }),
    sketchy: provider({
      id: 'sketchy',
      name: 'Sketchy',
      env: ['SKETCHY_API_KEY'],
      api: 'https://api.sketchy.example/v1',
      models: {
        // No context: the loud refusal names route and model.
        'sketchy-unsized': model({ name: 'Sketchy Unsized', limit: { output: 4096 } }),
        // An empty name and garbage fields floor to the id and honest values.
        'sketchy-odd': model({
          name: '',
          modalities: { input: 42 },
          limit: { context: 'big', output: -5 },
        }),
        'sketchy-fractional': model({ name: 'Sketchy Fractional', limit: { context: 1.5, output: 100 } }),
      },
    }),
    // Context but no output limit: the model serves with no output cap.
    capless: provider({
      id: 'capless',
      name: 'Capless',
      env: ['CAPLESS_API_KEY'],
      api: 'https://api.capless.example/v1',
      models: { 'capless-one': model({ name: 'Capless One', limit: { context: 8192 } }) },
    }),
    // No name and an empty-string api: the directory shows the id, and the
    // route needs a declared baseURL before any model resolves.
    endpointless: provider({
      id: 'endpointless',
      name: undefined,
      env: ['ENDPOINTLESS_API_KEY'],
      api: '',
      models: { 'endpointless-mini': model({ name: 'Mini' }) },
    }),
    // No models key at all: an empty catalog, still listed in the directory.
    bare: provider({
      id: 'bare',
      name: 'Bare',
      env: ['BARE_API_KEY'],
      api: 'https://api.bare.example/v1',
      models: undefined,
    }),
    anthropic: provider({
      id: 'anthropic',
      name: 'Anthropic',
      env: ['ANTHROPIC_API_KEY'],
      models: { 'claude-x': model({ name: 'Claude X' }) },
    }),
    google: provider({
      id: 'google',
      name: 'Google',
      env: ['GOOGLE_API_KEY'],
      models: {},
    }),
  }
}

const homes: string[] = []

afterEach(async () => {
  await Promise.all(homes.splice(0).map(dir => rm(dir, { recursive: true, force: true })))
})

/** A throwaway directory for cache and settings files. */
export async function home(prefix: string): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), prefix))
  homes.push(dir)
  return dir
}
