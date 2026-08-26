/**
 * The wire-protocol table this adapter serves, plus the registry-side
 * servability classification that hangs off it.
 *
 * v1 speaks exactly one protocol: `openai-completions`, implemented by the
 * harness-owned runtime (the DeepSeek twin's dialect generalized). The
 * models.dev registry carries no wire-protocol field of its own — its provider
 * `api` value is a base URL — so every servable route defaults to that one
 * protocol and a route needing another fails resolution naming what it asked
 * for. `anthropic-messages` arrives in v2.
 *
 * @module dsh-llm-ai/provider
 */

/**
 * Every wire protocol a configured route may name. One entry in v1; the table
 * is the single authority both the config schema's `api` union and resolution
 * read, so widening it later widens every surface at once.
 */
const PROTOCOLS = ['openai-completions'] as const

/**
 * Every wire protocol a configured route may name, in table order.
 * @returns the supported protocol identifiers.
 */
export function supportedProtocols(): readonly string[] {
  return [...PROTOCOLS]
}

/**
 * The protocol a draft naming none is asked as: the table's first entry, which
 * in v1 is also its only one and the one every servable route defaults to.
 * Widening the table changes this default with it.
 */
export const DEFAULT_PROTOCOL: (typeof PROTOCOLS)[number] = PROTOCOLS[0]

/**
 * Registry provider routes whose wire protocol `openai-completions` cannot
 * serve, mapped to the protocol each one actually speaks. These are the
 * provider ids the spec withholds from route registration in v1: the native
 * Anthropic, Google, and Bedrock/Vertex families, and the OAuth-only OpenAI
 * Codex route (its protocol exists, but this configuration shape cannot
 * express its login flow). Their ids stay in the configurable-provider
 * directory so configuration surfaces can explain what is missing.
 */
const WITHHELD_PROTOCOLS: Readonly<Record<string, string>> = {
  anthropic: 'anthropic-messages',
  google: 'google-generative-ai',
  'google-vertex': 'vertex-ai',
  'google-vertex-anthropic': 'vertex-ai',
  'amazon-bedrock': 'bedrock-converse-stream',
  'openai-codex': 'openai-codex-responses',
}

/**
 * The wire protocol a registry provider speaks that this adapter cannot serve.
 * @param provider - models.dev provider id (also the route key).
 * @returns the withheld protocol, or `undefined` for a servable provider.
 */
export function withheldProtocol(provider: string): string | undefined {
  return WITHHELD_PROTOCOLS[provider]
}

/**
 * The reasoning efforts a registry model offers when its entry declares
 * `reasoning: true` and no declaration reshapes them: the
 * `openai-completions` protocol's own effort vocabulary, each level spelled
 * as itself on the wire, with `off` sending nothing (for most endpoints not
 * thinking is the parameter's absence). A `reasoningEfforts` declaration
 * replaces this set; `false` strips reasoning entirely.
 */
export const DEFAULT_REASONING_EFFORTS: Readonly<Record<'off' | 'low' | 'medium' | 'high', string | null>> = {
  off: null,
  low: 'low',
  medium: 'medium',
  high: 'high',
}
