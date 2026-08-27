# dsh-llm-ai

English | [中文](README.zh.md)

A models.dev-cataloged multi-provider LLM adapter for the DeepSeek Harness: every provider and model fact comes from the community-maintained [models.dev](https://models.dev) `api.json` registry, and the request runtime is a harness-owned `openai-completions` streaming implementation (direct fetch + SSE) with no pi-ai dependency. It replaces `dsh-llm-pi-ai` on the same `ctx.llm` seam.

- **Protocol support**: v1 serves `openai-completions` only (the overwhelming majority of OpenAI-compatible endpoints); `anthropic-messages` and other protocols are v2 work.
- **Verification status**: 205+ unit tests at per-file 100% coverage; deployed as a complete replacement on DSH Desktop 2.0.3 (the dsh 0.1.1-rc.2 family) and in daily use.
- **Version baseline**: developed against `@deepseek-ai/*@next` (the 0.1.1-rc.2 line); peers are compatible with the same generation.

## Install (npm)

Published as the unscoped package `dsh-llm-ai` (the `@deepseek-ai/dsh-llm-ai` name is the harness monorepo's integrated twin). With the official CLI:

```sh
dsh plugin --profile <name> add dsh-llm-ai
```

One command installs and mounts: the CLI forwards to `pnpm add` in the profile directory, sees this package's `dsh.bundle.patch` declaration, appends it to the `dsh.profile.bundles` layer stack, and the profile boot merges the bundled patch — which disables the bundled `llm-pi-ai` mount and inserts `llm-ai` (the two adapters cannot coexist: the configurable-provider directory keys provider ids globally and both declaring the same catalog id fails `DUPLICATE_DIRECTORY` at load).

Notes:

- Migrating from a manual mount: remove the old `llm-ai` insert and `llm-pi-ai` disable lines from the profile's own `cordis.patch.yml` to avoid a duplicate entry id.
- Bundled-UI builds (DSH Desktop / web-app releases predating the llm-ai migration) hardcode the `llm-pi-ai` namespace in their Models page — apply the single-string alias documented under Deployment, step 4, to the installed copy at `<profile>/node_modules/dsh-llm-ai/lib/index.js`.
- Provider profiles live in the `llm-ai:` settings section (see the configuration reference); credential references need no migration.

## Features

- **models.dev catalog**: `api.json` is fetched once at plugin load and cached to disk under the DSH home (`storages/models-dev-cache.json`); offline boots serve the last good snapshot; a fetch failure is loud only when nothing is cached. `catalogUrl` / `catalogCachePath` override the endpoint and cache location.
- **openai-completions runtime**: streaming SSE (eventsource-parser), text, tool calls with raw-string arguments, tiered reasoning, image input with oldest-first offload under `maxRequestImageBytes`, usage and cache-hit accounting, an idle watchdog (`streamIdleTimeoutMs`), and exactly one provider request per `stream()` call.
- **Stable error codes**: `AUTH` / `QUOTA` / `RATE_LIMIT` / `CONTEXT_WINDOW_EXCEEDED` / `INVALID_REQUEST` / `SERVER` / `HTTP_<n>` / `TRANSPORT` / `TIMEOUT` / `ABORTED` / `STREAM_CLOSED` / `MALFORMED_RESPONSE` / `EMPTY_RESPONSE` (retryable classification).
- **Compat switches**: `maxTokensField` / `supportsDeveloperRole` / `thinkingFormat` (`openai` | `deepseek` | `openrouter`), resolved per field **model → route → protocol default**; unknown and valueless keys are refused listing the offered set — nothing is silently dropped.
- **Reasoning declarations**: `reasoningEfforts` maps each selectable level to its wire spelling; `off` is tri-state (absent = not offered; declared without a value = send the disabled spelling; declared with a value = send it); an undeclared level is refused before any network I/O.
- **Dynamic configuration**: the `providers` dict merges with the user settings section per provider, effective on the next request without a restart; dormant mounting (zero routes with no providers); atomic re-registration when the route set changes.
- **Credentials**: configuration stores `apiKeyEnv` references only; each request resolves them through the credentials seam, then the trusted environment; format checks (`INVALID_CREDENTIAL`) and empty references (`MISSING_CREDENTIAL`) name the route and every configuration entry point and never any part of the key.
- **Endpoint interrogation**: `GET /models` discovery for hand-declared gateways (a 4 MiB received-bytes ceiling, a typed draft key winning over stored references, and the `DISCOVERY_*` error family).
- **Configurable-provider directory**: declares all 203 models.dev providers to configuration surfaces (including families it cannot yet serve, with honest metadata).

## Configuration reference

Cordis composition entry:

```yaml
- id: llm-ai
  name: '@deepseek-ai/dsh-llm-ai'
  # Omitting config mounts dormant (zero routes); a settings section can
  # activate routes at any time.
  config:
    catalogUrl: https://models.dev/api.json      # optional: self-hosted mirror
    catalogCachePath: /path/to/cache.json        # optional: cache location
    providers:
      openai:                    # Catalog route: endpoint, protocol, and models
        apiKeyEnv: OPENAI_API_KEY    # all inherited from models.dev.
      zai-coding-cn:             # Hand-declared route: api + baseURL + a
        apiKeyEnv: ZAI_CODING_CN_API_KEY   # non-empty models list are required.
        api: openai-completions
        baseURL: https://open.bigmodel.cn/api/coding/paas/v4
        models:
          - { id: glm-5.3, contextWindow: 1000000, maxTokens: 131072 }
```

Provider profile fields: `apiKeyEnv` (credential reference), `displayName`, `api` (`openai-completions` only in v1), `baseURL`, `models` (**replaces** the route's catalog; unset fields default from the registry entry of the same id), `modelOverrides` (reshapes individual models while the rest of the catalog keeps serving), `compat` (the three switches), `reasoning` (the deployment default level), `retryPolicy` (omission = normal mode with five retries), `headers`, `defaultContextWindow` / `defaultMaxTokens` / `defaultInput` (fallbacks for configured entries that state no capacity), `streamIdleTimeoutMs` (five-minute default), `maxRequestImageBytes` (20 MiB default).

Catalog resolution notes: a models.dev model with no context window is refused rather than guessed; a `modelOverrides` key naming a model the catalog does not describe is refused; `timeoutMs` is gone (it named pi-ai runtime behavior) and configuring it fails with migration directions.

## DSH Desktop deployment (the complete procedure)

The following is the verified deployment path (executed on DSH Desktop 2.0.3 / dsh 0.1.1-rc.2). The desktop loads plugins through the pnpm mini-workspace at `~/.dsh/profiles/desktop/` — **the .app bundle itself is never modified**.

### 1. Build a self-sufficient install directory

```sh
mkdir -p ~/Downloads/project/github/dsh-llm-ai-app
# Take the build outputs from the harness repo (tsc lib/types + bundled runtime)
cp -R <harness>/packages/llm/llm-ai/lib ~/Downloads/project/github/dsh-llm-ai-app/
```

The install directory's `package.json` declares the plugin plus every peer as a **real** dependency from npm's `@next` dist-tag (self-sufficient, the vibe-island pattern):

```json
{
  "name": "@deepseek-ai/dsh-llm-ai",
  "version": "0.1.1-rc.2",
  "type": "module",
  "main": "lib/index.js",
  "exports": {
    ".": { "types": "./lib/types/index.d.ts", "default": "./lib/index.js" },
    "./invariant": { "types": "./lib/types/invariant.d.ts", "default": "./lib/invariant.js" }
  },
  "dependencies": {
    "@deepseek-ai/cordis": "next",
    "@deepseek-ai/dsh-attachment": "next",
    "@deepseek-ai/dsh-credentials": "next",
    "@deepseek-ai/dsh-home-paths": "next",
    "@deepseek-ai/dsh-launch-environment": "next",
    "@deepseek-ai/dsh-llm": "next",
    "@deepseek-ai/dsh-settings": "next",
    "@deepseek-ai/dsh-timeout": "next",
    "@deepseek-ai/schemastery": "next",
    "eventsource-parser": "^3.1.1"
  }
}
```

```sh
cd ~/Downloads/project/github/dsh-llm-ai-app && pnpm install
node --input-type=module -e "const m = await import('./lib/index.js'); console.log(typeof m.apply)"   # smoke test: function
```

### 2. Wire it into the profile workspace

Add to the dependencies of `~/.dsh/profiles/desktop/package.json`:

```json
"@deepseek-ai/dsh-llm-ai": "link:/Users/<you>/Downloads/project/github/dsh-llm-ai-app"
```

```sh
cd ~/.dsh/profiles/desktop && pnpm install
```

### 3. Composition patch

Append to `~/.dsh/profiles/desktop/cordis.patch.yml` (keep any managed blocks already there, such as vibe-island):

```yaml
- id: llm-pi-ai
  disabled: true
- insert:
    - id: llm-ai
      name: '@deepseek-ai/dsh-llm-ai'
```

### 4. Namespace compatibility (for the desktop's bundled UI)

**The key pitfall**: the Models page bundled with DSH Desktop (an upstream rc.2 build) hardcodes the `"llm-pi-ai"` namespace — the add-card's enable gate, protocol choices, form layout, and write target all recognize that name alone. Once llm-ai mounts, the page degrades to "other fields live in settings.yaml" hints.

The fix is a **single-string patch** on the plugin copy so it registers under `llm-pi-ai` (the settings section, directory entries, and discovery registration all flow from this one constant):

```js
// dsh-llm-ai-app/lib/index.js — the only occurrence in the file
- const NS = settingsNamespace("llm-ai");
+ const NS = settingsNamespace("llm-pi-ai");
```

Diagnostic message prefixes (`llm-ai: provider "..."`) need no change. When an upstream desktop release natively knows `llm-ai`, restore this string and rename the settings section to return to the canonical namespace.

### 5. settings.yaml migration

In `~/.dsh/settings.yaml`, migrate the previous `llm-pi-ai:` section to the curated routes (`anthropic-messages` routes must be removed — validation resolves the whole section, so one unserviceable profile refuses all of it). See the configuration reference above for a six-route example. Credential references (`apiKeyEnv` → environment variables / `~/.dsh/.credentials.yaml`) need no migration at all.

**Zero-downtime switch**: until the app restarts, the still-running pi-ai plugin keeps reading the old section — migrate with both sections present, restart, and delete the inert old section afterwards.

### 6. Pre-seed the models.dev cache (optional, recommended)

```sh
curl -s https://models.dev/api.json -o ~/.dsh/storages/models-dev-cache.json
```

This guarantees an offline-capable first boot; the plugin still tries a fresh fetch on every load and falls back to the cache on failure.

### 7. Restart and verify

Fully quit (⌘Q) and reopen DSH Desktop. Expected: the six routes are live, the model picker works, the Models page renders full editable cards (key / endpoint / protocol / model list), the add-provider card is usable, and the protocol dropdown offers `openai-completions` only.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| The Models page shows "Other fields live in settings.yaml; edit that section directly" | The bundled UI classifies `llm-ai` as an unknown layout — the step-4 namespace patch is not in effect; check that `settingsNamespace("llm-pi-ai")` occurs exactly once in the plugin copy |
| Writing files under `/Applications` fails with `EPERM` | macOS App Management (TCC) protects app bundles from every headless process (including node children of your own terminal) — which is why this procedure never touches the .app |
| The app fails to start after "disable the bundled UI entry + insert a replacement" | The desktop's composition loader rejects that substitution (verified to break boot); never replace the web-app's built-in client entries through a profile patch |
| A same-name link (shadowing a bundled package) does not take effect | Resolution precedence is not guaranteed to prefer the profile; when you need determinism, use a unique package name with an explicit entry, or an absolute path (the vibe-island precedent) |
| Every provider disappears at once | The settings section carries one unserviceable route (for example `anthropic-messages`) and was refused as a whole; fix or remove the route named in the error |
| First boot fails reporting a models.dev fetch failure | Nothing cached and the network is unreachable; run the step-6 seed |

**Full rollback**: delete the three patch entries from `cordis.patch.yml` → copy `settings.yaml.bak-llm-ai-swap` back over `settings.yaml` → restart. The install directory and profile link can stay (unreferenced means inert).

## Known limitations (v1)

- `openai-completions` only: the anthropic / google / bedrock / vertex / OAuth-only families stay visible in the directory but unserviceable, and naming them under `api` is refused; `anthropic-messages` is planned for v2.
- No replay envelope: cross-provider history converts provider-neutrally (no new session-log structure; logs recorded by pi-ai still load).
- The settings layer can add or override routes, never remove routes declared in the composition base (cordis.yml).
- One wire protocol per route: a mixed-protocol provider splits across two route keys.
- `tool_choice` and stop sequences are unsupported (an MVP cut shared with both predecessors).

## Development

```
src/
  index.ts       plugin apply: catalog load, dormant/atomic registration, the
                 settings section, directory and discovery registration
  adapter.ts     LlmAiAdapter: stream(), per-call snapshot freeze, timeout and
                 abort handling, error classification
  catalog.ts     profile → route/model resolution (models/modelOverrides/
                 compat/reasoning)
  config.ts      the schemastery Config schema and resolveProfiles
  modelsdev.ts   the api.json loader (fetch/cache/offline snapshot, injectable
                 fetchImpl)
  serialize.ts   request serialization, reasoning dispatch, image serialization
                 and offload
  sse.ts         eventsource-parser framing, the [DONE] sentinel, comment
                 watchdog pulses
  translate.ts   wire events → StreamChunk translation (usage precedes finish)
  discovery.ts   GET /models endpoint interrogation
  provider.ts    the protocol table (openai-completions only) and withheld
                 families
  types.ts       the wire vocabulary
```

```sh
pnpm install
npx tsc --noEmit        # typecheck
npx vitest run          # the whole suite (204+, no network)
npx vitest run --coverage  # per-file 100% coverage gate
pnpm run build          # tsdown: lib/ runtime bundle + declarations
```

The complete decision record lives in the deployment source repository under `.scratch/llm-ai/` (the spec plus twelve ticket resolutions); the harness monorepo's `packages/llm/llm-ai` is the integrated twin (with repo gates and doc regeneration). This directory is the origin for publication and standalone development.

## License

MIT (following the upstream DeepSeek Harness).
