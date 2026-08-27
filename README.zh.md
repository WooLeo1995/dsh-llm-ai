# dsh-llm-ai

中文 | [English](README.md)

基于 [models.dev](https://models.dev) 目录的 DeepSeek Harness 多供应商 LLM 适配器插件：供应商与模型元数据全部来自社区维护的 `api.json` 注册表，请求运行时为自研的 `openai-completions` 流式实现（direct fetch + SSE），不依赖 pi-ai。它是 `dsh-llm-pi-ai` 的替代品，挂载在同一 `ctx.llm` 接缝上。

- **协议支持**：v1 仅 `openai-completions`（覆盖绝大多数 OpenAI 兼容端点）；`anthropic-messages` 等留待 v2。
- **验证状态**：205+ 单元测试、逐文件 100% 覆盖；已在 DSH Desktop 2.0.3（dsh 0.1.1-rc.2 全家桶）上完成完整替换部署并日常使用。
- **版本基线**：面向 `@deepseek-ai/*@next`（0.1.1-rc.2 系列）开发，peer 兼容同代版本。

## npm 安装

以非域名包名 `dsh-llm-ai` 发布（`@deepseek-ai/dsh-llm-ai` 是 harness 主仓库的集成态孪生）。官方 CLI 一条命令：

```sh
dsh plugin --profile <name> add dsh-llm-ai
```

安装即挂载：CLI 在 profile 目录转发 `pnpm add`，识别包内 `dsh.bundle.patch` 声明后自动追加进 `dsh.profile.bundles` 层栈，profile 启动时合并随包补丁——禁用内置的 `llm-pi-ai` 挂载并插入 `llm-ai`（两个适配器不能并存：可配置目录按全局 id 键控，同目录 id 双双声明会在加载时 `DUPLICATE_DIRECTORY` 失败）。

注意：

- 从手工挂载迁移：删掉 profile 自己 `cordis.patch.yml` 里旧的 `llm-ai` 插入行和 `llm-pi-ai` 禁用行，避免条目 id 重复。
- 内置 UI 的构建（早于 llm-ai 迁移的 DSH Desktop / web-app 版本）在 Models 页硬编码 `llm-pi-ai` 命名空间——对安装副本 `<profile>/node_modules/dsh-llm-ai/lib/index.js` 打"部署第 4 步"的单字符串别名补丁。
- 供应商配置写入 `llm-ai:` settings 段（见配置参考）；凭证引用无需迁移。

## 功能特性

- **models.dev 目录**：插件加载时拉取 `api.json` 一次，磁盘缓存于 DSH home（`storages/models-dev-cache.json`）；离线时使用最后一次成功快照；仅当无缓存且拉取失败才响亮报错。`catalogUrl` / `catalogCachePath` 可覆盖。
- **openai-completions 运行时**：流式 SSE（eventsource-parser）、文本、工具调用（raw-string 参数）、reasoning 分级、图片输入 + `maxRequestImageBytes` 超限最旧先行卸载、usage / 缓存命中记账、空闲看门狗（`streamIdleTimeoutMs`）、单次 `stream()` 恰好一次请求。
- **稳定错误码**：`AUTH` / `QUOTA` / `RATE_LIMIT` / `CONTEXT_WINDOW_EXCEEDED` / `INVALID_REQUEST` / `SERVER` / `HTTP_<n>` / `TRANSPORT` / `TIMEOUT` / `ABORTED` / `STREAM_CLOSED` / `MALFORMED_RESPONSE` / `EMPTY_RESPONSE`（可重试分类）。
- **compat 开关**：`maxTokensField` / `supportsDeveloperRole` / `thinkingFormat`（`openai` | `deepseek` | `openrouter`），按 **模型 → 路由 → 协议默认** 逐字段解析；未知键、无值键一律拒绝并列出可选集，绝不静默丢弃。
- **reasoning 声明**：`reasoningEfforts` 映射"可选级别 → wire 拼写"；`off` 三态（不声明=不可选；声明无值=发 disabled 拼写；带值=按值发送）；未声明级别在 I/O 前拒绝。未声明时，注册表模型的可选级别取自 models.dev `reasoning_options` 的 effort 值（各级别按自身拼写上 wire；`none` → 无值 `off`）；`toggle`、空、缺失或全为非规范值的 options 保持协议默认集（`off`/`low`/`medium`/`high`）；profile 声明仍可整体重塑两者。
- **动态配置**：`providers` 字典 + settings 段热合并（下一个请求生效，无需重启）；休眠挂载（无 providers 时零路由）；路由集变更原子重注册。
- **凭证**：只存 `apiKeyEnv` 引用；逐请求经 credentials seam → 受信环境解析；格式校验（`INVALID_CREDENTIAL`）、引用落空（`MISSING_CREDENTIAL`）都点名路由与全部配置入口，永不泄露密钥内容。
- **端点探测**：手工声明网关的 `GET /models` 询问（4 MiB 实收字节上限、手输草稿密钥优先、`DISCOVERY_*` 错误码族）。
- **可配置目录**：向配置界面声明全部 203 家 models.dev 供应商（含暂不可服务的家族，附诚实元数据）。

## 配置参考

cordis 组合条目：

```yaml
- id: llm-ai
  name: '@deepseek-ai/dsh-llm-ai'
  # config 省略时休眠挂载（零路由），settings 段可随时激活路由
  config:
    catalogUrl: https://models.dev/api.json      # 可选：自建镜像
    catalogCachePath: /path/to/cache.json        # 可选：缓存位置
    providers:
      openai:                    # 目录路由：端点/协议/模型全部继承 models.dev
        apiKeyEnv: OPENAI_API_KEY
      zai-coding-cn:             # 手工路由：必须显式 api + baseURL + 非空 models
        apiKeyEnv: ZAI_CODING_CN_API_KEY
        api: openai-completions
        baseURL: https://open.bigmodel.cn/api/coding/paas/v4
        models:
          - { id: glm-5.3, contextWindow: 1000000, maxTokens: 131072 }
```

providers 路由字段：`apiKeyEnv`（凭证引用）、`displayName`、`api`（v1 仅 `openai-completions`）、`baseURL`、`models`（**整体替换**该路由目录，未写字段从注册表同名条目取默认）、`modelOverrides`（只改单个模型、其余目录照常服务）、`compat`（三开关）、`reasoning`（部署默认级别）、`retryPolicy`（省略=常规模式重试 5 次）、`headers`、`defaultContextWindow` / `defaultMaxTokens` / `defaultInput`（仅对未声明容量的配置条目兜底）、`streamIdleTimeoutMs`（默认 5 分钟）、`maxRequestImageBytes`（默认 20 MiB）。

目录解析要点：models.dev 模型**无上下文窗口则拒绝**（不猜测）；`modelOverrides` 指向目录不存在的模型会拒绝；`timeoutMs` 已删除（原属 pi-ai 运行时语义），配置了会得到带迁移指引的报错。

## DSH Desktop 部署（完整操作流程）

以下是经实际验证的完整部署路径（在 DSH Desktop 2.0.3 / dsh 0.1.1-rc.2 上执行）。桌面版通过 `~/.dsh/profiles/desktop/` 的 pnpm 迷你工作区加载插件，**无需改动 .app 本体**。

### 1. 构建自足安装目录

```sh
mkdir -p ~/Downloads/project/github/dsh-llm-ai-app
# 从 harness 仓库取构建产物（tsc lib/types + 打包 runtime）
cp -R <harness>/packages/llm/llm-ai/lib ~/Downloads/project/github/dsh-llm-ai-app/
```

安装目录的 `package.json`：插件本体 + 全部 peer 按 npm `@next` 作为**实依赖**安装（自足，vibe-island 模式）：

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
node --input-type=module -e "const m = await import('./lib/index.js'); console.log(typeof m.apply)"   # 烟测：function
```

### 2. 接入 profile 工作区

`~/.dsh/profiles/desktop/package.json` 的 dependencies 加：

```json
"@deepseek-ai/dsh-llm-ai": "link:/Users/<you>/Downloads/project/github/dsh-llm-ai-app"
```

```sh
cd ~/.dsh/profiles/desktop && pnpm install
```

### 3. 组合补丁

`~/.dsh/profiles/desktop/cordis.patch.yml` 追加（注意保留已有的托管块，如 vibe-island）：

```yaml
- id: llm-pi-ai
  disabled: true
- insert:
    - id: llm-ai
      name: '@deepseek-ai/dsh-llm-ai'
```

### 4. 命名空间兼容（桌面版内置 UI 专用）

**关键坑**：DSH Desktop 内置的模型配置页（上游 rc.2 构建）硬编码了 `"llm-pi-ai"` 命名空间——添加卡片的启用开关、协议选择、表单布局、写入目标全部只认它。llm-ai 挂载后配置页会退化为"请直接编辑 settings.yaml"提示。

解法：给插件拷贝打一个**单字符串补丁**，让它以 `llm-pi-ai` 命名空间注册（目录条目、settings 段、探测注册全部由这一个常量驱动）：

```js
// dsh-llm-ai-app/lib/index.js —— 全文仅此一处
- const NS = settingsNamespace("llm-ai");
+ const NS = settingsNamespace("llm-pi-ai");
```

诊断信息前缀（`llm-ai: provider "..."`）无需改动。上游桌面版未来若原生支持 `llm-ai`，改回这个字符串并把 settings 段改名即可回正。

### 5. settings.yaml 迁移

`~/.dsh/settings.yaml`：把原 `llm-pi-ai:` 段内容迁移为精选路由（`anthropic-messages` 路由必须移除——整段校验，一条不可服务会拒绝全段）。示例（六条路由）见上文配置参考。凭证引用（`apiKeyEnv` → 环境变量 / `~/.dsh/.credentials.yaml`）不需要任何迁移。

**零中断切换**：应用未重启期间，旧 pi-ai 插件仍在读旧段——迁移时可在新旧两段并存的状态下重启，重启后旧段变为惰性残留，事后删除。

### 6. models.dev 缓存预播种（可选但推荐）

```sh
curl -s https://models.dev/api.json -o ~/.dsh/storages/models-dev-cache.json
```

保证首启离线可用；此后插件每次加载仍会尝试刷新，失败时回落缓存。

### 7. 重启验证

完全退出（⌘Q）再打开 DSH Desktop。预期：六条路由在线、模型选择器可用、Models 页卡片为完整可编辑表单（密钥/端点/协议/模型列表）、添加供应商可用、协议下拉仅 `openai-completions`。

## 故障排查

| 症状 | 原因与处理 |
|---|---|
| 配置页显示"其余字段在 settings.yaml 中，请直接编辑对应段" | 内置 UI 把 `llm-ai` 判为 unknown 布局——第 4 步的命名空间补丁未生效，检查插件拷贝里 `settingsNamespace("llm-pi-ai")` 是否唯一存在 |
| 修改 `/Applications` 内文件报 `EPERM` | macOS App Management（TCC）保护应用本体，任何无头进程（含用户终端的 node 子进程）都写不进去——这正是本方案不碰 .app 的原因 |
| 尝试"禁用 bundle 内 UI 条目 + 插入替代品"后应用无法启动 | 桌面组合加载器不接受这种替换（实测会破坏启动）；**不要**通过 profile 补丁替换 web-app 的内置客户端条目 |
| 同名遮蔽（link 同名包）不生效 | 加载器解析优先级不保证 profile 优先；需要确定性指向时用「独有包名 + 显式条目」或绝对路径（vibe-island 先例） |
| 整段供应商全部消失 | settings 段里有一条不可服务路由（如 `anthropic-messages`）被整段拒绝；按报错里的路由/模型名修正或移除 |
| 首启失败提示 models.dev 拉取失败 | 无缓存且网络不可达；执行第 6 步播种 |

**完全回滚**：删掉 `cordis.patch.yml` 里的三条补丁 → 把 `settings.yaml.bak-llm-ai-swap` 拷回 `settings.yaml` → 重启。插件目录与 profile 链接可保留（无引用即惰性）。

## 已知限制（v1）

- 仅 `openai-completions` 协议：anthropic / google / bedrock / vertex / OAuth-only 家族在目录中可见但不可服务，`api` 声明它们会拒绝；`anthropic-messages` 为 v2 计划。
- 无 replay envelope：跨供应商历史走 provider-neutral 转换（不新增会话日志结构；旧 pi-ai 日志照常加载）。
- settings 层只能新增/覆盖路由，不能删除组合基线（cordis.yml）里声明的路由。
- 每路由单协议：混合协议供应商需拆成两个路由键。
- `tool_choice`、stop 序列不支持（与两个前身一致的 MVP 裁剪）。

## 开发

```
src/
  index.ts       插件 apply：目录加载、休眠/原子注册、settings 段、目录与探测注册
  adapter.ts     LlmAiAdapter：stream()、逐调用快照冻结、超时/中止、错误分类
  catalog.ts     profile → 路由/模型解析（models/modelOverrides/compat/reasoning）
  config.ts      schemastery Config 模式与 resolveProfiles
  modelsdev.ts   api.json 加载器（拉取/缓存/离线快照，fetchImpl 可注入）
  serialize.ts   请求序列化、reasoning 分发、图片序列化与卸载
  sse.ts         eventsource-parser 封装、[DONE] 哨兵、注释看门狗脉冲
  translate.ts   wire 事件 → StreamChunk 翻译（usage 先于 finish）
  discovery.ts   GET /models 端点探测
  provider.ts    协议表（仅 openai-completions）与 withheld 家族
  types.ts       wire 词汇表
```

```sh
pnpm install
npx tsc --noEmit        # 类型检查
npx vitest run          # 全部测试（204+，无网络依赖）
npx vitest run --coverage  # 逐文件 100% 覆盖门禁
pnpm run build          # tsdown：产出 lib/ 运行时 bundle 与类型声明
```

设计决策的完整记录见部署源仓库的 `.scratch/llm-ai/`（spec + 12 张票的 Resolution）；harness 主仓库的 `packages/llm/llm-ai` 是集成态孪生（含仓库门禁与文档再生成）。本目录是发布与独立开发的源头。

## 许可

MIT（随上游 DeepSeek Harness）。
