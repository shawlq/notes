# OpenClaw 模型配置说明

本文说明在安装 OpenClaw（AI 助手网关）后如何配置模型，适用于本机安装的 `openclaw` CLI / Gateway。若仅在 NVIDIA Build 等网页端使用，一般在平台界面选择模型即可。

## 1. 交互式向导（推荐入门）

```bash
openclaw onboard                     # 全流程引导（含模型等）
openclaw configure                   # 配置向导
openclaw configure --section model   # 仅「模型」一节
```

`configure` 向导中的 **Model** 会维护 `agents.defaults.models` 白名单，决定 `/model` 与模型选择器中可见的模型。向导通常会引导各厂商登录或填写凭证。

## 2. 网页 Control UI 配置

Gateway 运行后，在浏览器打开 Control UI，进入 **Config** 标签，可用表单或 Raw JSON 修改配置。

默认地址：`http://127.0.0.1:18789`

## 3. 直接编辑配置文件

- 默认路径：`~/.openclaw/openclaw.json`（JSON5 格式）
- 可通过环境变量 `OPENCLAW_CONFIG_PATH` 指向其他配置文件（需为真实普通文件，不建议用符号链接承接 OpenClaw 写入）

模型相关典型结构示例（主模型、备选、目录/别名）：

```json5
{
  agents: {
    defaults: {
      model: {
        primary: "anthropic/claude-sonnet-4-6",
        fallbacks: ["openai/gpt-5.4"],
      },
      models: {
        "anthropic/claude-sonnet-4-6": { alias: "Sonnet" },
        "openai/gpt-5.4": { alias: "GPT" },
      },
    },
  },
}
```

要点：

- 模型引用格式：`provider/model`（例如 `anthropic/claude-opus-4-6`）
- `agents.defaults.models` 既是模型目录，也作为可用模型的白名单

配置文件变更后，Gateway 一般会监听文件并热重载（详见官方文档「Configuration」）。

## 4. CLI 非交互修改

```bash
openclaw config get agents.defaults.model
# 使用 config set 等子命令按需合并；复杂 JSON 可使用 --strict-json --merge
```

查看/校验：

```bash
openclaw config validate
openclaw doctor
openclaw doctor --fix
```

配置不符合 schema 时 Gateway 可能无法启动，可用 `doctor` 排查与修复。

## 5. API Key 与环境变量

- 可从当前工作目录的 `.env` 或 `~/.openclaw/.env` 加载（不覆盖已存在的环境变量）
- 在 `openclaw.json` 中可使用 `env` 块，或在字符串里用 `${VAR_NAME}` 引用环境变量
- 敏感信息可使用 SecretRef 机制（见官方 Secrets 文档）

## 6. 与 NVIDIA Spark / Build 上 OpenClaw 的区别

本机安装的 OpenClaw 按上文通过 `configure`、`openclaw.json` 与 CLI 管理模型。若在 **NVIDIA Build（Spark）** 等网页环境中使用 OpenClaw，模型选择通常在网页界面完成，不一定需要编辑本地 `~/.openclaw`。

## 7. 实战经验备忘（替换提供商、白名单与 Agent 级 Failover）

以下整理自一次真实配置过程，便于日后按同样思路改 `~/.openclaw/openclaw.json`。

### 7.1 从某一提供商「下掉」模型并改用另一提供商

目标示例：聊天默认不再用 `ollama/...`，改为主用 `vllm/...`，并在选择器里可见。

需要同时顾及三层含义（容易只改一层导致 UI 仍显示旧模型或调用失败）：

1. **`agents.defaults.model.primary`**：默认会话的主模型 ID，格式 `provider/model`。
2. **`agents.defaults.models`**：模型白名单（`/model`、选择器）。新主用模型需在此有一键，否则可能无法选到。
3. **`models.providers.<provider>.models`**：各厂商的模型**目录**（元数据：context、cost 等）。
   - 要从 OpenClaw「模型列表」移除某一模型：删掉对应 `provider` 下 `models` 数组里的那条；若该提供商暂时没有任何聊天模型，可保留 `"models": []`（例如仍想保留 `tools.web.search` 等其它能力指向 Ollama 时）。
   - 新提供商：在 `models.providers` 里配置 `baseUrl`、`api`、以及含正确 `id` 的条目；插件 `plugins.entries` 中对应厂商需 `enabled: true`；`auth.profiles` 按需配置（如 `vllm:default`）。

改完后务必执行：

```bash
openclaw config validate
```

### 7.2 联网搜索与聊天模型分离

`tools.web.search.provider` 可以仍为 `ollama`，这与「对话主模型走 vLLM / NVIDIA」不矛盾：前者是搜索走哪个后端，后者是 Agent 推理用哪个模型。若希望完全停用 Ollama，需另行调整搜索 provider 或关闭搜索。

### 7.3 Per-agent 模型默认**不会**自动 Failover

官方行为见 [Model Failover](https://docs.openclaw.ai/concepts/model-failover)。

- **`agents.list[]` 里 `model` 写成字符串**（例如 `"model": "vllm/qwen36-35b-vllm"`）时，对该 Agent 视为 **strict**：后端连不上、或请求在产出回复前失败时，**一般直接报错**，**不会**自动改用 `agents.defaults.model.fallbacks` 或其它 Agent 的模型。
- **`agents.defaults.model.fallbacks`** 主要作用于「默认 primary、带 fallbacks 的 cron、以及自身显式配置了 fallbacks 的 agent」等场景，不能替代「未声明 fallbacks 的 per-agent 字符串 model」。

### 7.4 为单个 Agent 启用主模型 + 回退链

将 `model` 改为对象，包含 `primary` 与 `fallbacks` 数组（顺序即尝试顺序）。示例：

```json
{
  "id": "hm_atom",
  "model": {
    "primary": "vllm/qwen36-35b-vllm",
    "fallbacks": [
      "nvidia/nemotron-3-super-120b-a12b",
      "nvidia/minimaxai/minimax-m2.5",
      "nvidia/z-ai/glm5"
    ]
  }
}
```

说明：

- `fallbacks` 中的 ID 应与 `models.providers` 中已声明的模型一致；文档写明**可以不**依赖 `agents.defaults.models` 白名单也会按运维意图尝试，但保持与白名单一致更易审计。
- 若**禁止**该 Agent 做任何模型级回退，可使用 `"fallbacks": []`（显式 strict）。
- 是否进入下一条候选，取决于错误类型是否为官方定义的「可 failover」类（连接/超时/限流等）；部分错误（如上下文过长）会留在当前模型的重试/压缩逻辑，不按模型链切换。

配置变更后同样建议 `openclaw config validate`，必要时重启 Gateway。

---

## 参考链接

- [Configuration（总览与常见任务）](https://docs.openclaw.ai/configuration)
- [`openclaw configure` CLI 说明](https://docs.openclaw.ai/cli/configure)
- [Configuration Reference（完整字段）](https://docs.openclaw.ai/gateway/configuration-reference)
- [Configuration Examples](https://docs.openclaw.ai/gateway/configuration-examples)
- [Models 概念与 CLI](https://docs.openclaw.ai/concepts/models)
- [Model Failover](https://docs.openclaw.ai/concepts/model-failover)
- [Custom providers 与 base URL](https://docs.openclaw.ai/gateway/config-tools#custom-providers-and-base-urls)
- [Secrets Management](https://docs.openclaw.ai/gateway/secrets)
- [Environment 变量说明](https://docs.openclaw.ai/help/environment)
- [NVIDIA Build — OpenClaw（Spark）](https://build.nvidia.com/spark/openclaw)
