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
