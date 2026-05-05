# 使用 Ollama 部署 Qwen3.6-35B-A3B（API 说明）

## 1. 模型名称与拉取

在 [Ollama 官方库](https://ollama.com/library/qwen3.6:35b-a3b) 中，模型名为 **`qwen3.6:35b-a3b`**（`模型名:tag` 格式；与 Hugging Face 上的连字符命名不完全相同）。

```bash
ollama pull qwen3.6:35b-a3b
ollama run qwen3.6:35b-a3b
```

查看本地已安装模型：

```bash
ollama list
```

---

## 2. HTTP API 基础地址

- **本机默认**：`http://localhost:11434`
- **原生 REST**：`http://localhost:11434/api/...`
- **官方文档**：[Ollama API Introduction](https://docs.ollama.com/api)

若使用 ollama.com 托管的云服务，文档中基础地址为 **`https://ollama.com/api`**，需按平台说明配置鉴权等。

---

## 3. 原生 API（常用）

### 文本补全：`POST /api/generate`

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "qwen3.6:35b-a3b",
  "prompt": "用一句话介绍 Rust。",
  "stream": false
}'
```

### 对话：`POST /api/chat`

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "qwen3.6:35b-a3b",
  "messages": [
    {"role": "user", "content": "你好，简要自我介绍一下。"}
  ],
  "stream": false
}'
```

常用参数还包括：`options`（如 `temperature`、`top_p`）、`format`（JSON 模式）、多模态时的 `images` 等，详见 [Generate](https://docs.ollama.com/api/generate) 与 [Chat](https://docs.ollama.com/api/chat)。

### 其它常见端点（前缀均为 `/api/`）

| 端点 | 说明 |
|------|------|
| `POST /api/embeddings` | 向量嵌入 |
| `GET /api/tags` | 已安装模型列表 |
| `POST /api/pull` | 拉取模型 |
| `DELETE /api/delete` | 删除模型 |
| `GET /api/version` | 服务版本 |

更多端点见 [API Reference](https://docs.ollama.com/api) 导航。

---

## 4. OpenAI 兼容层（可选）

同一 Ollama 进程可提供 **`/v1/...`**，例如对话：

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6:35b-a3b",
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

在 Python 中可将 OpenAI SDK 的 `base_url` 设为 `http://localhost:11434/v1`，`api_key` 可使用占位值（本地常见写法为 `ollama`）。说明见 [OpenAI compatibility](https://ollama.com/blog/openai-compatibility)。

**注意**：该兼容层在文档中标注为**实验性**，行为可能随版本变化；需要稳定、完整能力时优先使用原生 `/api/*` 或官方客户端。

---

## 5. 官方客户端库

- **Python**：<https://github.com/ollama/ollama-python>
- **JavaScript**：<https://github.com/ollama/ollama-js>

二者均封装上述 REST API。

---

## 小结

本地启动 Ollama 后，基础地址为 **`http://localhost:11434`**。多轮对话推荐 **`POST /api/chat`**；若需对接现有 OpenAI 风格客户端，使用 **`POST http://localhost:11434/v1/chat/completions`**，模型字段填 **`qwen3.6:35b-a3b`**。
