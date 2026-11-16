# Cursor ACP：飞书桥 + Open WebUI API 部署方案

> 参考实现机：`spark-c3b6`（用户 `atom`）  
> 代码目录：`~/workdir/qiu-acp-bridge`  
> 官方 ACP：<https://cursor.com/docs/cli/acp>  
> 目标：在**另一台 Linux 本机**复现同一套能力（可只部署其中一条链路）

---

## 1. 方案结论（先看这个）

本方案用 **Cursor CLI 的 `agent acp`** 当执行引擎，外挂两种入口：

| 入口 | 进程 | 端口 | cwd | 身份/人设 | Session |
|------|------|------|-----|-----------|---------|
| 飞书机器人（邱姐） | `qiu-acp-bridge` | 飞书长连接（无本地 HTTP） | `~/.cursor/agents/qiu` | 有（IDENTITY/SOUL/memory 等） | `state/sessions.json` |
| Open WebUI / 任意 OpenAI 客户端 | `qiu-acp-api` | `18791` | `~/.cursor/agents/acp-webui`（空目录即可） | **无**（纯 Cursor agent） | `state/webui-sessions.json` |

要点：

1. **两条链路各自一个 `agent acp` 子进程**，不共用 session / 对话上下文。  
2. **身份文件不是 ACP 协议必需**；飞书侧用人设是产品选择，WebUI 侧故意不做。  
3. WebUI 侧提供 **OpenAI 兼容** `GET /v1/models`、`POST /v1/chat/completions`，模型名默认 `atom-cursor-acp`，**不推 thinking**。  
4. API 默认绑 `0.0.0.0:18791` → 本机 / 局域网 `192.168.*` / Tailscale `100.*` 均可访问；**必须 Bearer token，勿对公网暴露**。  
5. 与 OpenClaw **解耦**；同一飞书 App 同时只能有一个长连接消费者。

```text
飞书用户 ──WS──► qiu-acp-bridge ──stdio──► agent acp ──► ~/.cursor/agents/qiu
                                                      （人设可选）

浏览器 ──HTTP──► Open WebUI :13000
                    │
                    │ OpenAI /v1（容器内 host.docker.internal:18791）
                    ▼
              qiu-acp-api :18791 ──stdio──► agent acp ──► ~/.cursor/agents/acp-webui
                                                      （空 cwd，无人设）
```

---

## 2. 前置条件

在目标机准备：

1. **Linux** + 公网出站（飞书长连接需要；仅 WebUI API 可无飞书）  
2. **Node.js ≥ 18**（推荐 20/22/24，nvm 即可）  
3. **Cursor CLI**（`agent` / `cursor-agent`）  
   ```bash
   which agent
   agent login          # 或 export CURSOR_API_KEY=...
   ```
4. （飞书）企业自建应用：机器人能力 + 长连接订阅 `im.message.receive_v1`  
5. （WebUI）Docker（本机参考用 Open WebUI 容器映射 `:13000`）

---

## 3. 目录布局

把 `USER` 换成目标机用户名。

```text
/home/USER/workdir/qiu-acp-bridge/     # 桥 + API 同一仓库
  ├── src/
  │   ├── index.ts          # 飞书入口
  │   ├── api-server.ts     # WebUI OpenAI API 入口
  │   ├── acp-client.ts     # 可配置的 ACP 客户端
  │   └── …
  ├── deploy/
  │   ├── qiu-acp-bridge.service
  │   └── qiu-acp-api.service
  ├── state/
  │   ├── sessions.json           # 飞书 session 映射
  │   └── webui-sessions.json     # WebUI session 映射
  ├── .env                    # 密钥，chmod 600，勿提交
  └── docs/

/home/USER/.cursor/agents/qiu/         # 飞书人设（可选但推荐）
  ├── IDENTITY.md / SOUL.md / USER.md / AGENTS.md
  ├── memory/  media/

/home/USER/.cursor/agents/acp-webui/   # WebUI 空工作区（无人设文件）

/home/USER/.config/systemd/user/
  ├── qiu-acp-bridge.service
  └── qiu-acp-api.service
```

从参考机拷贝代码（排除密钥与依赖）：

```bash
rsync -a --exclude node_modules --exclude .env --exclude state \
  ~/workdir/qiu-acp-bridge/  TARGET:/home/USER/workdir/qiu-acp-bridge/
```

---

## 4. 安装与配置

### 4.1 依赖

```bash
cd ~/workdir/qiu-acp-bridge
npm install
npx tsc --noEmit
```

### 4.2 `.env`

```bash
cp .env.example .env
chmod 600 .env
```

完整变量说明（按需填写；路径改成目标机）：

```bash
# ========== 飞书桥 ==========
FEISHU_APP_ID=cli_xxxxxxxx
FEISHU_APP_SECRET=xxxxxxxx
ALLOW_FROM=ou_xxxx,ou_yyyy          # DM 白名单 open_id；空=不限制（不推荐）
REQUIRE_MENTION=true
GROUP_POLICY=open

ACP_BIN=/home/USER/.local/bin/agent
ACP_CWD=/home/USER/.cursor/agents/qiu
ACP_MODE=agent
# CURSOR_API_KEY=                   # 未 agent login 时可选

ACK_REACTION=true
ACK_EMOJI=Typing

STATE_DIR=/home/USER/workdir/qiu-acp-bridge/state
LOG_LEVEL=info

# ========== OpenAI 兼容 API（WebUI）==========
API_HOST=0.0.0.0                    # 本机+192.168+Tailscale；仅本机可改 127.0.0.1
API_PORT=18791
API_TOKEN=$(openssl rand -hex 24)   # 长随机串
API_MODEL_ID=atom-cursor-acp
API_ACP_CWD=/home/USER/.cursor/agents/acp-webui
API_STATE_FILE=/home/USER/workdir/qiu-acp-bridge/state/webui-sessions.json
```

生成 token 示例：

```bash
openssl rand -hex 24
```

### 4.3 Agent 目录

```bash
# 飞书人设（从参考机 rsync 或手写）
mkdir -p ~/.cursor/agents/qiu/{memory,media}

# WebUI：空目录即可，不要放 IDENTITY/SOUL
mkdir -p ~/.cursor/agents/acp-webui
```

飞书人设里务必写明 Runtime 是 **Cursor ACP 桥**，避免模型自称 OpenClaw。

---

## 5. systemd 常驻

编辑 `deploy/*.service`（或复制后改）：

- `WorkingDirectory=` → 目标机仓库路径  
- `PATH=` → 含本机 `node` 目录与 `~/.local/bin`  
- `ExecStart=` → **绝对路径** 的 `node` + `…/node_modules/tsx/dist/cli.mjs` + 入口脚本  

```bash
which node
# 例：/home/USER/.nvm/versions/node/v24.x.x/bin/node

cp ~/workdir/qiu-acp-bridge/deploy/qiu-acp-bridge.service ~/.config/systemd/user/
cp ~/workdir/qiu-acp-bridge/deploy/qiu-acp-api.service    ~/.config/systemd/user/
# 按上表改路径后：
systemctl --user daemon-reload
systemctl --user enable --now qiu-acp-bridge.service   # 只要飞书
systemctl --user enable --now qiu-acp-api.service      # 只要 WebUI API
# 关机后仍跑用户服务：
# loginctl enable-linger $USER
```

前台调试：

```bash
npm start              # 飞书
npm run start:api      # API
```

日志：

```bash
journalctl --user -u qiu-acp-bridge -f
journalctl --user -u qiu-acp-api -f
```

---

## 6. 会话与 `/clear`

| 通道 | Session 键 | `/clear`（及 `/new` `/reset`） |
|------|------------|-------------------------------|
| 飞书 | `chat_id::sender_id` | 删桥侧映射；**下一条消息**才 `session/new` |
| WebUI | `webui::<X-Chat-Id\|user\|首条user哈希>` | 同上 |

- `/clear` **不会**重启 `agent acp` 进程。  
- 清的是 **对话 session**，不是人设目录；WebUI 空 cwd 本来也没有跨会话 memory 约定。  
- 飞书与 WebUI **互不清、互不串**。

---

## 7. 接入 Open WebUI

### 7.1 启动容器（示例）

参考机脚本：`~/workdir/workboard/notes/scripts/launcher/start_webui.sh`

要点：

- 端口：`13000→8080`（映射 `0.0.0.0`，故 `192.168` / `100.x` 可打开 WebUI）  
- `--add-host=host.docker.internal:host-gateway`  
- `OPENAI_API_BASE_URLS` 追加：`http://host.docker.internal:18791/v1`  
- `OPENAI_API_KEYS` 对应位置填 **同一** `API_TOKEN`

### 7.2 关键坑：环境变量 ≠ 模型列表

Open WebUI 会把连接写入 Docker volume `open-webui` 内的 **`webui.db`**（键 `openai.api_base_urls`）。

- **只改容器 env、保留旧 volume** → UI 里往往仍是旧连接，**看不到 `atom-cursor-acp`**。  
- 处理：WebUI **设置 → 连接 → OpenAI → 新增**  
  - URL：`http://host.docker.internal:18791/v1`  
  - 密钥：`API_TOKEN`  
  - 保存后刷新模型列表  

从宿主机直连冒烟：

```bash
export API_TOKEN=…   # 与 .env 一致
curl -sS http://127.0.0.1:18791/v1/models -H "Authorization: Bearer $API_TOKEN"
curl -sS http://192.168.x.x:18791/v1/models -H "Authorization: Bearer $API_TOKEN"
curl -sS http://100.x.x.x:18791/v1/models -H "Authorization: Bearer $API_TOKEN"
```

### 7.3 WebUI 访问地址

容器健康后：

- 本机：`http://127.0.0.1:13000`  
- 局域网：`http://<本机192.168.*>:13000`  
- Tailscale：`http://<本机100.*>:13000`  

（启动脚本若只打印 `127.0.0.1`，不代表不能用其它 IP。）

---

## 8. 飞书侧注意

1. 启用本桥前，停掉 OpenClaw（或其它服务）对该飞书 App 的长连接，避免抢事件。  
2. DM 走 `ALLOW_FROM`；群聊默认需 @ 机器人（`REQUIRE_MENTION=true`）。  
3. 「Typing」表情在 **桥适配层** 实现，不是 Cursor agent。  
4. 桥会自动 `allow-always` 权限，并自动应答 `ask_question` / `create_plan`，避免无人值守卡死。

---

## 9. 验收清单

- [ ] `systemctl --user is-active qiu-acp-bridge qiu-acp-api` → 需要的服务为 `active`  
- [ ] `ps` 中可见对应数量的 `agent … acp`；`readlink /proc/<pid>/cwd` 分别为 `…/qiu` 与 `…/acp-webui`  
- [ ] `acp-webui` 下 **无** IDENTITY/SOUL（除非你主动放了）  
- [ ] `curl …:18791/v1/models` + Bearer → 含 `atom-cursor-acp`  
- [ ] Open WebUI 模型可选 `atom-cursor-acp`，多轮可工具执行；响应无 thinking 堆砌  
- [ ] 飞书与 WebUI 对话互不污染；一侧 `/clear` 不影响另一侧  
- [ ] 停 `qiu-acp-api` 后飞书仍正常（若两边都部署了）

---

## 10. 安全边界

| 项 | 建议 |
|----|------|
| `API_TOKEN` | 长随机；只放 `.env` / 容器环境；不进 git |
| `API_HOST=0.0.0.0` | 仅信任内网 / Tailscale；不要端口映射到公网 |
| ACP `allow-always` | 等于本机完整 agent 执行权；token 泄漏 = 本机被远程用 |
| 飞书密钥 | 仅 `.env`；App 与其它机器人隔离 |

---

## 11. 本机参考值（spark-c3b6，换机请改）

| 项 | 值 |
|----|-----|
| 用户 | `atom` |
| 仓库 | `/home/atom/workdir/qiu-acp-bridge` |
| 飞书 cwd | `/home/atom/.cursor/agents/qiu` |
| WebUI cwd | `/home/atom/.cursor/agents/acp-webui` |
| API | `0.0.0.0:18791`，模型 `atom-cursor-acp` |
| WebUI | Docker `open-webui`，`0.0.0.0:13000` |
| 示例局域网 IP | `192.168.1.5` |
| 示例 Tailscale IP | `100.97.133.46` |
| systemd | `qiu-acp-bridge.service`、`qiu-acp-api.service`（user） |

仓库内更细文档：

- 飞书：`qiu-acp-bridge/docs/cursor-acp-feishu-guide.md`  
- WebUI：`qiu-acp-bridge/docs/cursor-acp-webui.md`  

---

## 12. 可选：只部署其中一半

- **只要飞书**：配 `FEISHU_*` + `ACP_CWD=…/qiu`，只 enable `qiu-acp-bridge`。  
- **只要 WebUI agent**：不必飞书变量也能起 API——但当前 `config.ts` 在飞书入口会 `required(FEISHU_*)`；**只跑 API** 时用 `npm run start:api` / `qiu-acp-api.service`（读 `api-config.ts`，不强制飞书密钥）。仍建议 `.env` 里留好 `API_*`。  
