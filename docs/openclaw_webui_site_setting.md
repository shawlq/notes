# OpenClaw Web UI 局域网 / Tailscale 访问设置说明

本文记录将 OpenClaw Control UI（Web UI）从仅本机 `127.0.0.1` 改为可通过**局域网 IP** 或 **Tailscale IP** 访问（端口 **18789**），以及登录相关问题与 Gateway 重启方式。配置文件默认路径：`~/.openclaw/openclaw.json`（JSON5，可用 `openclaw config validate` 校验）。

---

## 1. 目标

| 项目 | 说明 |
|------|------|
| 端口 | 保持 **18789**（与默认一致，也可由 `gateway.port` 或环境变量覆盖） |
| 监听范围 | 不仅监听回环接口，使 **`http://<本机局域网 IP>:18789`**、**`http://<Tailscale IPv4>:18789`** 可访问 |
| 认证 | 使用 Gateway 的 **token** 模式（`gateway.auth.mode: "token"`） |

---

## 2. 监听地址：`gateway.bind`

- **原值**：`"loopback"` — 仅本机回环，外网/其他设备无法直连。
- **修改为**：`"lan"` — 等价于在 **`0.0.0.0`** 上监听，本机所有网卡（包括以太网/Wi‑Fi 局域网、Tailscale 虚拟网卡等）上的 **18789** 均可接受连接。

在 `openclaw.json` 的 `gateway` 段中设置：

```json
"gateway": {
  "port": 18789,
  "bind": "lan"
}
```

说明：若只需监听 Tailscale 接口，可选用文档中的 `"tailnet"`；需要**同时**支持局域网与 Tailscale 时，一般用 **`lan`** 即可（一台机器上 Tailscale 与局域网通常是不同 IP，但同属“对外监听”场景）。

---

## 3. WebSocket / 浏览器 Origin：`gateway.controlUi.allowedOrigins`

在非回环地址打开页面时，浏览器 **Origin** 会随访问 URL 变化（例如不同局域网 IP、不同 `100.x.x.x`）。Control UI 使用 WebSocket，需在配置中声明允许的来源。

为免每次换 IP 都改配置，可采用通配放行（**仅限信任内网**，且建议配合强 token）：

```json
"gateway": {
  "controlUi": {
    "allowedOrigins": ["*"]
  }
}
```

更稳妥的做法是列出**固定**的完整 Origin（示例）：`http://192.168.1.10:18789`、`http://100.64.0.2:18789` 等。

---

## 4. 放宽非标准环境下的认证检查：`allowInsecureAuth`

在部分代理或非标准部署下，可开启：

```json
"controlUi": {
  "allowInsecureAuth": true
}
```

该选项**不能单独解决**下文“设备身份 / 安全上下文”问题。

---

## 5. 登录报错：device identity / HTTPS / localhost

通过 **`http://` + 非 localhost 的 IP**（如 `http://192.168.x.x:18789`）访问时，浏览器**不处于**安全上下文，Control UI 可能提示类似：

> control ui requires device identity (use HTTPS or localhost secure context)

处理方式（二选一或组合）：

### 5.1 调试/信任内网：关闭设备身份校验（仅依赖 token）

在 `gateway.controlUi` 中设置：

```json
"dangerouslyDisableDeviceAuth": true
```

含义：跳过 Control UI 的**设备身份**步骤，**仅依赖 token/密码**。官方文档/schema 中将其标为高风险选项，适合**短期调试**或**明确信任的网络**；用完可改回 `false`。

### 5.2 更安全的方式（推荐长期）

- **SSH 本地转发**：在客户端执行端口转发，浏览器只用 **`http://127.0.0.1:18789`**（属于安全上下文），例如：
  ```bash
  ssh -L 18789:127.0.0.1:18789 用户@目标主机
  ```
- **HTTPS**：为 Gateway 配置 TLS（如 `gateway.tls`）或在前面加 **HTTPS 反代**，用 `https://` 访问。

---

## 6. 修改登录 Token

在 `gateway.auth` 中保持 `mode: "token"`，并设置你的共享口令：

```json
"auth": {
  "mode": "token",
  "token": "<你的 token>"
}
```

修改后需让配置生效（见下一节）。**切勿**将真实 token 提交到公开仓库；本文档不记录具体口令。

---

## 7. 重启 Gateway 使配置生效

监听地址、认证与 Control UI 相关项变更后，一般需要**重启 Gateway**。

### 7.1 已安装 systemd 用户服务（常见）

```bash
openclaw gateway restart
```

或：

```bash
systemctl --user restart openclaw-gateway.service
```

查看状态：

```bash
openclaw gateway status
```

### 7.2 前台运行

若使用 `openclaw gateway run` 手工启动，在终端 **Ctrl+C** 停止后再次执行 `openclaw gateway run`。

---

## 8. 校验与文档命令

```bash
openclaw config validate
openclaw config get gateway.bind
openclaw config get gateway.port
```

官方文档索引：

- [Configuration](https://docs.openclaw.ai/configuration)
- [Configuration Reference — Gateway](https://docs.openclaw.ai/gateway/configuration-reference)
- [CLI — gateway](https://docs.openclaw.ai/cli/gateway)

---

## 9. 安全提示（简要）

- `bind: "lan"` 会向**所有接口**开放端口，请配合**防火墙**仅允许需要的网段。
- `allowedOrigins: ["*"]` 与 `dangerouslyDisableDeviceAuth: true` 都会**扩大风险面**，务必在可信网络中使用强 token，并定期审视配置。
- 对外的长期访问优先 **HTTPS** 或 **SSH 转发 + localhost**。

---

*文档生成说明：基于将 Web UI 从本机扩展为局域网/Tailscale 访问、Token 登录与设备身份报错处理等操作整理。*
