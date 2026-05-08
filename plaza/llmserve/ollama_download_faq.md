# Ollama 安装与下载 FAQ（本次环境处理记录）

面向 Linux（Ubuntu）ARM64（aarch64）环境，记录一次「已有不完整/异常安装 → 恢复可用」的完整过程，便于日后遇到同类问题时查阅。

---

## 第 1 部分：最终操作步骤（按顺序可做）

以下为本次环境中**最终确认有效**的步骤；若你已能从官方渠道完整安装，可只参考与「权限、服务」相关的几条。

### 1. 检查是否已安装

```bash
which ollama
ollama --version
```

若提示「权限不够」，继续第 2 步；若段错误，可能是二进制损坏或库不全，需重新安装（见文末网络正常时的标准安装）。

### 2. 修正可执行权限（若 `ollama` 仅 root 可执行）

曾出现 `/usr/local/bin/ollama` 权限为 `700`（`-rwx------`），普通用户无法执行：

```bash
sudo chmod 755 /usr/local/bin/ollama
```

### 3. 优先使用官方一键安装（网络正常时）

```bash
curl -fsSL https://ollama.com/install.sh | sudo sh
```

脚本会将程序装到 `/usr/local`，并处理 systemd 用户/服务（视脚本版本而定）。

### 4. 启动并检查服务

```bash
sudo systemctl enable ollama   # 若尚未启用开机自启
sudo systemctl start ollama
systemctl is-active ollama
```

### 5. 验证 CLI 与守护进程

```bash
ollama --version
ollama list
```

若见 `Warning: could not connect to a running Ollama instance`，说明 CLI 正常但本机 **11434** 上服务未起或未监听，回到第 4 步检查 `ollama.service`。

### 6. 日常使用

```bash
ollama pull <模型名>
ollama run <模型名>
```

---

## 第 2 部分：中间遇到的问题与尝试过的下载/安装方式

### 2.1 初始现象

| 现象 | 说明 |
|------|------|
| `ollama` 在 PATH 中 | `/usr/local/bin/ollama` 存在 |
| 普通用户执行失败 | 二进制权限为 `700`，报错「权限不够」 |
| 一度 `ollama --version` 段错误 | 疑似不完整/损坏的安装或短暂不一致状态；后续同路径二进制更新为可用后，`--version` 正常 |

### 2.2 权限修复后仍怀疑二进制异常

在将权限改为 `755` 后，仍出现段错误（exit 139），因此判断需要**重新拉取官方包**或通过官方脚本覆盖安装。

### 2.3 下载/安装方式更换记录

以下为按时间顺序的尝试及结果（**高度依赖当前网络质量**）：

1. **官方安装脚本（推荐方式）**  
   - 命令：`curl -fsSL https://ollama.com/install.sh | sudo sh`  
   - 行为：会下载 `ollama-linux-arm64.tar.zst` 等平台相关包。  
   - 本次：**下载极慢**，长时间仅个位数进度，实际无法在合理时间内完成（环境网络至 `ollama.com` 不畅）。

2. **Snap 安装（备选）**  
   - 命令：`sudo snap install ollama`  
   - 本次失败原因：Snap 报错类似 **下载速度过慢（如 0 bytes/sec）**，无法完成 snap 包拉取。

3. **GitHub Releases 直链（手动大块下载）**  
   - 示例资源：`https://github.com/ollama/ollama/releases/download/v0.23.0/ollama-linux-arm64.tar.zst`（需与机器架构一致；ARM64 用 `arm64`）。  
   - 本次：`curl` 下载至约 **49MiB / ~1.2GiB** 后**长时间无新增流量**，相当于停滞，未能完成整包。

4. **最终落地状态**  
   - 磁盘上已存在较完整的 `/usr/local/bin/ollama`（例如版本 **0.23.0**），在权限修正与（可能由用户侧其它途径完成的）二进制就位后，**CLI 与 `systemd` 服务均可正常工作**。  
   - 结论：**问题根源混合了「权限」与「网络导致无法可靠重装」**；在网络恢复后，仍建议再跑一次官方安装脚本，以保证 `/usr/local/lib/ollama` 等依赖目录与版本一致。

### 2.4 与本环境相关的额外信息

- **发行版**：Ubuntu。  
- **架构**：ARM64（日志中出现 `ollama-linux-arm64`）。  
- **systemd 服务**：`ollama.service` 使用 `ExecStart=/usr/local/bin/ollama serve`；停止服务使用 `sudo systemctl stop ollama`。  
- **APT 默认仓库**：未发现可直接 `apt install` 的同名官方包（需以脚本、Snap 或手动解压安装为主）。

### 2.5 若再次遇到「下载极慢或中断」可做的选择

- 换网络/VPN/代理后再执行官方脚本或 `curl`/`wget` 断点续传。  
- 在网速稳定的机器上下好对应平台的 `*.tar.zst`，再用 `scp`、U 盘等拷贝到目标机，按 [Ollama 官方文档](https://github.com/ollama/ollama) 手工解压到 `/usr/local`（或与现有布局一致的路径），并确保 `ollama` 可执行权限与 `ollama` 用户、systemd 单元配置一致。  
- Snap 仅在网络可达 Snap Store 且速度正常时作为备选。

---

*文档根据单次真实排障过程整理，版本号与包体大小随 Ollama 发布而变，请以官方 Release 为准。*
