# Ubuntu DNS 解析故障完整排查与解决方案

## 目录

1. [Quick-Start](#0-quick-start)
2. [问题现象](#1-问题现象)
3. [核心原因分析](#2-核心原因分析)
4. [排查步骤](#3-排查步骤)
5. [解决方案](#4-解决方案)
6. [DNS 顺序与上网速度的关系](#5-dns-顺序与上网速度的关系)
7. [Tailscale 兼容性说明](#6-tailscale-兼容性说明)
8. [验证与测试](#7-验证与测试)
9. [常见问题 FAQ](#8-常见问题-faq)
10. [命令速查表](#9-命令速查表)

---

## 0. Quick-Start

就下面几步：

```bash
sudo rm /etc/resolv.conf
sudo ln -sf /run/systemd/resolve/stub-resolv.conf /etc/resolv.conf

# 重启服务
sudo systemctl restart systemd-resolved
sudo systemctl restart NetworkManager         # 如果你使用 NetworkManager
sudo systemctl restart tailscaled
```

## 1. 问题现象

- 浏览器或终端中访问域名（如 `baidu.com`）无法打开；
- 使用 `ping 8.8.8.8` 可以成功，说明网络层连通；
- 使用 `nslookup baidu.com` 或 `dig baidu.com` 解析失败或超时；
- 或者：能上网但每次打开网页都有明显卡顿延迟。

> **典型特征**：能 ping 通 IP，但无法解析域名 → 问题定位在 DNS 解析环节。

---

## 2. 核心原因分析

### 2.1 DNS 配置问题

- tailscale 网络影响了；
- DNS 服务器配置错误或不可达；
- `/etc/resolv.conf` 被错误修改或未指向正确的 DNS 服务；
- 内网 DNS（公司、VPN、Tailscale）无法解析公网域名；
- systemd-resolved 服务未正确运行或配置文件冲突。

---

## 3. 排查步骤

### 3.1 确认网络基本连通性

```bash
ping 8.8.8.8
```

- ✅ 通则网络层正常，问题集中在 DNS。
- ❌ 不通则需先检查网卡、路由、网关等。

### 3.2 查看当前 DNS 配置

```bash
cat /etc/resolv.conf
resolvectl status   # 推荐，信息更全面
```

### 3.3 测试特定 DNS 服务器

```bash
nslookup baidu.com 8.8.8.8      # 测试 Google DNS
nslookup baidu.com 10.0.0.98    # 测试公司 DNS
```

### 3.4 检查 systemd-resolved 状态

```bash
systemctl status systemd-resolved
```

### 3.5 检查 `/etc/resolv.conf` 是否为正确的软链接

```bash
ls -l /etc/resolv.conf
```

期望输出：

```
/etc/resolv.conf -> /run/systemd/resolve/stub-resolv.conf
```

---

## 4. 解决方案

### 方案一：临时添加 DNS（用于快速验证）

```bash
sudo sh -c "echo 'nameserver 8.8.8.8' >> /etc/resolv.conf"
```

> ⚠️ 重启网络或系统后可能失效。

### 方案二：永久配置（推荐，适用于 Ubuntu 18.04+）

#### 4.2.1 确保 systemd-resolved 接管

```bash
sudo systemctl enable --now systemd-resolved
sudo ln -sf /run/systemd/resolve/stub-resolv.conf /etc/resolv.conf
```

#### 4.2.2 编辑配置文件

```bash
sudo nano /etc/systemd/resolved.conf
```

写入以下内容（根据实际需求调整顺序）：

```ini
[Resolve]
DNS=10.0.0.98 10.0.0.100
FallbackDNS=8.8.8.8
```

- `DNS=` → 首选 DNS（公司内网 DNS）
- `FallbackDNS=` → 当首选 DNS 均失败时使用（公网 DNS）

#### 4.2.3 重启服务

```bash
sudo systemctl restart systemd-resolved
```

#### 4.2.4 验证配置

```bash
resolvectl status | grep -A10 "Global"
```

应能看到 `DNS Servers` 和 `Fallback DNS Servers` 输出正确地址。

### 方案三：图形界面配置（桌面版）

1. 打开 **设置** → **网络**；
2. 选择当前连接，点击齿轮图标；
3. 在 **IPv4** 选项卡中，将 DNS 改为“手动”；
4. 按顺序填入 DNS：`10.0.0.98, 10.0.0.100, 8.8.8.8`；
5. 保存并重新连接网络。

---

## 5. DNS 顺序与上网速度的关系

### 5.1 问题现象

- 配置了多个 `nameserver`，其中第一个（或前几个）是无效的；
- 每次访问域名时感觉**明显卡顿**，网页加载缓慢；
- 最终仍能打开网页，但延迟很高。

### 5.2 原因分析

#### DNS 解析器的超时等待机制

- 向第一个 `nameserver` 发送查询请求；
- 如果服务器**没有响应**，解析器等待一个**超时时间**（通常 **2~5 秒**）；
- 超时后，才会转向下一个 `nameserver`。

#### 累积效应

```
总延迟 = 超时时间 × 需要解析的不同域名数量
```

例如：超时 2 秒，网页包含 20 个不同域名 → 仅 DNS 部分就可能额外增加 **40 秒** 的等待时间。

### 5.3 解决方案


| 场景                    | 推荐做法                             |
| --------------------- | -------------------------------- |
| DNS 已经失效（IP 不通、服务下线）  | **直接删除或注释**，避免无意义等待              |
| DNS 只对特定域名有效（如内网 DNS） | **使用条件转发**，或将其放在**最后一位**         |
| 所有 DNS 都有效，但响应速度不同    | **把最快的放在最前面**                    |
| 不确定哪个 DNS 有效          | 先用 `dig @dns_ip domain` 逐个测试，再配置 |


### 5.4 条件转发配置示例

```bash
sudo nano /etc/systemd/resolved.conf
```

```ini
[Resolve]
DNS=8.8.8.8
FallbackDNS=114.114.114.114

# 为特定域名指定专用 DNS
[Resolve]
Domains=~ts.net
DNS=100.100.100.100
```

---

## 6. Tailscale 兼容性说明

### 6.1 结论

采用上述 `systemd-resolved` 配置方式，Tailscale **不会冲突**，反而能和谐共存。

### 6.2 原理

- Tailscale 检测到 `systemd-resolved` 正在运行时，不会强行修改 `/etc/resolv.conf`；
- 通过 D-Bus 接口将 `100.100.100.100` 注册到 `systemd-resolved`；
- 效果：
  - 对内（Tailscale 网络）：MagicDNS 查询发送给 `100.100.100.100`，解析 `.ts.net` 域名；
  - 对外（公司内网）：使用公司 DNS 解析内网服务；
  - 兜底（公网）：使用 `FallbackDNS` 如 `8.8.8.8`。

### 6.3 确保 Tailscale 与 systemd-resolved 兼容的操作

```bash
# 修复软链接
sudo ln -sf /run/systemd/resolve/stub-resolv.conf /etc/resolv.conf

# 重启服务
sudo systemctl restart systemd-resolved
sudo systemctl restart tailscaled
```

### 6.4 验证 Tailscale DNS

```bash
resolvectl status | grep -A5 "tailscale0"
```

应出现类似：

```
Link 5 (tailscale0)
      DNS Servers: 100.100.100.100
      DNS Domain: ~.
```

---

## 7. 验证与测试

### 7.1 测试域名解析

```bash
nslookup baidu.com
dig baidu.com
ping -c 3 baidu.com
```

### 7.2 查看实际使用的 DNS

```bash
resolvectl query baidu.com
```

### 7.3 测试 DNS 响应时间

```bash
# 测试无效 DNS（会明显卡住几秒）
time dig @10.251.1.1 baidu.com

# 测试有效 DNS（快速返回）
time dig @8.8.8.8 baidu.com
```

### 7.4 刷新 DNS 缓存

```bash
sudo systemd-resolve --flush-caches
# 或
sudo resolvectl flush-caches
```

---

## 8. 常见问题 FAQ

**Q：我修改了 `/etc/systemd/resolved.conf` 但 `resolvectl status` 没变化？**  
A：确认已重启服务 `sudo systemctl restart systemd-resolved`，并且 `/etc/resolv.conf` 正确指向了存根文件。

**Q：添加 `FallbackDNS` 后，公司 DNS 解析失败时会自动用 8.8.8.8 吗？**  
A：是的。`systemd-resolved` 会按顺序尝试 `Link` DNS → `Global` DNS → `FallbackDNS`。

**Q：用了公司 DNS 还能解析 Tailscale 内网主机名吗？**  
A：可以。Tailscale 会自动将 `100.100.100.100` 添加到对应网卡的 DNS 列表中。

**Q：重启后配置丢失怎么办？**  
A：检查是否由 NetworkManager 或 dhclient 动态覆写了 `/etc/resolv.conf`。推荐使用 `systemd-resolved` 方案，并确认软链接正确。

**Q：为什么系统不“记住”哪个 DNS 是坏的，下次跳过它？**  
A：标准的 DNS 解析器（如 glibc）默认不做动态故障检测。`systemd-resolved` 有一定优化，但仍建议将无效 DNS 移出配置。

**Q：如何彻底避免 DNS 延迟影响上网体验？**  
A：使用本地 DNS 缓存服务（如 `systemd-resolved`），确保只配置可靠、响应快的 DNS 服务器，将无效或慢速 DNS 移除或移至末尾。

---

## 9. 命令速查表


| 命令                                                                   | 作用          |
| -------------------------------------------------------------------- | ----------- |
| `cat /etc/resolv.conf`                                               | 查看当前 DNS 配置 |
| `resolvectl status`                                                  | 查看详细 DNS 状态 |
| `systemctl status systemd-resolved`                                  | 检查 DNS 服务状态 |
| `sudo systemctl restart systemd-resolved`                            | 重启 DNS 服务   |
| `sudo systemd-resolve --flush-caches`                                | 刷新 DNS 缓存   |
| `nslookup baidu.com`                                                 | 测试域名解析      |
| `dig baidu.com`                                                      | 详细 DNS 查询   |
| `resolvectl query baidu.com`                                         | 查看实际使用的 DNS |
| `time dig @8.8.8.8 baidu.com`                                        | 测试 DNS 响应时间 |
| `sudo ln -sf /run/systemd/resolve/stub-resolv.conf /etc/resolv.conf` | 修复软链接       |


---

## 10. 总结

- **问题核心**：DNS 配置缺失、无效或顺序不当；
- **推荐方案**：使用 `systemd-resolved` 管理 DNS，将有效的公司 DNS 设为首选，公网 DNS（如 8.8.8.8）设为备用；
- **顺序优化**：无效或慢速 DNS 会严重影响上网速度，应删除或移至末尾；
- **与 Tailscale 共存**：只需保证 `systemd-resolved` 处于活动状态且 `/etc/resolv.conf` 为软链接，Tailscale 会自动适配；
- **持久化要点**：编辑 `/etc/systemd/resolved.conf`，然后重启服务。

按照本文档操作，绝大多数 DNS 解析问题（无论是完全无法上网还是上网慢）都能解决。若仍有异常，请检查防火墙规则或公司网络策略是否拦截了外部 DNS 请求。