# AGENTS.md

## Cursor Cloud specific instructions

### 仓库概述

这是一个 **运维脚本与知识笔记仓库**（非传统应用代码库）。内容包括：

- `scripts/` — 用于在 NVIDIA GPU 服务器（Spark）上启动 LLM 推理服务的 Bash 脚本
- `docs/` — 部署指南、基准测试结果、学习计划等 Markdown 文档

**没有** `package.json`、`requirements.txt`、`Makefile` 或任何编译构建系统。

### Lint / 检查

- **语法检查**: `bash -n scripts/<script>.sh`
- **静态分析**: `shellcheck scripts/*.sh`（已通过更新脚本自动安装）

### 脚本说明

所有脚本都依赖 Docker + NVIDIA GPU 环境，无法在 Cloud Agent VM 中直接运行。脚本分为以下几类：

| 类别 | 脚本 | 说明 |
|------|------|------|
| 推理引擎 (SGLang) | `start_qwen36-35b.sh`, `start_nemotron3-super-120b-a12b-nvfp4.sh`, `start_qwen35-122b-a10b.sh` | 端口 30000 |
| 推理引擎 (vLLM) | `vllmstart_qwen36-35b.sh`, `vllmstart_nemotron3-super-120b-a12b-nvfp4.sh` | 端口 8000，需要 `HF_TOKEN` |
| 前端 | `start_webui.sh` | Open WebUI，端口 13000 |
| 基准测试 | `test_nemotron3.sh` | SGLang bench_serving |
| 工具 | `code_server_start.sh`, `vpn_start.sh` | IDE 和 VPN |

### 注意事项

- 修改脚本后务必运行 `shellcheck` 进行验证。
- `shellcheck` 当前报告的 info/style 级别问题（如 SC2086、SC2006）属于已知项，不阻塞功能。
- 文档为中文 Markdown，修改后无需特殊构建步骤。
