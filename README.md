# notes 说明

本目录存放本地运维脚本与知识笔记：`scripts/` 为可执行 Shell 脚本，`docs/` 为 Markdown 文档。

## 脚本

| 中文名称 | 描述 | 用法 |
| --- | --- | --- |
| [Code Server 网页 IDE 启动](scripts/code_server_start.sh) | 以 Docker 后台运行 `codercom/code-server`，映射端口 `12200→8080`，挂载 `/home/atom/code`，密码见脚本内环境变量。 | `bash scripts/code_server_start.sh` |
| [Tailscale 出口节点 VPN](scripts/vpn_start.sh) | 通过指定 exit node 连接/断开 Tailscale，并显示 `tailscale status` 与当前公网出口 IP。 | `bash scripts/vpn_start.sh start` 或 `bash scripts/vpn_start.sh stop` |
| [vLLM 启动 Qwen3.6-35B](scripts/vllmstart_qwen36-35b.sh) | 若容器 `vllm-qwen3.6-35b` 已存在则 `docker start`，否则创建并启动 vLLM OpenAI 兼容服务（端口 8000，ModelScope 模型路径等）。需环境变量 `HF_TOKEN`（若上游需要）。 | `bash scripts/vllmstart_qwen36-35b.sh` |
| [SGLang 启动 Qwen3.6-35B](scripts/start_qwen36-35b.sh) | 若容器 `qwen36-35b` 已存在则启动，否则以 `sglang.launch_server` 拉起服务（host 网络、端口 30000，ModelScope 挂载）。 | `bash scripts/start_qwen36-35b.sh` |
| [SGLang Nemotron3 压测容器](scripts/test_nemotron3.sh) | 与正式推理同名容器逻辑类似，但入口为 `sglang.bench_serving`，用于基准测试场景。 | `bash scripts/test_nemotron3.sh` |
| [SGLang 启动 Nemotron3 Super 120B](scripts/start_nemotron3-super-120b-a12b-nvfp4.sh) | 若容器 `nemotron-120b-a12b` 已存在则启动，否则以 `sglang serve` 部署 NVFP4 模型（FlashInfer、modelopt_fp4 等）。 | `bash scripts/start_nemotron3-super-120b-a12b-nvfp4.sh` |
| [vLLM 启动 Nemotron3 Super 120B](scripts/vllmstart_nemotron3-super-120b-a12b-nvfp4.sh) | 若容器 `vllm-nemotron-120b-a12b` 已存在则启动，否则以 vLLM 镜像部署 FP4 模型（端口 8000）。需 `HF_TOKEN` 等环境变量与模型挂载。 | `bash scripts/vllmstart_nemotron3-super-120b-a12b-nvfp4.sh` |
| [SGLang 启动 Qwen3.5-122B-A10B](scripts/start_qwen35-122b-a10b.sh) | 若容器 `Qwen35-122B-A10B` 已存在则启动，否则以 `sglang.launch_server` 加载 NVFP4 权重（端口 30000）。 | `bash scripts/start_qwen35-122b-a10b.sh` |
| [Open WebUI 启动](scripts/start_webui.sh) | 若容器 `open-webui` 已存在则启动，否则运行 Open WebUI 镜像，API 指向 `host.docker.internal:30000`。 | `bash scripts/start_webui.sh` |

## 知识文档

| 中文名称 | 摘要 |
| --- | --- |
| [Spark 上安装与使用 SGLang](docs/spark上安装sglang.md) | 模型下载（Hugging Face / ModelScope）、SGLang 镜像与容器、`launch_server` 启动命令与参数说明。 |
| [SGLang + Qwen3.6-35B 压测](docs/benchmark_sglang_qwen36_35b.md) | 后端容器启动、`sglang.bench_serving` 与 ShareGPT 数据集用法及 benchmark 结果记录。 |
| [知识链接归档](docs/知识链接归档.md) | 外链备忘（如 Claude 源码解析相关链接）。 |
| [vLLM + Qwen3.6 压测](docs/benchmark_vllm_qwen36.md) | vLLM 服务容器与容器内 `vllm bench serve` 压测步骤及结果。 |
| [Spark 上 vLLM 部署 Nemotron3](docs/spark上vllm部署nemotron3.md) | 官方与魔塔文档要点、镜像与 ModelScope 模型下载、完整 Docker 启动示例。 |
| [vLLM Nemotron3 压测](docs/benchmark_vllm_nemotron3.md) | 以 `vllm bench serve` 为主的压测命令与结果整理。 |
| [AI-Infra 与 vLLM 八周学习计划](docs/瑞康-0417-study.md) | 约 56 天每日阅读清单：论文、文档、仓库与量化 / GPU / 推理栈主题。 |
| [SGLang + Nemotron3 压测](docs/benchmark_sglang_nemotron3.md) | Nemotron3 FP4 下 SGLang 服务与 `bench_serving` 多轮测试与数据记录。 |
| [Spark 实验与任务清单](docs/2016Q2计划.md) | WebUI、Tailscale、模型与 Agent、微调与 Spark 教程链接等待办勾选列表。 |
