## 下载模型
本地/data/models/hub目录下
**注意**： 需要vpn连hugging face
```
sudo hf download Qwen/Qwen3.6-35B-A3B \
  --local-dir /root/models/hub/Qwen3.6-35B-A3B
```
或
```
sudo /home/atom/miniforge3/bin/modelscope download --model Qwen/Qwen3.6-35B-A3B  --local_dir /data/models/hub/Qwen3-6-35B-A3
```
### 模型列表
- https://huggingface.co/Qwen/Qwen3.6-35B-A3B
- https://www.modelscope.cn/models/Qwen/Qwen3.6-35B-A3B
- https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
- https://www.modelscope.cn/models/unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
- https://huggingface.co/RedHatAI/Qwen3.5-122B-A10B-NVFP4

## 下载sglang 
启动容器
```
docker run --gpus all -it --rm \
  -p 30000:30000 \
  -v /data/models:/root/.cache/huggingface \
  --network host \
  --ipc=host \
  lmsysorg/sglang:nightly-dev-cu13-20260416-a4cf2ea1 bash
```

## 启动模型
进入容器，启动模型
```
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3.6-35B-A3B \
  --host 0.0.0.0 \
  --port 30000 \
  --trust-remote-code \
  --tp 1 \
  --mem-fraction-static 0.75 \
  --context-length 131072
```
**注意**：`    --attention-backend flashinfer` 怀疑是GB10和torch不兼容，不能加，会卡死。

一行命令，用modelscope
```
docker run  -d --gpus all --rm --network host --ipc=host -v /data/models/modelscope:/root/.cache/modelscope/hub/models -e SGLANG_USE_MODELSCOPE=true  --name sglang-server lmsysorg/sglang:nightly-dev-cu13-20260416-a4cf2ea1 python3 -m sglang.launch_server  --model-path Qwen3-6-35B-A3 --host 0.0.0.0 --port 30000 --trust-remote-code --tp 1 --mem-fraction-static 0.75 --context-length 131072
```
其他：

| 参数                | 功能                           |
| ----------------- | ---------------------------- |
| `--no-stream`     | 关闭 streaming 输出，避免后台推理线程持续占用 |
| `--reasoning-parser none` | 关闭自动 Think / 预计算             |
| `--tool-call-parser none` | 关闭自动 工具调用             |
| `--lazy-load`     | 按需加载模型，模型不立即占用 GPU，只有收到请求才加载 |




## 安装webui
之前docker pull的open-webui:ollama 自带ollama。
后面就统一改成
```
docker run -d \
  -p 8080:8080 \
  -v open-webui:/app/backend/data \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:30000/v1 \
  -e OPENAI_API_KEY=sk-anything \
  --name open-webui \
  ghcr.io/open-webui/open-webui:main


docker run -d --network host -v open-webui:/app/backend/data -e OPENAI_API_BASE_URL=http://127.0.0.1:30000/v1 -e OPENAI_API_KEY=sk-anything --name open-webui-main ghcr.io/open-webui/open-webui:main
```
注意： host.docker.internal 是容器和主机的默认路径，不要动

### webui 设置
进入 WebUI： `Settings → Models → Add OpenAI-compatible API`，
填写：
- Base URL: `http://你的IP:30000/v1`
- API Key: `随便填（SGLang不校验）`
- Model: `Qwen/Qwen3.6-35B-A3B`

## 故障处理
### sglang容器下载模型异常
官方教程docker pull sglang:spark，这个版本太老，1. 报错torch版本太老，2. 不支持moe架构的qwen3.6:35b。
nemotron的介绍中推荐容器 `docker pull lmsysorg/sglang:nightly-dev-cu13-20260316-d852f26c`，找不到。
找到个类似的 `lmsysorg/sglang:nightly-dev-cu13-20260416-a4cf2ea1`

即使用正确的容器，PyTorch 仍然会发出错误的警告，不影响：https://github.com/sgl-project/sglang/issues/11658 


SGLANG_USE_MODELSCOPE=true python -m sglang.launch_server --model-path Qwen/Qwen3.6-35B-A3B-FP8 --port 8000 --tp-size 8 --mem-fraction-static 0.8 --context-length 262144 --reasoning-parser qwen3


sudo modelscope download --model Qwen/Qwen3.5-122B-A10B --local_dir /data/models/modelscope/Qwen3.5-122B-A10B


sudo modelscope download --model unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 --local_dir /data/models/modelscope/Nemotron-3-Super-120B-A12B-NVFP4