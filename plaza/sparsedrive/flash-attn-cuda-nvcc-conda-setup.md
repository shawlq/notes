# SparseDrive：`flash-attn` 安装与 CUDA `nvcc`（Conda 方案）

## 第一部分：思路与命令（速查）

### 思路（一句话）

**PyTorch 的 cu116 wheel 自带运行时，不等于本机有可用的 `nvcc`；系统若只有 CUDA 11.5 的 `nvcc`，`flash-attn` 会在 `pip install` 阶段直接失败。** 在 **`sparse`（或你自己的）Conda 环境里安装 `cuda-nvcc` 11.8，并把 `CUDA_HOME` 指到 `CONDA_PREFIX`**，再安装依赖即可。

### 命令（按顺序执行）

```bash
# 1）进入 Python 3.8 环境（示例：sparse）
conda activate sparse

# 2）在环境中安装 11.8 版 nvcc（无需系统级 CUDA 替换）
conda install -y 'cuda-nvcc=11.8.89' -c nvidia

# 3）编译/安装前设置环境变量（手动会话可写这一句；见下文“持久化”）
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"

# 4）按仓库文档先装 PyTorch（若尚未安装）
pip install --upgrade pip
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 \
  --extra-index-url https://download.pytorch.org/whl/cu116

# 5）安装 flash-attn 与整份依赖（若在仓库根目录）
cd /path/to/SparseDrive
pip install 'flash-attn==2.3.2' --no-build-isolation   # 可选：单独先装
pip install -r requirement.txt
```

### 持久化 `CUDA_HOME`（可选，推荐）

在目标 Conda 环境的 `etc/conda/activate.d` / `deactivate.d` 里各放一小段脚本：激活环境时 `export CUDA_HOME="$CONDA_PREFIX"`，退出时恢复原值。这样以后 **`conda activate sparse`** 无需再手动 export。

### 自检

```bash
conda activate sparse
which nvcc && nvcc -V          # 应为 env 内路径，且 release ≥ 11.6（示例 11.8）
echo "$CUDA_HOME"              # 建议为当前 env 前缀
python -c "import flash_attn; print(flash_attn.__version__)"
```

---

## 第二部分：推理与细节说明

### 现象与报错含义

`pip install -r requirement.txt` 在 `flash-attn==2.3.2` 处可能出现：

- `metadata-generation-failed` / `python setup.py egg_info` 失败；
- 核心信息类似：**FlashAttention 仅支持 CUDA 11.6 及以上，请用 `nvcc -V` 确认**；
- 日志里有时还会出现 **`fatal: not a git repository`**，多为构建脚本里顺带调用 `git` 的副作用，**通常不是根因**。

### 为何「torch 已是 cu116」仍会失败

- **CUDA 运行时（随 PyTorch wheel）**：决定 `torch` 在 GPU 上用什么版本的运行时库。
- **CUDA 编译工具链（`nvcc` + 头文件等）**：决定 **源码包**（如 `flash-attn`、`mmcv` 的本地编译）能否通过版本检查并完成编译。

二者解耦：机器上可以 **torch==1.13.0+cu116** 正常 import，但 **`/usr/bin/nvcc` 仍是 11.5**，则 `flash-attn` 的安装脚本按 **`nvcc` 主版本**判断，仍会失败。

### 为何选用 Conda 内的 `cuda-nvcc` 而不是改系统 CUDA

- 不依赖 `sudo` 改系统默认 Toolkit，避免影响其他项目。
- 版本与路径集中在当前 env，`which nvcc` 在激活环境后自然指向 **11.8**。
- **`CUDA_HOME="$CONDA_PREFIX"`** 让多数构建脚本把“Toolkit 根目录”对齐到该环境，减少找错头文件/编译器的问题。

### 与 `MAX_JOBS`、`--no-build-isolation`

- 资源紧张时可设 `MAX_JOBS`（例如 4）降低并行编译压力。
- `flash-attn` 某些环境下使用 `--no-build-isolation` 可避免隔离环境里缺少已安装的 `torch` 等问题；具体是否必须取决于当时的 pip/setuptools 行为，本记录中的实践环境里使用后可正常生成元数据并编出 wheel。

### 项目侧注意点

SparseDrive 在 `projects/mmdet3d_plugin/models/attention.py` 中 **直接依赖 `flash_attn`**，没有无 FlashAttention 的纯 PyTorch 回退路径，因此 **不能只删 `requirement.txt` 里的 `flash-attn` 来绕过安装**，否则训练/推理会在 import 阶段失败。

---

*文档对应环境示例：Ubuntu 22.04，Conda 环境名 `sparse`，Python 3.8，`torch==1.13.0+cu116`。*
