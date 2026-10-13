# h3-live serverless worker：一个任务 = 一场直播。
# 环境与直播用的 GPU 机器（comfy.tar，2026-10-13 导出清单）逐项一致：Ubuntu 24.04 系统 Python 3.12.3、
# torch 2.14.1+cu130、ComfyUI v0.39.0（b0b7435）、10 个插件按提交号固定、全部 Python 包按清单固定（--no-deps）。
# 模型不进镜像：端点的缓存模型（HF GloriaWang23/h3-live-runtime）挂在 /runpod-volume/huggingface-cache。
# 直播代码也不进镜像：每个任务开始时从云端控制台取最新代码包（改代码不用重建镜像）。
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all \
    UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never

# 系统依赖：Python 3.12（与 GPU 机器同版本）、ffmpeg（分片编码 / VHS）、编译器（triton 运行时 JIT 需要）、OpenCV 运行库
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3.12-dev build-essential git curl ca-certificates ffmpeg \
      libgl1 libglib2.0-0 procps \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv

# ComfyUI + 插件（按提交号）
COPY custom_nodes.txt /tmp/custom_nodes.txt
RUN git clone -q https://github.com/Comfy-Org/ComfyUI.git /opt/ComfyUI \
    && git -C /opt/ComfyUI checkout -q b0b743566f65daafc423b4fea8a2fbda94b3384a \
    && grep -v '^#' /tmp/custom_nodes.txt | while read -r name url commit; do \
         git clone -q "$url" "/opt/ComfyUI/custom_nodes/$name" && git -C "/opt/ComfyUI/custom_nodes/$name" checkout -q "$commit" || exit 1; \
       done

# ComfyUI 环境：全部包按 GPU 机器上的清单原样安装（不解析依赖，保证与测过的环境逐字节同版本）
COPY requirements-comfy.txt /tmp/requirements-comfy.txt
RUN uv venv -q --python /usr/bin/python3.12 /opt/venv-comfy \
    && VIRTUAL_ENV=/opt/venv-comfy uv pip install -q --no-deps -r /tmp/requirements-comfy.txt \
         --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu130 \
         --index-strategy unsafe-best-match \
    && rm -rf /root/.cache

# 控制环境：serverless 入口 + 直播控制服务
COPY requirements-ctl.txt /tmp/requirements-ctl.txt
RUN uv venv -q --python /usr/bin/python3.12 /opt/venv-ctl \
    && VIRTUAL_ENV=/opt/venv-ctl uv pip install -q -r /tmp/requirements-ctl.txt && rm -rf /root/.cache

# 构建时自检：CPU 模式起一次 ComfyUI，导入或依赖坏了让构建失败，而不是让线上 worker 失败
RUN cd /opt/ComfyUI && timeout 600 /opt/venv-comfy/bin/python main.py --quick-test-for-ci --cpu \
    && /opt/venv-comfy/bin/python -c "import torch, comfy_kitchen; print('torch', torch.__version__, torch.version.cuda)"

RUN mkdir -p /workspace/logs /workspace/models /workspace/h3-live
COPY handler.py /opt/h3live/handler.py
CMD ["/opt/venv-ctl/bin/python", "-u", "/opt/h3live/handler.py"]
