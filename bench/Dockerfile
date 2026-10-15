# h3-live 基准测试镜像（只给测速 pod 用，不给 serverless）：一个镜像里两套引擎
#   SGLang Diffusion 0.5.21：官方镜像原样（CUDA 13 开发版、FlashInfer 预编译内核，B200 sm_100 / PRO 6000 sm_120 都覆盖）
#   ComfyUI：从 serverless 镜像整套拷入（/opt/ComfyUI + /opt/venv-comfy + /opt/venv-ctl，与直播用的环境逐字节一致）
# 模型不进镜像：开机后从 HF 下到容器盘。SSH 由 start.sh 用 RunPod 注入的 PUBLIC_KEY 起。
FROM ghcr.io/gloriawang23/h3-live-worker:sha-6a781e2 AS comfy

FROM lmsysorg/sglang:v0.5.21-cu130
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 NVIDIA_DRIVER_CAPABILITIES=all
# ComfyUI 那套 venv 指向系统 Python 3.12（serverless 镜像是 Ubuntu 24.04 自带的 3.12）；ffmpeg 给分片编码，ssh/rsync 给测速脚本
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv ffmpeg openssh-server rsync tmux procps libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* && mkdir -p /run/sshd
COPY --from=comfy /opt/ComfyUI /opt/ComfyUI
COPY --from=comfy /opt/venv-comfy /opt/venv-comfy
COPY --from=comfy /opt/venv-ctl /opt/venv-ctl
# SGLang 读 ComfyUI 的 int8 convrot 权重（10Eros、Comfy-Org int8）要 comfy-kitchen；--no-deps 不动官方镜像的 torch
RUN SGPY=$(command -v python3 | xargs readlink -f); ls /opt/sglang/bin/python && SGPY=/opt/sglang/bin/python; \
    $SGPY -m pip install --no-cache-dir --no-deps comfy-kitchen==0.2.37 \
    && $SGPY -c "import torch, sglang, comfy_kitchen; print('sglang env', torch.__version__, torch.version.cuda)" \
    && /opt/venv-comfy/bin/python -c "import torch, comfy_kitchen; print('comfy env', torch.__version__, torch.version.cuda)" \
    && cd /opt/ComfyUI && timeout 600 /opt/venv-comfy/bin/python main.py --quick-test-for-ci --cpu
RUN mkdir -p /workspace/logs /workspace/models /workspace/h3-live
COPY bench/start.sh /opt/bench/start.sh
ENTRYPOINT []
CMD ["bash", "/opt/bench/start.sh"]
