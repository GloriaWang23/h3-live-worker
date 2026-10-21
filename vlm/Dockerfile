# h3-vlm-worker：RunPod 官方 worker-vllm 原样 + 我们的预处理 handler（见 h3_vlm_handler.py 文件头）。
# 官方 main.py / args_builder / 启动与缓存逻辑都不动；只把 /src/handler.py 换成包了一层的版本。
FROM runpod/worker-v1-vllm:v2.28.0
RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m pip install --no-cache-dir "av>=14" "librosa>=0.10" "soundfile>=0.12" \
 && mv /src/handler.py /src/official_handler.py
COPY h3_vlm_handler.py /src/handler.py
# vLLM 只读本地素材目录（handler 落盘到这里）
ENV H3_MEDIA_DIR=/tmp/h3media \
    ALLOWED_LOCAL_MEDIA_PATH=/tmp/h3media
RUN mkdir -p /tmp/h3media \
 && python3 -c "import sys; sys.path.insert(0, '/src'); import handler, official_handler, av, librosa; print('h3-vlm handler OK')" \
 && ffmpeg -version | head -1
