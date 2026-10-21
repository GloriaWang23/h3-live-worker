# h3-vlm：VLM 评委 / 视频打标共用服务。RunPod 官方 worker-vllm 原样 + 音频解码库（官方镜像缺 librosa / soundfile，Qwen3-Omni 听音轨要用）。
# 其他一律不改：vLLM 版本、handler、环境变量 → vllm serve 参数的机制都沿用官方。
FROM runpod/worker-v1-vllm:v2.28.0
RUN python3 -m pip install --no-cache-dir "librosa>=0.10" "soundfile>=0.12" \
 && python3 -c "import librosa, soundfile, vllm; print('librosa', librosa.__version__, 'soundfile', soundfile.__version__, 'vllm', vllm.__version__)"
