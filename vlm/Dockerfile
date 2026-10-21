# h3-vlm：VLM Serverless 共用镜像（Omni 打标/评委 + Qwen3.6 VL）。RunPod 官方 worker-vllm 原样 + 音频解码库：
# 官方镜像缺 av（vLLM 用 PyAV 从视频里抽音轨，Qwen3-Omni use_audio_in_video 要用）和 librosa / soundfile。
# 其他一律不改：vLLM 版本、handler、环境变量 → vllm serve 参数的机制、HF 缓存路径（/runpod-volume/huggingface-cache/hub）都沿用官方。
FROM runpod/worker-v1-vllm:v2.28.0
RUN python3 -m pip install --no-cache-dir "av>=14" "librosa>=0.10" "soundfile>=0.12" \
 && python3 -c "import av, librosa, soundfile, vllm; print('av', av.__version__, 'librosa', librosa.__version__, 'vllm', vllm.__version__)"
