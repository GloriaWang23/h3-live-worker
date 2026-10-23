#!/bin/bash
# 2×H100 SXM（sm_90）开机前 CPU 检查（在 bench-sgl-20261023 镜像里跑）：
#   ① 各 CUDA 扩展 .so 里有没有 sm_90 机器码或 ≤90 的 PTX（没有就会在 H100 上报 no kernel image）
#   ② SGLang JIT 内核能否为 sm_90 编译  ③ ffmpeg 有 libx264（H100 没有 NVENC，推流改 CPU 编码）
set -uo pipefail
PY=/opt/venv-comfy/bin/python; SP=$($PY -c "import site; print(site.getsitepackages()[0])")
apt-get update -qq >/dev/null && apt-get install -y -qq --no-install-recommends cuda-cuobjdump-13-0 >/dev/null && echo "cuobjdump ok"
echo "== ① 扩展的 GPU 架构（只列含 GPU 代码的 .so）"
for so in $(find $SP /opt/ComfyUI/custom_nodes -name "*.so" -size +200k 2>/dev/null | grep -viE "torch/lib|nvidia/|triton/_C|cv2|scipy|numpy|PIL|av\.libs|tokenizers|pydantic|grpc|sklearn|pyarrow|safetensors|xformers.libs"); do
  arch=$(/usr/local/cuda-13.0/bin/cuobjdump --list-elf --list-ptx "$so" 2>/dev/null | grep -oE "sm_[0-9]+[af]?|compute_[0-9]+[af]?" | sort -u | paste -sd, -)
  [ -z "$arch" ] && continue
  ok=no; echo "$arch" | grep -qE "sm_90([^0-9]|$)|sm_90a" && ok=sm_90
  [ $ok = no ] && echo "$arch" | grep -oE "compute_[0-9]+" | sed 's/compute_//' | awk '$1<=90{f=1} END{exit !f}' && ok=ptx
  echo "$ok  ${so#$SP/}  [$arch]"
done | sort | tee /tmp/arch.txt | head -80
echo "缺 sm_90 的扩展：$(grep -c '^no ' /tmp/arch.txt)"
echo "== ② SGLang JIT 编译 sm_90"
cd /tmp && $PY - <<'PYEOF'
import glob, os, torch
torch.cuda.current_device = lambda: 0
torch.cuda.get_device_capability = lambda *a, **k: (9, 0)
os.environ["SGLANG_JIT_CACHE_DIR"] = "/tmp/jit90"
from sglang.kernels.ops.layernorm.norm import _jit_qknorm_module
try:
    _jit_qknorm_module(128, torch.bfloat16); print("qknorm sm_90 编译并加载成功")
except Exception as e:
    print("加载异常（CPU 上预期）:", type(e).__name__, str(e)[:200])
libs = glob.glob("/tmp/jit90/**/*.so", recursive=True)
print("产物:", libs[:2]); print("H100_JIT_CHECK", "ok" if libs and any("sm90" in p for p in libs) else "FAIL")
PYEOF
echo "== ③ ffmpeg 编码器"; ffmpeg -hide_banner -encoders 2>/dev/null | grep -E "libx264|h264_nvenc"
echo "== sol-attn 后端支持的架构（源码关键字）"; grep -rnE "sm_?90|capability|major *[<>=]" $SP/sol_attn 2>/dev/null | grep -v test | head -15
