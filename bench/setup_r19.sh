#!/bin/bash
# 第十九轮（开机后、起 ComfyUI 之前；GitHub CPU 探针 bench-probe-r19.yml 跑的是同一个文件）：
#   往 /opt/venv-comfy 装 sglang[diffusion]==0.5.22（PyPI wheel，含 ComfyUI_SGLDiffusion 插件与 H3 原生 DiT、IPC all-to-all）
#   + sol-attn（NVlabs/Sana sol-engine 固定提交，SGLang 的 sol_attn 后端要用）。
#   约束：镜像里已有的包一律保持原版本（--overrides 为当前 freeze），只新增缺的包；冲突在这里就报错，不带到 GPU 上。
set -euo pipefail
C=/opt/ComfyUI; PY=/opt/venv-comfy/bin/python; T=/tmp/r19src; mkdir -p $T
SGL=0.5.22; SOL=9f0af774e7be7a82cb429de7428e678e8157c750
t0=$(date +%s)
$PY -m pip freeze --exclude-editable | grep -v "^-e \|@ file:" > $T/keep.txt
echo "保持原版本的包：$(wc -l < $T/keep.txt) 个"
VIRTUAL_ENV=/opt/venv-comfy uv pip install --python $PY --overrides $T/keep.txt \
  --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match \
  "sglang[diffusion]==$SGL" "sol-attn @ git+https://github.com/NVlabs/Sana.git@$SOL#subdirectory=techniques/sparse_backends" 2>&1 | tail -25
echo "安装用时 $(( $(date +%s)-t0 )) s"
$PY -m pip freeze --exclude-editable | grep -v "^-e \|@ file:" > $T/after.txt
$PY - $T/keep.txt $T/after.txt <<'PY'
import re, sys
def read(path):
    return {re.sub(r"[-_.]+", "-", l.split("==")[0].lower()): l.split("==", 1)[1].strip() for l in open(path) if "==" in l}
keep, after = read(sys.argv[1]), read(sys.argv[2])
changed = {k: (keep[k], after.get(k)) for k in keep if after.get(k) != keep[k]}
print("新增包：", len(set(after) - set(keep)), "个")
if changed:
    sys.exit(f"!!! 镜像原有的包版本被改了：{changed}")
PY
# 插件随 wheel 发布：软链进 custom_nodes（目录名照 README）
SP=$($PY -c "import sglang.multimodal_gen, os; print(os.path.dirname(sglang.multimodal_gen.__file__))")
ln -sfn $SP/apps/ComfyUI_SGLDiffusion $C/custom_nodes/ComfyUI_SGLDiffusion
$PY -c "import torch, sglang; from sglang.multimodal_gen import DiffGenerator; import sol_attn; print('sglang', sglang.__version__, 'torch', torch.__version__, 'import ok')"
echo R19_SGL_OK
