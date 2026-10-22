#!/bin/bash
# 第十八轮节点（开机后、起 ComfyUI 之前执行；GitHub CPU 探针 bench-probe-r18.yml 跑的是同一个文件）：
#   LightVAE（ModelTC，自带 "Load H3 LightVAE" 节点，复用原生 H3 VAE）、ComfyUI-PTD（vita-epfl，release/comfyui-inference，pip -e）、
#   淘宝 PTD Ref2VA 4/8 步节点包（HF taobao-content-ai/MiniMax-H3-PTD-LoRAs 的 comfyui_node）。提交全部固定。
set -euo pipefail
C=/opt/ComfyUI; PY=/opt/venv-comfy/bin/python; T=/tmp/r18src; mkdir -p $T
LVAE=38b0587d47df9188e3e05973265352962b20ada4; PTD=43ae22c3c94f8a24cc5e82ca82e4e3a5f736de85; PACK=9d2c9ad8289762008cad3880ef0d9f9c2454326b
git clone -q https://github.com/ModelTC/LightVAE.git $T/LightVAE && git -C $T/LightVAE checkout -q $LVAE
(cd $C && $PY $T/LightVAE/comfyui/install.py --comfy-root .)
rm -rf $C/custom_nodes/ComfyUI-PTD && git clone -q https://github.com/vita-epfl/PTD-inference.git $C/custom_nodes/ComfyUI-PTD && git -C $C/custom_nodes/ComfyUI-PTD checkout -q $PTD
# --no-deps: keep the image's torch / ComfyUI stack; import check below catches anything missing
$PY -m pip install -q --no-deps -e $C/custom_nodes/ComfyUI-PTD
P=$C/custom_nodes/ComfyUI-MiniMaxH3-PTD; rm -rf $P; mkdir -p $P
for f in __init__.py delta.py heads.py loader.py mapping.py nodes.py patcher.py sampler.py schedule.py NOTICE; do
  curl -fsSL "https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-LoRAs/resolve/$PACK/comfyui_node/ComfyUI-MiniMaxH3-PTD/$f" -o $P/$f
done
$PY -c "import ptd, ptd_comfy; print('ptd import ok')"
echo R18_NODES_OK
