"""SGLang 0.5.22 ComfyUI 单文件加载补丁：剪枝 AdaLN（adaln_basis/adaln_mean）。

原版只按 adaln_t_table 设曲线维度，不设 adaln_affine_input_dim，模型建出来没有 adaln_basis，
加载 10Eros 报 "Parameter adaln_basis not found"（主分支 c7994e7 同样）。
这里按 adaln_basis 形状 [rank, 原始宽度] 补上，与 diffusers 剪枝配置（update_model_arch）一致；
LoRA 的 AdaLN 投影（prepare_lora_adapter）也依赖它。可重复执行。
"""
import pathlib
import sys

import sglang.multimodal_gen.runtime.loader.comfyui_checkpoints.spec as spec_mod

p = pathlib.Path(spec_mod.__file__)
src = p.read_text()
MARK = "# r19 patch: pruned AdaLN basis"
if MARK in src:
    print("R19_ADALN_PATCH already")
    sys.exit(0)
anchor = """                logger.info(
                    "MiniMax-H3 ComfyUI checkpoint uses AdaLN curve %s",
                    adaln_curve_shape,
                )
"""
assert src.count(anchor) == 1, "anchor not found"
patch = anchor + f"""                {MARK}
                from safetensors import safe_open as _so

                with _so(model_path, framework="pt", device="cpu") as _ck:
                    if "adaln_basis" in _ck.keys():
                        _rank, _width = _ck.get_slice("adaln_basis").get_shape()
                        assert _rank == adaln_curve_shape[1], (_rank, adaln_curve_shape)
                        dit_config.arch_config.adaln_affine_input_dim = int(_width)
                        logger.info("MiniMax-H3 pruned AdaLN basis [%d, %d]", _rank, _width)
"""
p.write_text(src.replace(anchor, patch))
print("R19_ADALN_PATCH ok", p)
