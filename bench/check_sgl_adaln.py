"""CPU 检查（CI）：SGLang ComfyUI 单文件加载 10Eros（剪枝 AdaLN + int8 convrot）时，模型参数与权重键是否一一对上。

用真实文件头 hdr_10eros.json 拼一个稀疏 safetensors（数据区全 0，不占盘），走真实入口 load_comfyui_transformer；
只把 maybe_load_fsdp_model 换成"在 meta 设备上建模型 + 对键"，不读数据。打印 R19_ADALN_CHECK ok/FAIL。
用法：python check_sgl_adaln.py hdr_10eros.json
"""
import json
import os
import struct
import sys
import tempfile
import types

import torch

# CI 无显卡：vsa 等在导入时查算力，假装是 Blackwell（sm_120），不影响 meta 建模
torch.cuda.get_device_capability = lambda *a, **k: (12, 0)

hdr = json.load(open(sys.argv[1]))
end = max(v["data_offsets"][1] for k, v in hdr.items() if k != "__metadata__")
raw = json.dumps(hdr, separators=(",", ":")).encode()
raw += b" " * (-len(raw) % 8)
path = os.path.join(tempfile.mkdtemp(), "10Eros_Max_h3_TURBO-hybrid_beta5_int8.safetensors")
with open(path, "wb") as f:
    f.write(struct.pack("<Q", len(raw)) + raw)
    f.truncate(8 + len(raw) + end)

import sglang.multimodal_gen.runtime.loader.comfyui_checkpoints.spec as spec_mod
import sglang.multimodal_gen.runtime.loader.transformer_load_utils as tlu
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping

result = {}


def fake_load(model_cls, init_params, checkpoint_key_filter=None, **kw):
    arch = init_params["config"].arch_config
    print("arch: adaln_curve_grid", arch.adaln_curve_grid, "time_embed_dim", arch.time_embed_dim,
          "adaln_affine_input_dim", arch.adaln_affine_input_dim)
    with torch.device("meta"):
        model = model_cls(**init_params)
    names = set(model.state_dict().keys())
    mapper = get_param_names_mapping(model_cls.param_names_mapping)
    keys = [k for k in hdr if k != "__metadata__" and (checkpoint_key_filter is None or checkpoint_key_filter(k))]
    mapped = {mapper(k)[0] for k in keys}
    result["ckpt_not_in_model"] = sorted(mapped - names)
    result["model_not_in_ckpt"] = sorted(n for n in names - mapped if not n.startswith("rope."))
    return model


from sglang.multimodal_gen.runtime.distributed import parallel_state as ps

# 单进程 gloo 并行组（tp=sp=1），建模型时要查 tp/sp 大小
ps.init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo",
                                distributed_init_method="tcp://127.0.0.1:29555", device_id=torch.device("cpu"))
ps.initialize_model_parallel(sequence_parallel_degree=1, backend="gloo")
spec_mod.maybe_load_fsdp_model = fake_load
spec_mod.resolve_precision = lambda *a, **k: torch.bfloat16
tlu.resolve_transformer_gguf_to_load = lambda *a, **k: None

server_args = types.SimpleNamespace(
    pipeline_config=types.SimpleNamespace(dit_config=None), model_paths={}, quantization=None, nunchaku_config=None,
    model_variant=None, hsdp_replicate_dim=1, hsdp_shard_dim=1, pin_cpu_memory=False,
    should_use_fsdp_for_component=lambda c: False, should_start_component_on_cpu=lambda c: True,
)
pipeline = types.SimpleNamespace(pipeline_name="MiniMaxH3Pipeline", model_path=path, get_module=lambda n: None)
spec_mod.load_comfyui_transformer(pipeline, server_args)
a, b = result["ckpt_not_in_model"], result["model_not_in_ckpt"]
print("权重有、模型无：", len(a), a[:10])
print("模型有、权重无：", len(b), b[:10])
print("R19_ADALN_CHECK", "ok" if not a and not b else "FAIL")
