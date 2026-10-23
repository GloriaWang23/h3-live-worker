"""CPU 检查（CI）：SGLang JIT 内核能否用镜像里的 nvcc + ninja 为 PRO 6000（sm_120）编译出来。

CI 没有显卡：假装算力 12.0，只看编译产物（.so）是否生成；没有驱动时加载 .so 失败属预期。打印 R20_JIT_CHECK ok/FAIL。
"""
import glob
import os

import torch

torch.cuda.current_device = lambda: 0
torch.cuda.get_device_capability = lambda *a, **k: (12, 0)

from sglang.kernels.ops.layernorm.norm import _jit_qknorm_module

cache = os.environ.get("SGLANG_JIT_CACHE_DIR", os.path.expanduser("~/.cache/sglang/jit"))
try:
    _jit_qknorm_module(128, torch.bfloat16)
    print("qknorm 编译并加载成功")
except Exception as e:  # 无驱动时加载失败是预期，编译失败才算问题
    print("qknorm 加载异常（CPU 上预期）:", type(e).__name__, str(e)[:300])
libs = glob.glob(os.path.join(cache, "**", "*.so"), recursive=True)
print("编译产物：", libs[:5])
print("R20_JIT_CHECK", "ok" if libs else "FAIL")
