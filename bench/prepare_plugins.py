"""CPU image-build preparation. No GPU detection, model inference, or source kernel builds."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile


def run(*args):
    subprocess.run(args, check=True, timeout=180)


def prepare(root, python, lock_path):
    lock = json.loads(lock_path.read_text())
    nodes = root / "custom_nodes"
    nodes.mkdir(parents=True, exist_ok=True)
    for key, name in (("multistream", "ComfyUI-H3-MultiStream"), ("veda", "Veda-on-ComfyUI")):
        spec = lock[key]
        dest = nodes / name
        if not dest.exists():
            run("git", "clone", "--filter=blob:none", "--no-checkout", "https://github.com/" + spec["repo"], str(dest))
        else:
            run("git", "-C", str(dest), "fetch", "origin", spec["commit"])
        run("git", "-C", str(dest), "checkout", "--detach", spec["commit"])
        actual = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
        if actual != spec["commit"]:
            raise RuntimeError(f"Wrong {key} revision: {actual}")
    # VEDA uses the Triton already pinned by the base image. Do not upgrade torch / Triton.
    run(str(python), "-c", "import triton; assert int(triton.__version__.split('.')[0]) >= 3")
    spec = lock["spark"]
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "spark.zip"
        with urllib.request.urlopen(spec["url"], timeout=60) as response:
            archive.write_bytes(response.read())
        if hashlib.sha256(archive.read_bytes()).hexdigest() != spec["sha256"]:
            raise RuntimeError("Spark release checksum mismatch")
        with zipfile.ZipFile(archive) as zipped:
            for entry in zipped.infolist():
                target = (Path(tmp) / entry.filename).resolve()
                if not target.is_relative_to(Path(tmp).resolve()):
                    raise RuntimeError("Unsafe archive path")
            zipped.extractall(tmp)
        source = Path(tmp) / "ComfyUI-Spark-H3"
        dest = nodes / "ComfyUI-Spark-H3"
        if dest.exists():
            raise RuntimeError(f"Refusing to overwrite an existing plugin: {dest}")
        shutil.copytree(source, dest)
    # The release already contains the SM120 / CUDA 13 / ComfyUI 0.39 wheel.
    # Install on CPU during image build, rather than invoking a GPU-detecting installer on paid time.
    wheel = dest / "wheelhouse" / spec["wheel"]
    run("uv", "pip", "install", "--python", str(python), "--no-deps", "--reinstall", str(wheel))
    run(str(python), "-c", "from importlib.metadata import version; assert version('comfy-kitchen') == '0.2.37+spark.h3.sm120.cu130.1'")
    (lock_path.parent / "plugins-ready.json").write_text(json.dumps(lock, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/opt/ComfyUI"))
    parser.add_argument("--python", type=Path, default=Path("/opt/venv-comfy/bin/python"))
    parser.add_argument("--lock", type=Path, default=Path(__file__).with_name("plugins.lock.json"))
    args = parser.parse_args()
    prepare(args.root, args.python, args.lock)
