"""h3-live serverless 入口：一个任务 = 一场直播。
任务输入 {control_url, pod_token, session_id}（云端控制台提交）。流程：
  从控制台取最新代码包 → 解到 /workspace/h3-live → 跑代码包里的 pod/serverless_job.sh（直播服务，直到本场结束）。
直播逻辑都在代码包里，镜像只提供环境；这里尽量保持不变，改直播代码不用重建镜像。
"""
import glob
import io
import os
import shutil
import subprocess
import tarfile
import urllib.request

import runpod

W = "/workspace"
H = f"{W}/h3-live"
CACHE = "/runpod-volume/huggingface-cache/hub/models--GloriaWang23--h3-live-runtime/snapshots"


def model_dir():
    sha = os.environ.get("H3_MODEL_SHA", "")
    snaps = sorted(glob.glob(f"{CACHE}/{sha}*" if sha else f"{CACHE}/*"))
    if not snaps:
        raise RuntimeError(f"没有找到缓存模型：{CACHE}/{sha or '*'}（检查端点的 Model 设置）")
    return snaps[-1]


def handler(job):
    i = job["input"]
    url, tok, sid = i["control_url"].rstrip("/"), i["pod_token"], i["session_id"]
    # 自定义 User-Agent：Cloudflare 默认拦截 Python-urllib（错误码 1010）
    req = urllib.request.Request(f"{url}/api/pod/bundle", headers={"X-Pod-Token": tok, "X-Session": sid, "User-Agent": "h3live-worker/1.0"})
    data = urllib.request.urlopen(req, timeout=120).read()
    shutil.rmtree(H, ignore_errors=True)
    os.makedirs(H)
    with tarfile.open(fileobj=io.BytesIO(data)) as t:
        t.extractall(H, filter="data")
    env = {**os.environ, "CONTROL_URL": url, "POD_TOKEN": tok, "SESSION_ID": sid,
           "H3LIVE_RUNTIME": "serverless", "H3_MODELS": model_dir(), "RUNPOD_JOB_ID": job["id"]}
    rc = subprocess.call(["bash", f"{H}/pod/serverless_job.sh"], env=env)
    return {"session_id": sid, "rc": rc}


runpod.serverless.start({"handler": handler})
