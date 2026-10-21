"""h3-vlm handler：官方 worker-vllm 的 /src/handler.py 换成本文件，官方原文件改名 official_handler.py，逻辑一行不改。

在把请求交给官方 handler（→ 本机 vLLM）之前做三件事：
1. 素材自己下载：messages 里 http(s) 的 image_url / video_url / audio_url 先下到本地（120 s × 3 次重试、浏览器 UA、
   大小上限），可选回源（MEDIA_FALLBACK_HOSTS：CDN 失败改走源站），再改写成 file://。data: URL 也落盘。
   vLLM 自带下载器默认 10–30 s 超时且部分机房到源站很慢，冒烟验收时整条请求卡死。
2. 视频音轨（H3_AUDIO_FROM_VIDEO=1，Omni 端点用）：ffmpeg 抽 16 kHz 单声道 wav，作为独立 audio_url 插在视频后面；
   去掉 use_audio_in_video —— vLLM 0.30–0.32 的音视频交错合并有未修 bug（#53700 / #47788），引擎直接崩。
3. 引擎崩溃自愈：vLLM 进程退出或返回 EngineDead 时，回报错误后让本 worker 退出，平台换新 worker，
   不再让后续请求落到坏 worker 上。
输出（非流式）额外带 "h3": {timings, media}，便于工作台显示各阶段耗时。
"""
from __future__ import annotations

import asyncio
import base64
import copy
import logging
import os
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, AsyncGenerator

import official_handler as official

# main.py 在启动后给 handler 模块设置这两个属性；这里同名占位，转交时同步给官方模块
vllm_process = None
startup_error = None

MEDIA_ROOT = os.getenv("H3_MEDIA_DIR", "/tmp/h3media")              # 必须在 vLLM --allowed-local-media-path 之内
MAX_MEDIA_BYTES = int(os.getenv("H3_MAX_MEDIA_MB", "500")) << 20
DOWNLOAD_TIMEOUT = float(os.getenv("H3_DOWNLOAD_TIMEOUT", "120"))
DOWNLOAD_TRIES = int(os.getenv("H3_DOWNLOAD_TRIES", "3"))
AUDIO_FROM_VIDEO = os.getenv("H3_AUDIO_FROM_VIDEO", "0").lower() in ("1", "true", "yes")
# "cdn.example.com=origin.example.com,..."：CDN 下载失败时换源站再试
FALLBACK = dict(p.split("=", 1) for p in os.getenv("MEDIA_FALLBACK_HOSTS", "").split(",") if "=" in p)
MEDIA_KINDS = {"image_url": ".png", "video_url": ".mp4", "audio_url": ".wav"}
DEAD_MARKERS = ("EngineDeadError", "EngineCore encountered", "vLLM server process is not running")

log = logging.getLogger("h3-vlm")


class Permanent(Exception):
    """重试也没用的错误（链接 4xx、文件过大）。"""


def _download(url: str, path: str) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (h3-vlm-worker)"})
    size = 0
    with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as r, open(path, "wb") as f:
        while chunk := r.read(1 << 20):
            size += len(chunk)
            if size > MAX_MEDIA_BYTES:
                raise Permanent(f"素材超过 {MAX_MEDIA_BYTES >> 20} MB：{url[:120]}")
            f.write(chunk)
    return size


def _fetch(url: str, path: str) -> tuple[int, str]:
    """下载一个素材，返回 (字节数, 实际用的地址)。CDN 失败时按 MEDIA_FALLBACK_HOSTS 回源。"""
    host = urllib.parse.urlsplit(url).hostname or ""
    candidates = [url] + ([url.replace(f"//{host}/", f"//{FALLBACK[host]}/", 1)] if host in FALLBACK else [])
    last: Exception | None = None
    for u in candidates:
        for k in range(DOWNLOAD_TRIES):
            try:
                return _download(u, path), u
            except Permanent:
                raise
            except urllib.error.HTTPError as e:
                if 400 <= e.code < 500 and e.code not in (408, 429):
                    last = e
                    break                                    # 这个地址不会好了，换下一个
                last = e
            except Exception as e:                           # 超时、断连、DNS
                last = e
            if k < DOWNLOAD_TRIES - 1:
                time.sleep(2 * (k + 1))
            log.warning("download %s failed (%s), attempt %d/%d", u[:120], last, k + 1, DOWNLOAD_TRIES)
    raise Permanent(f"素材下载失败：{url[:120]}（{last}）")


def _save_data_url(url: str, path: str) -> int:
    data = base64.b64decode(url.split(",", 1)[1])
    if len(data) > MAX_MEDIA_BYTES:
        raise Permanent(f"素材超过 {MAX_MEDIA_BYTES >> 20} MB")
    with open(path, "wb") as f:
        f.write(data)
    return len(data)


def _extract_audio(video: str) -> str | None:
    """抽 16 kHz 单声道 wav；视频没有音轨返回 None。"""
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", video],
                           capture_output=True, text=True, timeout=60)
    if not probe.stdout.strip():
        return None
    out = video.rsplit(".", 1)[0] + ".audio.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", video, "-vn", "-ac", "1", "-ar", "16000", out], check=True, timeout=300)
    return out


def _prepare(body: dict, workdir: str, timings: dict, media: list) -> dict:
    """改写 messages：远程 / data 素材落盘为 file://；Omni 视频后面插独立音轨。"""
    body = copy.deepcopy(body)
    mm = body.get("mm_processor_kwargs")
    if isinstance(mm, dict):
        mm.pop("use_audio_in_video", None)                   # 见文件头第 2 条
        if not mm:
            body.pop("mm_processor_kwargs")
    for msg in body.get("messages") or []:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        new: list = []
        for part in content:
            kind = part.get("type") if isinstance(part, dict) else None
            if kind not in MEDIA_KINDS or not isinstance(part.get(kind), dict):
                new.append(part)
                continue
            url = str(part[kind].get("url") or "")
            path = os.path.join(workdir, f"{len(media):02d}{MEDIA_KINDS[kind]}")
            t = time.time()
            if url.startswith(("http://", "https://")):
                size, used = _fetch(url, path)
                timings["download_s"] = round(timings.get("download_s", 0) + time.time() - t, 2)
                media.append({"kind": kind, "bytes": size, "source": "fallback" if used != url else "url"})
            elif url.startswith("data:"):
                size = _save_data_url(url, path)
                media.append({"kind": kind, "bytes": size, "source": "inline"})
            else:                                            # 已是 file:// 等，原样交给 vLLM
                new.append(part)
                continue
            new.append({**part, kind: {**part[kind], "url": f"file://{path}"}})
            if kind == "video_url" and AUDIO_FROM_VIDEO:
                t = time.time()
                wav = _extract_audio(path)
                timings["audio_s"] = round(timings.get("audio_s", 0) + time.time() - t, 2)
                if wav:
                    new.append({"type": "audio_url", "audio_url": {"url": f"file://{wav}"}})
                    media.append({"kind": "audio_url", "bytes": os.path.getsize(wav), "source": "from_video"})
        msg["content"] = new
    return body


def _engine_dead(err: str) -> bool:
    proc = official.vllm_process
    return any(m in err for m in DEAD_MARKERS) or (proc is not None and proc.poll() is not None)


def _exit_soon(reason: str) -> None:
    """回报完错误再退：杀 vLLM、退出进程，平台换新 worker。"""
    def bye():
        log.error("vLLM engine is dead (%s); exiting so the platform replaces this worker", reason[:200])
        proc = official.vllm_process
        if proc is not None and proc.poll() is None:
            proc.kill()
        os._exit(1)
    threading.Timer(3.0, bye).start()


async def handler(job: dict) -> AsyncGenerator[Any, None]:
    official.vllm_process = vllm_process
    official.startup_error = startup_error
    job_input = job.get("input") or {}
    body = job_input.get("openai_input")
    timings: dict = {}
    media: list = []
    workdir = os.path.join(MEDIA_ROOT, job.get("id") or uuid.uuid4().hex)
    t0 = time.time()
    try:
        if isinstance(body, dict) and body.get("messages"):
            os.makedirs(workdir, exist_ok=True)
            try:
                body = await asyncio.to_thread(_prepare, body, workdir, timings, media)
            except Permanent as e:
                yield {"error": {"message": str(e), "type": "media_error", "code": None}}
                return
            except subprocess.SubprocessError as e:
                yield {"error": {"message": f"抽音轨失败：{e}", "type": "media_error", "code": None}}
                return
            job = {**job, "input": {**job_input, "openai_input": body}}
        timings["prepare_s"] = round(time.time() - t0, 2)
        t1 = time.time()
        async for out in official.handler(job):
            if isinstance(out, dict) and "error" in out:
                msg = str((out.get("error") or {}).get("message", ""))
                if _engine_dead(msg):
                    _exit_soon(msg)
                yield out
                continue
            if isinstance(out, dict):
                timings["infer_s"] = round(time.time() - t1, 2)
                out = {**out, "h3": {"timings": timings, "media": media}}
            yield out
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
