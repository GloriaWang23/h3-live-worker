"""离线测试 h3_vlm_handler（不需要 GPU / vLLM）：官方 handler 用假模块代替，素材用本机 HTTP 服务提供。
  python3 -m unittest discover -s tests
"""
import asyncio
import base64
import http.server
import importlib
import os
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TMP = Path(tempfile.mkdtemp())
CALLS: list = []


def _fake_official():
    m = types.ModuleType("official_handler")
    m.vllm_process = None
    m.startup_error = None
    m.reply = {"choices": [{"message": {"content": "ok"}}]}

    async def handler(job):
        CALLS.append(job)
        for p in job["input"]["openai_input"]["messages"][0]["content"]:   # 交给 vLLM 时本地文件必须存在
            for k in ("video_url", "audio_url", "image_url"):
                if k in p:
                    assert os.path.exists(p[k]["url"].removeprefix("file://")), p
        yield m.reply
    m.handler = handler
    return m


def _make_video(path: Path, audio: bool) -> None:
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=8"]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100", "-c:a", "aac", "-shortest"]
    subprocess.run(cmd + ["-t", "2", "-c:v", "libx264", str(path)], check=True)


class Handler(http.server.SimpleHTTPRequestHandler):
    hits: dict = {}

    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(TMP), **k)

    def do_GET(self):
        Handler.hits[self.path] = Handler.hits.get(self.path, 0) + 1
        if self.path.startswith("/flaky/") and Handler.hits[self.path] < 2:   # 第一次 500，第二次成功
            self.send_error(500)
            return
        if self.path.startswith("/flaky/"):
            self.path = self.path.removeprefix("/flaky")
        super().do_GET()

    def log_message(self, *a):
        pass


class TestHandler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _make_video(TMP / "av.mp4", audio=True)
        _make_video(TMP / "silent.mp4", audio=False)
        cls.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        sys.modules["official_handler"] = _fake_official()
        sys.path.insert(0, str(ROOT))
        os.environ.update(H3_MEDIA_DIR=str(TMP / "media"), H3_AUDIO_FROM_VIDEO="1", H3_DOWNLOAD_TRIES="2",
                          MEDIA_FALLBACK_HOSTS=f"cdn.invalid=127.0.0.1:{cls.srv.server_address[1]}")
        import h3_vlm_handler
        cls.h = importlib.reload(h3_vlm_handler)

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def run_job(self, content, extra=None):
        body = {"model": "omni", "messages": [{"role": "user", "content": content}], **(extra or {})}
        async def go():
            return [o async for o in self.h.handler({"id": "t1", "input": {"openai_route": "/v1/chat/completions", "openai_input": body}})]
        return asyncio.run(go())

    def test_video_url_downloaded_and_audio_added(self):
        out = self.run_job([{"type": "video_url", "video_url": {"url": f"{self.base}/av.mp4"}}, {"type": "text", "text": "hi"}],
                           {"mm_processor_kwargs": {"use_audio_in_video": True}})
        sent = CALLS[-1]["input"]["openai_input"]
        kinds = [p["type"] for p in sent["messages"][0]["content"]]
        self.assertEqual(kinds, ["video_url", "audio_url", "text"])
        self.assertNotIn("mm_processor_kwargs", sent)                       # use_audio_in_video 被去掉
        self.assertEqual(out[0]["choices"][0]["message"]["content"], "ok")
        self.assertIn("download_s", out[0]["h3"]["timings"])
        self.assertEqual([m["source"] for m in out[0]["h3"]["media"]], ["url", "from_video"])
        self.assertFalse(os.path.exists(TMP / "media" / "t1"))               # 用完清理

    def test_silent_video_gets_no_audio(self):
        self.run_job([{"type": "video_url", "video_url": {"url": f"{self.base}/silent.mp4"}}])
        self.assertEqual([p["type"] for p in CALLS[-1]["input"]["openai_input"]["messages"][0]["content"]], ["video_url"])

    def test_data_url_and_retry_and_fallback(self):
        b64 = base64.b64encode((TMP / "av.mp4").read_bytes()).decode()
        out = self.run_job([{"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}},
                            {"type": "image_url", "image_url": {"url": f"{self.base}/flaky/silent.mp4"}},
                            {"type": "video_url", "video_url": {"url": "http://cdn.invalid/silent.mp4"}}])
        self.assertEqual([m["source"] for m in out[0]["h3"]["media"]], ["inline", "from_video", "url", "fallback"])

    def test_bad_link_is_reported_not_forwarded(self):
        n = len(CALLS)
        out = self.run_job([{"type": "video_url", "video_url": {"url": f"{self.base}/missing.mp4"}}])
        self.assertEqual(out[0]["error"]["type"], "media_error")
        self.assertEqual(len(CALLS), n)

    def test_engine_dead_schedules_exit(self):
        fired = []
        self.h._exit_soon = lambda reason: fired.append(reason)
        sys.modules["official_handler"].reply = {"error": {"message": "vLLM returned HTTP 500: EngineCore encountered an issue"}}
        try:
            out = self.run_job([{"type": "text", "text": "hi"}])
        finally:
            sys.modules["official_handler"].reply = {"choices": [{"message": {"content": "ok"}}]}
        self.assertIn("error", out[0])
        self.assertEqual(len(fired), 1)


if __name__ == "__main__":
    unittest.main()
