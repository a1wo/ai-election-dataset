"""Runs scripts/eval.py against a fake detector API in both modes.

    python3 -m unittest discover tests
"""

import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eval as harness  # noqa: E402

AI_KEYS = {
    key for split in ("sample", "full") for r in harness.load_items(split)
    if r["label"] == "ai" for key in (r["id"], r["url"])
}


class FakeDetector(BaseHTTPRequestHandler):
    """Cheats by looking at the id in the filename / url, so it is always right."""

    seen: list[dict] = []

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        ctype = self.headers["Content-Type"]
        if ctype.startswith("multipart/form-data"):
            head = body.split(b"\r\n\r\n", 1)[0].decode()
            name = head.split('filename="')[1].split('"')[0]
            key, kind = name.rsplit(".", 1)[0], "file"
        else:
            key, kind = json.loads(body)["url"], "url"
        ai = key in AI_KEYS
        FakeDetector.seen.append({"kind": kind, "bytes": len(body)})
        out = json.dumps({"is_ai": 0.9 if ai else 0.1}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *_):
        pass


class EvalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), FakeDetector)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}/classify"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        FakeDetector.seen.clear()

    def test_sample_sends_video_files(self):
        report = harness.run("sample", harness.http_evaluator(self.url, 30), 0.5, None)
        self.assertEqual(report["summary"]["items"], 20)
        self.assertEqual(report["summary"]["accuracy"], 1.0)
        self.assertTrue(all(s["kind"] == "file" and s["bytes"] > 100_000 for s in FakeDetector.seen))

    def test_full_sends_links(self):
        report = harness.run("full", harness.http_evaluator(self.url, 30), 0.5, 40)
        self.assertEqual(report["summary"]["items"], 40)
        self.assertEqual(report["summary"]["error"], 0)
        self.assertEqual(report["summary"]["accuracy"], 1.0)
        self.assertTrue(all(s["kind"] == "url" for s in FakeDetector.seen))

    def test_python_function_and_metrics(self):
        def always_ai(_payload):
            return "ai"
        report = harness.run("sample", always_ai, 0.5, None)
        s = report["summary"]
        self.assertEqual((s["tp"], s["fp"], s["tn"], s["fn"]), (10, 10, 0, 0))
        self.assertEqual(s["recall"], 1.0)
        self.assertEqual(s["accuracy"], 0.5)

    def test_verdict_parsing(self):
        for raw in ({"ai": True}, {"label": "AI"}, 0.7, "0.9", "yes", {"result": {"score": 1}}):
            self.assertTrue(harness.to_bool(raw, 0.5), raw)
        for raw in ({"ai": False}, {"label": "real"}, 0.2, "no"):
            self.assertFalse(harness.to_bool(raw, 0.5), raw)


if __name__ == "__main__":
    unittest.main()
