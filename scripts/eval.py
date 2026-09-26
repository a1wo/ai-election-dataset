#!/usr/bin/env python3
"""Evaluate an "is this video AI-generated?" detector against the dataset.

The detector is either an HTTP endpoint (--api URL) or a Python callable
(--fn package.module:function). It receives one item at a time:

  sample mode  -> the video file itself      (multipart field "file" / bytes)
  full mode    -> the video's public link    (JSON {"url": ...} / str)

and must answer with something that can be read as a boolean:
  {"ai": true} | {"is_ai": 0.93} | {"label": "ai"} | true | 0.93 | "ai"

Usage:
  python3 scripts/eval.py --mode sample --api http://localhost:8000/classify
  python3 scripts/eval.py --mode full   --fn my_detector:predict --limit 50
"""

import argparse
import importlib
import json
import mimetypes
import sys
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
TRUE_WORDS = {"ai", "true", "yes", "fake", "generated", "synthetic", "1"}


def load_items(mode: str) -> list[dict]:
    return json.loads((ROOT / mode / "index.json").read_text(encoding="utf-8"))


def to_bool(answer, threshold: float) -> bool:
    if isinstance(answer, dict):
        for key in ("ai", "is_ai", "ai_generated", "prediction", "label", "score", "result"):
            if key in answer:
                return to_bool(answer[key], threshold)
        raise ValueError(f"cannot read verdict from {answer!r}")
    if isinstance(answer, bool):
        return answer
    if isinstance(answer, (int, float)):
        return answer >= threshold
    if isinstance(answer, str):
        text = answer.strip().lower()
        try:
            return float(text) >= threshold
        except ValueError:
            return text in TRUE_WORDS
    raise ValueError(f"cannot read verdict from {answer!r}")


def multipart(field: str, filename: str, data: bytes) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    ctype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; "
        f"filename=\"{filename}\"\r\nContent-Type: {ctype}\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def http_evaluator(url: str, timeout: float) -> Callable:
    def call(payload):
        if isinstance(payload, Path):
            body, ctype = multipart("file", payload.name, payload.read_bytes())
        else:
            body, ctype = json.dumps({"url": payload}).encode(), "application/json"
        req = urllib.request.Request(url, data=body, headers={"Content-Type": ctype, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return call


def fn_evaluator(spec: str) -> Callable:
    module, _, name = spec.partition(":")
    sys.path.insert(0, str(Path.cwd()))
    fn = getattr(importlib.import_module(module), name or "predict")

    def call(payload):
        return fn(payload.read_bytes() if isinstance(payload, Path) else payload)
    return call


def run(mode: str, evaluator: Callable, threshold: float, limit: int | None) -> dict:
    items = load_items(mode)
    if limit:
        items = items[:limit]
    results, tp = [], {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "error": 0}
    for i, item in enumerate(items, 1):
        payload = ROOT / item["path"] if mode == "sample" else item["url"]
        expected = item["label"] == "ai"
        t0 = time.time()
        try:
            raw = evaluator(payload)
            predicted = to_bool(raw, threshold)
            key = {(True, True): "tp", (True, False): "fn", (False, True): "fp", (False, False): "tn"}[(expected, predicted)]
        except Exception as exc:  # detector failures count against it
            raw, predicted, key = str(exc), None, "error"
        tp[key] += 1
        results.append({
            "id": item["id"], "title": item["title"], "expected": item["label"],
            "predicted": None if predicted is None else ("ai" if predicted else "not-ai"),
            "correct": None if predicted is None else predicted == expected,
            "raw": raw, "seconds": round(time.time() - t0, 2),
            "input": str(payload.relative_to(ROOT)) if mode == "sample" else payload,
        })
        print(f"[{i}/{len(items)}] {key.upper():5} {item['label']:6} {item['title'][:60]}", flush=True)
    n = len(items) - tp["error"]
    precision = tp["tp"] / (tp["tp"] + tp["fp"]) if tp["tp"] + tp["fp"] else 0.0
    recall = tp["tp"] / (tp["tp"] + tp["fn"]) if tp["tp"] + tp["fn"] else 0.0
    summary = {
        "mode": mode, "items": len(items), **tp,
        "accuracy": round((tp["tp"] + tp["tn"]) / n, 4) if n else 0.0,
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
    }
    return {"summary": summary, "results": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("sample", "full"), required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--api", help="POST endpoint; gets multipart file (sample) or JSON {url} (full)")
    source.add_argument("--fn", help="module:function; gets bytes (sample) or url str (full)")
    parser.add_argument("--threshold", type=float, default=0.5, help="numeric answers >= this mean AI")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path, help="write full report JSON here")
    parser.add_argument("--min-accuracy", type=float, default=None, help="exit 1 when accuracy is below this")
    args = parser.parse_args()

    evaluator = http_evaluator(args.api, args.timeout) if args.api else fn_evaluator(args.fn)
    report = run(args.mode, evaluator, args.threshold, args.limit)
    print(json.dumps(report["summary"], indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.min_accuracy is not None and report["summary"]["accuracy"] < args.min_accuracy:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
