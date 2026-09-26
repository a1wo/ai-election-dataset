# AI election dataset

Israeli political / advertising videos labelled **AI** (contains AI-generated
footage) or **not-AI** (old TV commercials and Knesset election broadcasts,
2005–2022), for benchmarking AI-video detectors.

```
sample/                      small set with the video files themselves
  AI/          10 × mp4 + json + jpg    hand-picked political AI clips
  not-AI/      10 × mp4 + json + jpg    5 old commercials + 5 old election broadcasts
  index.json                            one row per video, label + local path
full/                        everything, metadata + thumbnails + public links only (no video files)
  AI/                        14 hand-picked clips (X / zebeai.info) – curated.json lists them with reviewer notes
  not-AI/
    Old Commercials/<year>/          638 Israeli TV commercials, 2005–2022
    Old Elections videos/<year>/     345 Knesset election broadcasts, 2006–2022
  index.json
scripts/eval.py              evaluation harness
tests/test_eval.py           runs the harness against a fake detector
```

Every `<id>.json` holds the public metadata (title, channel, upload date,
duration, description, thumbnail, `url`) plus `label` (`ai` / `not-ai`). The
AI rows also carry `note` (which scenes are AI-made), and for zebeai rows the
tracker's `party`, `platform`, `ocr_text`, `verification`, `engagement` and a
direct `media_url`.

## Evaluating a detector

The detector is an HTTP endpoint or a Python function that answers whether a
video is AI-generated. `eval.py` feeds it one item at a time:

| mode     | what is sent                                     | items |
|----------|--------------------------------------------------|-------|
| `sample` | the video file (multipart field `file` / bytes)  | 20    |
| `full`   | the public link (JSON `{"url": ...}` / `str`)    | 997   |

Any of these answers is understood: `{"ai": true}`, `{"is_ai": 0.93}`,
`{"label": "ai"}`, `true`, `0.93`, `"ai"` (numbers are compared to
`--threshold`, default 0.5).

```bash
python3 scripts/eval.py --mode sample --api http://localhost:8000/classify --out report.json
python3 scripts/eval.py --mode full   --fn my_detector:predict --limit 100
python3 scripts/eval.py --mode sample --api ... --min-accuracy 0.9   # exit 1 when below
```

The summary reports tp/fp/tn/fn, accuracy, precision, recall and F1; `--out`
writes the per-video verdicts. Detector errors count as wrong answers.

```bash
python3 -m unittest discover tests      # self-test with a fake API
```

## Shared-Drive copy

`python3 scripts/make_drive.py` builds `drive/`: the 20 sample videos with
readable file names plus `sample.xlsx`, and for `full/` only spreadsheets
(`AI`, `Old Commercials`, `Old Elections videos` — title, year, channel, party,
link, note). The spreadsheets are committed; the sample videos are not
(`drive/sample/` is gitignored, they are the same files as `sample/`).
