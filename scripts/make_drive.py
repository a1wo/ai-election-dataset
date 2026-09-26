#!/usr/bin/env python3
"""Build drive/ — a human-friendly copy of the dataset for a shared Google Drive.

  drive/
    sample/AI/<nn> - <title>.mp4          the 10 AI sample videos
    sample/not-AI/<nn> - <title>.mp4      the 10 not-AI sample videos
    sample/sample.xlsx|csv                one row per sample video
    full/AI.xlsx|csv                      14 curated AI clips: title, publisher, party, link, note
    full/Old Commercials.xlsx|csv         638 rows: year, title, channel, link
    full/Old Elections videos.xlsx|csv    345 rows: year, title, channel, link
    README.txt

Requires openpyxl for the xlsx files (csv is always written).
"""

import csv
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVE = ROOT / "drive"

try:
    from openpyxl import Workbook
    from openpyxl.utils import get_column_letter
except ImportError:  # pragma: no cover
    Workbook = None


def safe_name(text: str, limit: int = 60) -> str:
    text = re.sub(r'[\\/:*?"<>|\r\n\t]+', " ", text or "").strip(" .")
    return re.sub(r"\s+", " ", text)[:limit].strip() or "untitled"


def date(s: str | None) -> str:
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}" if s and len(s) == 8 else ""


def title(r: dict) -> str:
    return r.get("title") or r.get("altText") or f"{r.get('channel') or r['id']} - {r.get('note') or ''}".strip(" -")


def row(r: dict, extra: dict | None = None) -> dict:
    base = {
        "label": "AI" if r["label"] == "ai" else "not-AI",
        "year": r["year"],
        "title": title(r),
        "channel / publisher": r.get("channel") or "",
        "party": r.get("party") or "",
        "published": date(r.get("upload_date")),
        "duration (s)": r.get("duration") or "",
        "link": r["url"],
        "note": r.get("note") or "",
        "id": r["id"],
    }
    return {**base, **(extra or {})}


def write_table(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0])
    with (path.with_suffix(".csv")).open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    if Workbook is None:
        return
    wb = Workbook()
    ws = wb.active
    ws.title = path.stem[:31]
    ws.append(cols)
    for r in rows:
        ws.append([r[c] for c in cols])
        cell = ws.cell(row=ws.max_row, column=cols.index("link") + 1)
        cell.hyperlink = r["link"]
        cell.style = "Hyperlink"
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    widths = {"title": 60, "note": 45, "link": 45, "channel / publisher": 25, "file": 50}
    for i, c in enumerate(cols, 1):
        ws.column_dimensions[get_column_letter(i)].width = widths.get(c, 12)
    ws.sheet_view.rightToLeft = True
    wb.save(path.with_suffix(".xlsx"))


def main() -> None:
    if DRIVE.exists():
        shutil.rmtree(DRIVE)

    sample_rows = []
    counters = {"AI": 0, "not-AI": 0}
    for r in json.loads((ROOT / "sample" / "index.json").read_text(encoding="utf-8")):
        label = "AI" if r["label"] == "ai" else "not-AI"
        counters[label] += 1
        name = f"{counters[label]:02d} - {safe_name(title(r))}.mp4"
        dest = DRIVE / "sample" / label / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / r["path"], dest)
        sample_rows.append(row(r, {"file": f"sample/{label}/{name}"}))
    write_table(DRIVE / "sample" / "sample", sample_rows)

    full = json.loads((ROOT / "full" / "index.json").read_text(encoding="utf-8"))
    (DRIVE / "full").mkdir(parents=True)
    groups = {
        "AI": [r for r in full if r["label"] == "ai"],
        "Old Commercials": [r for r in full if r["category"] == "israeli-commercials"],
        "Old Elections videos": [r for r in full if r["category"] == "election-ads"],
    }
    for name, rows in groups.items():
        rows.sort(key=lambda r: (r["year"], r["title"]))
        write_table(DRIVE / "full" / name, [row(r) for r in rows])

    (DRIVE / "README.txt").write_text(
        "AI election dataset\n\n"
        "sample/   20 short videos with the files themselves: AI/ (10 clips with AI-made scenes)\n"
        "          and not-AI/ (10 old Israeli commercials + election broadcasts). sample.xlsx lists them.\n"
        "full/     spreadsheets only (title, year, channel, party, link, note):\n"
        f"          AI ({len(groups['AI'])}), Old Commercials ({len(groups['Old Commercials'])}), "
        f"Old Elections videos ({len(groups['Old Elections videos'])}).\n\n"
        "Source + evaluation harness: https://github.com/a1wo/ai-election-dataset\n",
        encoding="utf-8",
    )
    print(f"drive/: {sum(1 for _ in DRIVE.rglob('*') if _.is_file())} files")


if __name__ == "__main__":
    main()
