"""
Reproduces the cleaning step used to build data/shl_catalog.json from a raw
CSV of scraped SHL catalog rows with columns:
    name,url,duration,test_type,remote_testing,adaptive_irt

Filtering logic (see APPROACH.md for the reasoning):
  1. De-duplicate by (name, url).
  2. Keep only rows with a SINGLE test_type code — SHL's own taxonomy uses
     one code per construct (K, P, A, S, ...); items with 2+ codes are
     composite batteries, i.e. Pre-packaged Job Solutions, which are
     explicitly out of scope for this assignment.
  3. Drop anything with "solution" in the name as an extra safety net,
     since a few single-type items are still named "<Role> Solution".

Usage:
    python scripts/rebuild_catalog.py path/to/raw.csv data/shl_catalog.json
"""
import csv
import json
import sys
from collections import Counter

TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


def clean(raw_csv_path: str):
    rows = list(csv.DictReader(open(raw_csv_path, encoding="utf-8")))
    seen = set()
    final = []
    for r in rows:
        name = r["name"].strip()
        url = r["url"].strip()
        key = (name.lower(), url)
        if key in seen:
            continue
        seen.add(key)

        types = [t.strip() for t in r["test_type"].split(",") if t.strip()]
        if len(types) != 1:
            continue
        if "solution" in name.lower():
            continue

        code = types[0]
        final.append(
            {
                "name": name,
                "url": url,
                "test_type_code": code,
                "test_type_label": TYPE_MAP.get(code, code),
                "duration_minutes": None if r["duration"].strip() in ("N/A", "") else r["duration"].strip(),
                "remote_testing": r["remote_testing"].strip(),
                "adaptive_irt": r["adaptive_irt"].strip(),
            }
        )
    return final


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python rebuild_catalog.py <raw_csv> <out_json>")
        sys.exit(1)
    final = clean(sys.argv[1])
    print("final catalog size:", len(final))
    print(Counter(x["test_type_code"] for x in final))
    json.dump(final, open(sys.argv[2], "w", encoding="utf-8"), indent=2, ensure_ascii=False)
