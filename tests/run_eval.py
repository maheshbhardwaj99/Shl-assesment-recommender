"""
Lightweight local eval harness.

This is NOT the official evaluator (we didn't get access to the real 10-trace
zip — see APPROACH.md). It scripts through our own synthetic traces
(tests/traces.json) turn by turn against a running local server, and checks:
  - schema compliance on every response,
  - every recommended url actually exists in our catalog (no hallucination),
  - the 8-turn cap is never exceeded,
  - the specific behavior each trace is designed to probe (clarify-first,
    compare grounding, refuse on off-topic/injection, refine growing the
    shortlist, etc).

Run with the server already up: `python tests/run_eval.py`
"""
import json
import sys
import time
from pathlib import Path

import requests

BASE_URL = "http://localhost:8000"
TRACES_PATH = Path(__file__).parent / "traces.json"
CATALOG_PATH = Path(__file__).parent.parent / "data" / "shl_catalog.json"


def load_catalog_urls():
    data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    return {d["url"] for d in data}


def post_chat(messages):
    r = requests.post(f"{BASE_URL}/chat", json={"messages": messages}, timeout=30)
    r.raise_for_status()
    return r.json()


def check_schema(resp, errors, trace_id, turn_idx):
    for key in ("reply", "recommendations", "end_of_conversation"):
        if key not in resp:
            errors.append(f"[{trace_id} turn {turn_idx}] missing key '{key}'")
    if not isinstance(resp.get("recommendations"), list):
        errors.append(f"[{trace_id} turn {turn_idx}] recommendations is not a list")
    else:
        for rec in resp["recommendations"]:
            for k in ("name", "url", "test_type"):
                if k not in rec:
                    errors.append(f"[{trace_id} turn {turn_idx}] recommendation missing '{k}'")
        if len(resp["recommendations"]) > 10:
            errors.append(f"[{trace_id} turn {turn_idx}] more than 10 recommendations")


def main():
    catalog_urls = load_catalog_urls()
    traces = json.loads(TRACES_PATH.read_text(encoding="utf-8"))

    total_checks = 0
    failed_checks = 0
    errors = []

    for trace in traces:
        tid = trace["id"]
        messages = []
        last_resp = None
        first_resp = None
        max_turns_seen = 0

        for i, turn in enumerate(trace["turns"]):
            messages.append({"role": "user", "content": turn["user"]})
            time.sleep(2.5)  # stay under free-tier requests-per-minute limits
            try:
                resp = post_chat(messages)
            except Exception as e:  # noqa: BLE001
                errors.append(f"[{tid} turn {i}] request failed: {e}")
                failed_checks += 1
                continue

            check_schema(resp, errors, tid, i)
            max_turns_seen = len(messages) + 1

            # hallucination backstop check
            for rec in resp.get("recommendations", []):
                total_checks += 1
                if rec["url"] not in catalog_urls:
                    failed_checks += 1
                    errors.append(f"[{tid} turn {i}] HALLUCINATED url not in catalog: {rec['url']}")

            if first_resp is None:
                first_resp = resp
            last_resp = resp
            messages.append({"role": "assistant", "content": resp["reply"]})

            if resp.get("end_of_conversation"):
                break

        exp = trace.get("expect", {})

        if "recommendations_empty" in exp:
            total_checks += 1
            ok = len(last_resp.get("recommendations", [])) == 0
            if ok != exp["recommendations_empty"]:
                failed_checks += 1
                errors.append(f"[{tid}] expected recommendations_empty={exp['recommendations_empty']}")

        if exp.get("final_recommendations_nonempty"):
            total_checks += 1
            if len(last_resp.get("recommendations", [])) == 0:
                failed_checks += 1
                errors.append(f"[{tid}] expected a non-empty final shortlist, got none")

        if "keywords_any" in exp:
            total_checks += 1
            names = " ".join(r["name"].lower() for r in last_resp.get("recommendations", []))
            if not any(kw in names for kw in exp["keywords_any"]):
                failed_checks += 1
                errors.append(f"[{tid}] none of keywords {exp['keywords_any']} found in recommendation names: {names}")

        if exp.get("turn_cap_honored"):
            total_checks += 1
            if max_turns_seen > 8:
                failed_checks += 1
                errors.append(f"[{tid}] turn cap exceeded: saw {max_turns_seen}")

        print(f"{'OK ' if tid not in ' '.join(errors) else 'ISS'} {tid}: reply='{(last_resp or {}).get('reply','')[:80]}...'")

    print("\n" + "=" * 60)
    print(f"checks: {total_checks}  failed: {failed_checks}")
    if errors:
        print("\nissues:")
        for e in errors:
            print(" -", e)
    sys.exit(1 if failed_checks else 0)


if __name__ == "__main__":
    main()
