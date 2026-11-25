#!/usr/bin/env python3
"""tag.py -- the two existing origins of xgap-v1, tagged.
  calib   : the challenge calib split (raw/challenge/corpus/calib.jsonl; never the public split) -> corpus/calib/calib.jsonl
            tags calib_idx, pool (pools-v3.json), judge = memberships {verify, j217, jc228, jlong}, cont228_train
  xcorp-v1: /data/xcorp/v1/*.jsonl (every `pisess*` file refused) -> corpus/xcorp-v1/<file>.jsonl, tag = source file
Lines get the manifest `id` (calib-NNNNN, xcorp-<file>-NNNNNN); the manifest rows go to work/tags-<origin>.jsonl.
Inputs (small json files from rad-local/workers/data/w229) in ROOT/inputs/."""
import glob, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

INP = f"{ROOT}/inputs"
PREAMBLE = re.compile(r"Reasoning effort is set to \w+\.[^\n]*?(?:final answer\.|brief and focused[^\n]*?elaboration\.)\n*")
SYS = re.compile(r"<\|im_start\|>system\n(.*?)<\|im_end\|>\n", re.S)


def chat_tags(prompt):
    """thinking (on = the xhigh / low preamble present, else off) and has_system (real system text beyond preamble + tools)."""
    m = SYS.match(prompt)
    think = "on" if m and "Reasoning effort is set to" in m.group(1) else "off"
    if not m: return think, False
    s = PREAMBLE.sub("", m.group(1))
    if "# Tools" in s:
        a = s.find("# Tools"); b = s.find("</IMPORTANT>")
        s = s[:a] + (s[b + len("</IMPORTANT>"):] if b >= 0 else "")
    return think, bool(s.strip())


def calib(rev):
    pools = json.load(open(f"{INP}/pools-v3.json"))["pool"]
    j217 = set(json.load(open(f"{INP}/j217-calib-ids.json")))
    jlong = set(json.load(open(f"{INP}/jlong-ids.json")))
    cp = json.load(open(f"{INP}/contpool.json"))
    jc228 = set(cp["judge"]); ctrain = {i for i, _ in cp["train"]}
    os.makedirs(f"{ROOT}/corpus/calib", exist_ok=True)
    rows = []
    with open(f"{ROOT}/corpus/calib/calib.jsonl", "w") as fo:
        for i, l in enumerate(open(CALIB)):
            d = json.loads(l); pool = pools[i]
            judge = [n for n, ok in (("verify", pool == "verify"), ("j217", i in j217), ("jc228", i in jc228), ("jlong", i in jlong)) if ok]
            did = f"calib-{i:05d}"
            fo.write(json.dumps({"id": did, "source": d["source"], "category": d["category"], "kind": d["kind"], "tokens": d["tokens"],
                                 "prompt": d["prompt"]}, ensure_ascii=False) + "\n")
            r = {"id": did, "origin": "calib", "cell": None, "category": d["category"], "source": d["source"], "dataset": "StillDeadcode/qwen3.8-27b-quant-challenge",
                 "config": "corpus/calib.jsonl", "split": "calib", "revision": rev, "licence": "challenge corpus (per-source licences)", "kind": d["kind"],
                 "tokens": d["tokens"], "length_band": band(d["tokens"]), "holdout": False, "judge": judge, "pool": pool, "calib_idx": i,
                 "cont228_train": i in ctrain, "file": "corpus/calib/calib.jsonl", "line": i}
            if d["kind"] == "chat": r["thinking"], r["has_system"] = chat_tags(d["prompt"])
            rows.append(r)
    with open(f"{ROOT}/work/tags-calib.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("calib", len(rows))


def xcorp():
    os.makedirs(f"{ROOT}/corpus/xcorp-v1", exist_ok=True)
    rows = []; skipped = []
    for p in sorted(glob.glob(f"{XCORP}/*.jsonl")):
        name = os.path.basename(p)
        if name.startswith("pisess") or "pisess" in name: skipped.append(name); continue
        slug = name[:-6]
        st = json.load(open(f"{XCORP}/{slug}.stats.json")) if os.path.exists(f"{XCORP}/{slug}.stats.json") else {}
        rel = f"corpus/xcorp-v1/{name}"
        with open(f"{ROOT}/{rel}", "w") as fo:
            for i, l in enumerate(open(p)):
                d = json.loads(l); did = f"xcorp-{slug}-{i:06d}"
                fo.write(json.dumps({"id": did, "source": d["source"], "category": d["category"], "kind": d["kind"], "tokens": d["tokens"],
                                     "prompt": d["prompt"]}, ensure_ascii=False) + "\n")
                r = {"id": did, "origin": "xcorp-v1", "cell": None, "category": d["category"], "source": d["source"], "xcorp_file": name,
                     "dataset": st.get("repo"), "config": st.get("config"), "split": st.get("split"), "revision": st.get("revision"),
                     "licence": "per source dataset (see sources.json)", "kind": d["kind"], "tokens": d["tokens"], "length_band": band(d["tokens"]),
                     "holdout": False, "file": rel, "line": i}
                if d["kind"] == "chat": r["thinking"], r["has_system"] = chat_tags(d["prompt"])
                rows.append(r)
    with open(f"{ROOT}/work/tags-xcorp.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("xcorp", len(rows), "refused pisess files:", skipped)


if __name__ == "__main__":
    revs = os.environ["CALIB_REVISION"]
    calib(revs); xcorp()
