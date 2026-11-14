#!/usr/bin/env python3
"""mkset.py DL OUT -- the #182 vision eval set (w156): ~450 public image + text chat items in seven kinds, drawn with a
fixed seed from the pinned parquet shards dl.py fetched into DL. Writes OUT/img/<id>-<k>.<ext> (the original bytes,
JPEG / PNG only) and OUT/items.jsonl: {id, kind, source, licence, images: [file names], question}. No answers here:
those are model-written (gen182.py), then scored by the bf16 reference. Also OUT/text-items.jsonl, the format-matched
text-only control (dolly-15k prompts, same generation path). Never calib / public challenge data."""
import ast, glob, json, os, random, sys
import pyarrow.parquet as pq

DL, OUT = sys.argv[1:3]
os.makedirs(f"{OUT}/img", exist_ok=True)
rng = random.Random(182)
LIC = {"aokvqa": "A-OKVQA annotations Apache-2.0; COCO images (Flickr, CC licences per image)",
       "textvqa": "TextVQA CC BY 4.0; Open Images (CC BY 2.0)",
       "docvqa": "DocVQA (lmms-lab mirror card: Apache-2.0); UCSF Industry Documents Library pages",
       "infovqa": "InfographicVQA (lmms-lab mirror card: Apache-2.0); public web infographics",
       "chartqa": "ChartQA GPL-3.0 (charts from Statista / Pew / OWID / OECD)",
       "websight": "WebSight CC BY 4.0 (synthetic screenshots)",
       "screenspot": "ScreenSpot Apache-2.0",
       "nlvr2": "NLVR2 CC BY 4.0 (web images)",
       "mmmu": "MMMU Apache-2.0"}

def rows(pat):
    out = []
    for f in sorted(glob.glob(f"{DL}/{pat}")):
        out += pq.read_table(f).to_pylist()
    return out

def ext(b):
    return "jpg" if b[:3] == b"\xff\xd8\xff" else "png" if b[:4] == b"\x89PNG" else None

items = []
def add(kind, src, key, imgs, q):
    names = []
    for k, im in enumerate(imgs):
        b = im["bytes"]; e = ext(b or b"")
        if not e: return False
        n = f"{src}-{key}-{k}.{e}".replace("/", "_")
        with open(f"{OUT}/img/{n}", "wb") as f: f.write(b)
        names.append(n)
    items.append({"id": f"{src}-{key}", "kind": kind, "source": src, "licence": LIC[src], "images": names, "question": q})
    return True

def take(rs, n, fn):
    rs = rs[:]; rng.shuffle(rs); got = 0
    for r in rs:
        if got >= n: break
        got += bool(fn(r))
    return rs

# photo: A-OKVQA questions (45) + detailed descriptions of other A-OKVQA images (30)
ao = rows("HuggingFaceM4__A-OKVQA/data/*.parquet"); rng.shuffle(ao)
for r in ao[:45]:
    add("photo", "aokvqa", r["question_id"], [r["image"]], r["question"] + " Answer, then explain briefly.")
for r in ao[45:75]:
    add("photo", "aokvqa", r["question_id"] + "-desc", [r["image"]], "Describe this image in detail.")
# scene text: TextVQA (45), one question per image
tv = rows("lmms-lab__textvqa/data/*.parquet"); seen = set()
def tvf(r):
    if r["image_id"] in seen: return False
    seen.add(r["image_id"]); return add("scenetext", "textvqa", str(r["question_id"]), [r["image"]], r["question"][:1].upper() + r["question"][1:])
take(tv, 45, tvf)
# documents: DocVQA questions (60) + full-page transcriptions (15), distinct pages
dv = rows("lmms-lab__DocVQA/DocVQA/*.parquet"); seen = set(); cnt = [0]
def dvf(r):
    if r["docId"] in seen: return False
    seen.add(r["docId"]); cnt[0] += 1
    if cnt[0] <= 60: return add("document", "docvqa", r["questionId"], [r["image"]], r["question"])
    return add("document", "docvqa", r["questionId"] + "-ocr", [r["image"]], "Transcribe all the text on this page.")
take(dv, 75, dvf)
# infographics (30)
iv = rows("lmms-lab__DocVQA/InfographicVQA/*.parquet"); seen = set()
def ivf(r):
    if r["image_url"] in seen: return False
    seen.add(r["image_url"]); return add("infographic", "infovqa", r["questionId"], [r["image"]], r["question"])
take(iv, 30, ivf)
# charts: ChartQA questions (60) + chart-to-table (15)
cq = rows("lmms-lab__ChartQA/data/*.parquet"); rng.shuffle(cq)
for i, r in enumerate(cq[:60]): add("chart", "chartqa", f"{i}", [r["image"]], r["question"])
for i, r in enumerate(cq[60:75]): add("chart", "chartqa", f"{60 + i}-table", [r["image"]], "Extract the data in this chart as a Markdown table, then summarize the main trend in one sentence.")
# screenshots / UI: WebSight screenshot-to-HTML (30) + ScreenSpot element questions (45)
ws = rows("HuggingFaceM4__WebSight/data/*.parquet"); rng.shuffle(ws)
for i, r in enumerate(ws[:30]): add("screenshot", "websight", f"{i}", [r["image"]], "Write the HTML and CSS that reproduces this web page.")
ss = rows("rootsautomation__ScreenSpot/data/*.parquet"); rng.shuffle(ss)
for r in ss[:45]:
    add("screenshot", "screenspot", r["file_name"].rsplit(".", 1)[0], [r["image"]],
        f"In this screenshot, where is the element for \"{r['instruction']}\"? Describe where it is on the screen and what it does.")
# multi-image: NLVR2 pairs (45) + MMMU questions with two or more images (up to 30)
nl = rows("lmms-lab__NLVR2/data/*.parquet"); rng.shuffle(nl)
for r in nl[:45]:
    add("multiimage", "nlvr2", r["question_id"], [r["left_image"], r["right_image"]],
        f"Statement: \"{r['question']}\" Is the statement true or false for this pair of images (left, then right)? Explain.")
mm = [r for r in rows("MMMU__MMMU/*/validation-*.parquet") if sum(r[f"image_{k}"] is not None for k in range(1, 8)) >= 2]
rng.shuffle(mm)
for r in mm[:30]:
    ims = [r[f"image_{k}"] for k in range(1, 8) if r[f"image_{k}"] is not None]
    q = r["question"]
    try: opts = ast.literal_eval(r["options"])
    except Exception: opts = []
    if opts: q += "\n" + "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(opts))
    add("multiimage", "mmmu", r["id"], ims, q + "\n(The images are given in order: <image 1>, <image 2>, ...)")
with open(f"{OUT}/items.jsonl", "w") as f:
    for it in items: f.write(json.dumps(it) + "\n")
from collections import Counter
print(len(items), Counter(i["kind"] for i in items), sum(len(i["images"]) for i in items), "images")
