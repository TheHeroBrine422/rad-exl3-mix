# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""test_strata186.py -- strata186.py's labels (RULES-strata §1). Pure checks always run; the golden check runs when
STRATA_GOLD (the ver3-8k golden dir) and STRATA_TOK (the bf16 tokenizer dir) are set (ds8: the calib golden's sibling
/data/q27/goldens/golden-ver3-8k, /data/q27/models/qwen38-27b-bf16), and must find exactly docs 5, 15, 18, 22, 97, 100
as N (the six non-Latin documents of ver3-8k, LOG #186).
python3 -I search/v3/test_strata186.py  (exit 0 iff every check passes)"""
import hashlib, json, os, sys, tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strata186 as ST  # noqa: E402

ok = True


def check(name, cond, msg=""):
    global ok
    ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'} {name} {msg}")


# pure rules
check("latin text is L", ST.label_text("The quick brown fox jumps over the lazy dog.")[:2] == ("L", "Latin"))
s, sc, sh = ST.label_text("Привет, как дела? Это длинный русский текст без латиницы.")
check("cyrillic text is N, script CYRILLIC", (s, sc) == ("N", "CYRILLIC"), f"share {sh:.2f}")
s, sc, sh = ST.label_text("这是一个中文句子，用来测试非拉丁文字的比例。 plus a little latin")
check("chinese text is N (CJK)", s == "N" and sc == "CJK", f"script {sc} share {sh:.2f}")
s, sc, sh = ST.label_text("Ich habe heute ein paar Bücher gekauft und Äpfel und ein Ähnliches Wort: Größe")
check("German with umlauts is L (Latin script)", s == "L" and sc == "Latin", f"share {sh:.3f}")
s, sc, sh = ST.label_text("abcd окок")                            # 4 Latin, 4 Cyrillic letters: exactly 50 % -> L
check("exactly 50 % non-Latin is L (rule: more than 50 %)", s == "L" and abs(sh - 0.5) < 1e-9, f"share {sh}")
s, sc, sh = ST.label_text("abc " + "ок" * 4)                       # 3 Latin, 8 Cyrillic
check("73 % non-Latin is N", s == "N" and sc == "CYRILLIC", f"share {sh:.2f}")
s, sc, sh = ST.label_text("12345 !!! ???")                         # no letters
check("no letters is L", s == "L" and sc == "Latin")
s, sc, sh = ST.label_text("مرحبا بك في العالم، هذا نص عربي")
check("arabic is N, script ARABIC", (s, sc) == ("N", "ARABIC"), f"share {sh:.2f}")
docs = [{"tokens": 3, "score_from": 0}, {"tokens": 2, "score_from": 0}]
toks = np.arange(5, dtype=np.int32)
lab = ST.label_docs(docs, toks, lambda t: "ок" if int(t[0]) >= 3 else "abc")
check("label_docs splits by document offsets", lab[0][0] == "L" and lab[1][0] == "N", str([x[:2] for x in lab]))

# the golden (ver3-8k): exactly the six N documents of LOG #186, labels written to a copy only
G, TOK = os.environ.get("STRATA_GOLD"), os.environ.get("STRATA_TOK")
if G and TOK:
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(TOK)
    before = hashlib.sha256(open(f"{G}/kld.json", "rb").read()).hexdigest()
    tmp = tempfile.mkdtemp(prefix="w164-strata-")
    out = f"{tmp}/kld.json"
    m = ST.label_golden(G, tk, out)
    nN = [i for i, d in enumerate(m["docs"]) if d["stratum"] == "N"]
    check("ver3-8k: N docs == 5, 15, 18, 22, 97, 100", nN == [5, 15, 18, 22, 97, 100], f"got {nN}")
    check("ver3-8k: every doc labelled", all(d.get("stratum") in ("L", "N") and d.get("script") for d in m["docs"]))
    check("ver3-8k: golden's kld.json untouched", hashlib.sha256(open(f"{G}/kld.json", "rb").read()).hexdigest() == before)
    check("ver3-8k: other fields unchanged", all({k: v for k, v in d.items() if k not in ("stratum", "script")} == json.load(open(f"{G}/kld.json"))["docs"][i] for i, d in enumerate(m["docs"])))
    for i in nN: print(f"   doc {i:3d} {m['docs'][i]['source']:12} {m['docs'][i]['script']}")
    print("   N docs by script:", {s: sum(1 for i in nN if m["docs"][i]["script"] == s) for s in sorted({m["docs"][i]["script"] for i in nN})})
else:
    print("SKIP golden check (set STRATA_GOLD and STRATA_TOK)")
print("ALL PASS" if ok else "FAILED")
sys.exit(0 if ok else 1)
