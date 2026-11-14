# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""strata186.py -- the stratum of a search document (RULES-strata.md §1, #186, w164).

N (non-Latin) when more than 50 % of the alphabetic characters of its decoded text are not Latin script
(unicodedata.name without "LATIN"; the rule of run183/ml186.py), otherwise L. script = the commonest non-Latin script
(the first word of unicodedata.name, as run186/classify186.py writes it), else "Latin". Computed once per document from
its whole token sequence; the slice parts inherit it (a calib document keeps its calib label through golden_pick).

  python3 -I search/v3/strata186.py label GOLDEN_DIR TOKENIZER OUT_KLD.json
        a copy of GOLDEN_DIR/kld.json whose docs carry "stratum" and "script" (the golden itself is not touched)
  python3 -I search/v3/strata186.py count OUT_KLD.json
        N / L documents and gen rows (from kld.json's docs, row0 / tokens) and the N scripts
Pure functions (label_text, label_docs) need no tokenizer: decode is any callable tokens -> str."""
import collections, json, os, sys, unicodedata

import numpy as np

NONLATIN_MIN = 0.5                              # RULES-strata §1: more than 50 %


def label_text(text):
    """(stratum, script, nonlatin share) of one decoded document"""
    letters = [c for c in text if c.isalpha()]
    if not letters: return "L", "Latin", 0.0
    names = [unicodedata.name(c, "") for c in letters]
    non = [n for n in names if "LATIN" not in n]
    share = len(non) / len(letters)
    if share <= NONLATIN_MIN: return "L", "Latin", share
    cnt = collections.Counter(n.split()[0] if n else "UNKNOWN" for n in non)
    return "N", cnt.most_common(1)[0][0], share


def label_docs(docs, tokens, decode):
    """docs: kld.json docs (tokens, the golden's order); tokens: one flat int array; decode: tokens -> text.
    Returns [(stratum, script, share)] per doc"""
    off = np.concatenate([[0], np.cumsum([d["tokens"] for d in docs])]).astype(np.int64)
    return [label_text(decode(np.asarray(tokens[off[i]:off[i + 1]]))) for i in range(len(docs))]


def label_golden(gdir, tokenizer, out):
    """write out (a kld.json copy) with stratum / script per doc; tokenizer: a HF tokenizer"""
    m = json.load(open(f"{gdir}/kld.json"))
    tok = np.memmap(f"{gdir}/tokens.i32", np.int32, "r")
    lab = label_docs(m["docs"], tok, lambda t: tokenizer.decode(t))
    for d, (s, sc, sh) in zip(m["docs"], lab):
        d["stratum"], d["script"] = s, sc
    m["strata"] = {"rule": "N if > 50 % of alphabetic characters are non-LATIN (RULES-strata §1)", "n_N": sum(s == "N" for s, _, _ in lab)}
    tmp = out + ".part"; json.dump(m, open(tmp, "w"), indent=0); os.replace(tmp, out)
    return m


def count(out):
    m = json.load(open(out)); D = m["docs"]
    gen = np.fromfile(os.path.join(os.path.dirname(out), "gen.u8"), np.uint8) if os.path.exists(os.path.join(os.path.dirname(out), "gen.u8")) else None
    res = collections.defaultdict(lambda: [0, 0, collections.Counter()])
    for d in D:
        s = d.get("stratum", "L"); n = d["tokens"] - 1 - d["score_from"]
        g = int(gen[d["row0"]:d["row0"] + n].sum()) if gen is not None else n
        res[s][0] += 1; res[s][1] += g
        if s == "N": res[s][2][d.get("script", "?")] += g
    return {k: {"docs": v[0], "gen_rows": v[1], "scripts": dict(v[2].most_common())} for k, v in res.items()}


def main(argv):
    if argv[0] == "label":
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(argv[2]); m = label_golden(argv[1], tk, argv[3])
        print(f"{argv[3]}: {len(m['docs'])} docs, N {m['strata']['n_N']}")
    elif argv[0] == "count":
        print(json.dumps(count(argv[1]), indent=1))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
