#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""genmask181.py check CALIB_GOLDEN TOKENIZER  /  genmask181.py make GOLDEN OUT.u8 TOKENIZER -- the challenge's gen mask
(README "The score"): a row is scored iff its next token is one the model writes itself: every token of a plain-text
document; in a chat (a document holding <|im_start|>) the body of each assistant turn, tool calls included, up to and
including the <|im_end|> closing it. System / user turns (tool results arrive in user turns), the `<|im_start|>role\n`
header and the tokens between turns are the prompt's (0). A chat piece that starts inside a turn (xcorp 8k pieces 1..3)
has an unknown role until its first header: 0 (conservative). `check` rebuilds calib's gen.u8 from its tokens and
reports the agreement (the rule's test); `make` writes OUT.u8 (one byte per golden row, data181.Gold's gen layout).
#181: the xcorp / trace goldens (gold3 / gold4 / night1105) were recorded without a mask, so their prompt-side rows
(arbitrary bf16 distributions, 2.5% of them > 1 nat between two bf16 implementations) were trained on."""
import json, os, sys

import numpy as np


def ids(tk):
    s, e = tk.convert_tokens_to_ids("<|im_start|>"), tk.convert_tokens_to_ids("<|im_end|>")
    a = tk.encode("assistant", add_special_tokens=False); nl = tk.encode("\n", add_special_tokens=False)
    assert len(a) == 1 and len(nl) == 1, (a, nl)
    return s, e, a[0], nl[0]


def doc_gen(t, S, E, A, NL):
    """per token k (k >= 1): 1 iff token k is model-written; rows j predict token j + 1"""
    n = len(t); w = np.zeros(n, np.uint8)
    st = np.where(t == S)[0]
    if len(st) == 0: w[:] = 1; return w
    for p in st:
        role_asst = p + 1 < n and t[p + 1] == A
        # header: <|im_start|> role ... \n  (the role's tokens up to the first newline token)
        q = p + 1
        while q < n and t[q] != NL: q += 1
        b = q + 1                                              # first body token
        if not role_asst: continue
        c = b
        while c < n and t[c] != E and t[c] != S: c += 1
        hi = c + 1 if c < n and t[c] == E else c               # <|im_end|> included
        w[b:hi] = 1
    return w


def rows_mask(g_docs, tok, off, S, E, A, NL, total_rows):
    out = np.zeros(total_rows, np.uint8)
    for i, d in enumerate(g_docs):
        t = np.asarray(tok[off[i]:off[i] + d["tokens"]])
        w = doc_gen(t, S, E, A, NL)
        j = np.arange(d["score_from"], d["tokens"] - 1)
        out[d["row0"]:d["row0"] + len(j)] = w[j + 1]
    return out


def load(gd):
    m = json.load(open(f"{gd}/kld.json")); tok = np.memmap(f"{gd}/tokens.i32", np.int32, "r")
    off = np.concatenate([[0], np.cumsum([d["tokens"] for d in m["docs"]])])
    return m, tok, off


def main():
    from transformers import AutoTokenizer
    mode = sys.argv[1]
    if mode == "check":
        gd, tkp = sys.argv[2], sys.argv[3]
        tk = AutoTokenizer.from_pretrained(tkp); S, E, A, NL = ids(tk)
        m, tok, off = load(gd); ref = np.fromfile(f"{gd}/gen.u8", np.uint8)
        mine = rows_mask(m["docs"], tok, off, S, E, A, NL, m["rows"])
        bad = np.where(mine != ref)[0]
        print(f"rows {m['rows']}, official gen {int(ref.sum())}, rebuilt {int(mine.sum())}, disagree {len(bad)} "
              f"({100 * len(bad) / m['rows']:.4f}%)")
        if len(bad):
            r0 = np.array([d["row0"] for d in m["docs"]]); di = np.searchsorted(r0, bad, side="right") - 1
            print("docs with disagreement:", len(set(di.tolist())), "first rows", bad[:10].tolist(), "docs", sorted(set(di.tolist()))[:10])
        return
    gd, out, tkp = sys.argv[2], sys.argv[3], sys.argv[4]
    tk = AutoTokenizer.from_pretrained(tkp); S, E, A, NL = ids(tk)
    m, tok, off = load(gd)
    mask = rows_mask(m["docs"], tok, off, S, E, A, NL, m["rows"])
    if os.path.exists(f"{gd}/gen.u8"):                         # a golden with its own mask: keep the intersection
        mask &= np.fromfile(f"{gd}/gen.u8", np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    mask.tofile(out + ".part"); os.replace(out + ".part", out)
    chat = sum(1 for i, d in enumerate(m["docs"]) if (np.asarray(tok[off[i]:off[i] + d["tokens"]]) == S).any())
    print(f"{os.path.basename(gd)}: {m['rows']} rows, gen {int(mask.sum())} ({100 * mask.mean():.1f}%), chat docs {chat} of {len(m['docs'])}")


if __name__ == "__main__":
    main()
