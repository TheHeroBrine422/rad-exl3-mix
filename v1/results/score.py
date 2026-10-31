#!/usr/bin/env python3
"""score.py -- the challenge's score, as a reference implementation in numpy.

Any engine can be scored: produce the candidate's log-softmax (or logits) over the full vocabulary
at a document's scored positions, and pass them to `Reference.score_doc`. The arithmetic is the
leaderboard's exactly (radiance's --kld-ref and llama_kld): the reference's 512 tokens plus one
tail cell, the reference's cells renormalised, the candidate's tail cell everything outside the
reference's 512. The summaries are taken over the rows the reference's gen.u8 marks, the tokens the
model writes itself; every row is scored, so per-row results line up with the reference's rows.

    ref = Reference("golden/public")
    acc = Accumulator()
    for d in range(ref.n_docs):
        toks, score_from = ref.doc_tokens(d), ref.docs[d]["score_from"]
        logits = my_model(toks)                    # [len(toks), n_vocab], position i predicts i+1
        acc.add(ref.score_doc(d, logits[score_from:len(toks) - 1]), ref.docs[d]["source"])
    print(acc.report())

`python score.py GOLDEN --hf MODEL` runs a transformers model (bf16, one document at a time, the
LM head a slice of positions at a time) as an example; a long document still needs a device that
holds the model's activations over its whole length.
"""
import json
import os
import sys

import numpy as np

TINY = np.finfo(np.float64).tiny


class Reference:
    def __init__(self, d):
        self.dir = d
        self.meta = json.load(open(os.path.join(d, "kld.json")))
        if self.meta["format"] != "radiance-kld-top-1":
            raise ValueError(f"{d}: format {self.meta['format']}")
        self.top = self.meta["top"]
        self.n_vocab = self.meta["n_vocab"]
        self.docs = self.meta["docs"]
        self.n_docs = len(self.docs)
        rows = self.meta["rows"]
        self.tokens = np.memmap(os.path.join(d, "tokens.i32"), np.int32, "r")
        self.head = np.memmap(os.path.join(d, "top.bin"), np.dtype([("id", "<i4"), ("lp", "<f4")]), "r",
                              shape=(rows, self.top))
        self.rows = np.memmap(os.path.join(d, "rows.f32"), np.float32, "r", shape=(rows, 3))
        gp = os.path.join(d, "gen.u8")
        self.gen = np.fromfile(gp, np.uint8).astype(bool) if os.path.exists(gp) else np.ones(rows, bool)
        if len(self.gen) != rows:
            raise ValueError(f"{gp}: {len(self.gen)} rows, kld.json {rows}")
        off = 0
        self.off = []
        for doc in self.docs:
            self.off.append(off)
            off += doc["tokens"]

    def doc_tokens(self, d):
        return np.asarray(self.tokens[self.off[d]:self.off[d] + self.docs[d]["tokens"]])

    def n_rows(self, d):
        """The number of scored positions of document d."""
        return self.docs[d]["tokens"] - 1 - self.docs[d]["score_from"]

    def score_doc(self, d, cand, chunk=256):
        """Per-position (kl, top1, nll_q, nll_p, gen) for document d. `cand` is [rows, n_vocab] logits
        or log-probs of the candidate at the document's scored positions, in order."""
        n = self.n_rows(d)
        if tuple(np.shape(cand)) != (n, self.n_vocab):
            raise ValueError(f"doc {d}: want [{n}, {self.n_vocab}], got {list(np.shape(cand))}")
        parts = [self.score_rows(d, j, cand[j:j + chunk]) for j in range(0, n, chunk)]
        return tuple(np.concatenate(x) for x in zip(*parts))

    def score_rows(self, d, j0, cand):
        """score_doc for the scored positions j0 .. j0 + len(cand) of document d only, so a long
        document can be scored a slice of logits at a time."""
        doc = self.docs[d]
        cand = np.asarray(cand, np.float64)
        n = cand.shape[0]
        if j0 < 0 or j0 + n > self.n_rows(d) or cand.shape[1] != self.n_vocab:
            raise ValueError(f"doc {d}: rows {j0}+{n} of {self.n_rows(d)}, vocab {cand.shape[1]}")
        r0 = doc["row0"] + j0
        lq = cand - cand.max(1, keepdims=True)
        lq -= np.log(np.exp(lq).sum(1, keepdims=True))
        head = self.head[r0:r0 + n]
        ids, lp = head["id"].astype(np.int64), head["lp"].astype(np.float64)
        p = np.exp(lp)
        lq_head = np.take_along_axis(lq, ids, 1)
        # Summed outside the head, not taken as 1 - head: a tail of 1e-12 would cancel to noise.
        e = np.exp(lq)
        np.put_along_axis(e, ids, 0.0, 1)
        q_tail = e.sum(1)
        p_tail = self.rows[r0:r0 + n, 2].astype(np.float64)
        S = p.sum(1) + np.where(p_tail > 0, p_tail, 0)
        A = (np.where(p > 0, p * (lp - lq_head), 0)).sum(1)
        A += np.where(p_tail > 0, p_tail * (np.log(np.where(p_tail > 0, p_tail, 1)) - np.log(np.maximum(q_tail, TINY))), 0)
        kl = A / S - np.log(S)
        nxt = self.doc_tokens(d)[doc["score_from"] + 1 + j0:doc["score_from"] + 1 + j0 + n]
        top1 = lq.argmax(1) == ids[:, 0]
        nll_q = -lq[np.arange(n), nxt]
        return kl, top1, nll_q, self.rows[r0:r0 + n, 0].astype(np.float64), self.gen[r0:r0 + n]


class Accumulator:
    def __init__(self):
        self.kl, self.top1, self.nq, self.np_, self.src = [], [], [], [], []

    def add(self, scored, source=""):
        """One document's score_doc result; only the rows its gen mask marks are kept."""
        kl, top1, nq, np_, gen = scored
        self.kl.append(kl[gen]); self.top1.append(top1[gen]); self.nq.append(nq[gen]); self.np_.append(np_[gen])
        self.src.append(np.full(int(gen.sum()), source, dtype=object))

    def report(self):
        kl, top1 = np.concatenate(self.kl), np.concatenate(self.top1)
        nq, np_ = np.concatenate(self.nq), np.concatenate(self.np_)
        src = np.concatenate(self.src)
        long_ = np.array([str(s).startswith("agent-trace/") for s in src])
        return {
            "positions": int(kl.size),
            "kld": {"mean": float(kl.mean()), "median": float(np.median(kl)),
                    "p99": float(np.quantile(kl, 0.99)), "p99_9": float(np.quantile(kl, 0.999))},
            "top1_agreement": float(top1.mean()),
            "ppl": {"candidate": float(np.exp(nq.mean())), "reference": float(np.exp(np_.mean())),
                    "ratio": float(np.exp(nq.mean() - np_.mean()))},
            "kld_mean_long_agentic": float(kl[long_].mean()) if long_.any() else None,
        }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("golden")
    ap.add_argument("--hf", required=True, help="a transformers model directory or repo id")
    ap.add_argument("--docs", type=int, default=0, help="score only the first N documents")
    ap.add_argument("--chunk", type=int, default=256, help="scored positions per LM-head slice")
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM
    ref = Reference(a.golden)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(a.hf, dtype=torch.bfloat16).to(dev).eval()
    # The decoder's last hidden state, then the LM head a slice at a time: the full logits of a
    # 256K-token document would not fit anywhere.
    decoder, head = model.get_decoder(), model.get_output_embeddings()
    acc = Accumulator()
    n = a.docs or ref.n_docs
    for d in range(n):
        doc = ref.docs[d]
        toks = torch.from_numpy(ref.doc_tokens(d).astype(np.int64))[None].to(dev)
        parts = []
        with torch.no_grad():
            h = decoder(input_ids=toks).last_hidden_state[0, doc["score_from"]:doc["tokens"] - 1]
            for j in range(0, h.shape[0], a.chunk):
                logits = head(h[j:j + a.chunk]).float().cpu().numpy()
                parts.append(ref.score_rows(d, j, logits))
        acc.add(tuple(np.concatenate(x) for x in zip(*parts)), doc["source"])
        print(f"\r{d + 1}/{n}", end="", file=sys.stderr)
    print(file=sys.stderr)
    print(json.dumps(acc.report(), indent=1))


if __name__ == "__main__":
    main()
