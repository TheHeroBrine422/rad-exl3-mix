"""common.py -- shared pieces of the Qwen3.5-2B proxy (ROADMAP "Research method and scale"): the model, the reference
golden format (the challenge's radiance-kld-top-1: top.bin [rows, 512] {i32 id, f32 log p}, rows.f32 [rows, 3] = nll of
the true token, entropy, tail mass summed outside the 512 from the full row; gen.u8; kld.json docs), and the scoring
arithmetic of the challenge's tools/score.py on the GPU (checked against it: check_score.py)."""
import json, os
import numpy as np
import torch

TOP = 512


def load_model(path, dev="cuda", dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(dev).eval()
    return m, m.get_decoder(), m.get_output_embeddings()


def doc_args(args):
    """golden_pick.py's document syntax: idx, idx:cap, idx:cap:from (cap 0 = whole document)"""
    out = []
    for x in args:
        i, _, c = x.partition(":"); c, _, f = c.partition(":")
        out.append((int(i), int(c) if c else 0, int(f) if f else 0))
    return out


class Golden:
    def __init__(self, d):
        self.dir = d; self.meta = json.load(open(f"{d}/kld.json")); m = self.meta
        assert m["format"] == "radiance-kld-top-1" and m["top"] == TOP, (d, m["format"])
        self.docs, R = m["docs"], m["rows"]
        self.tokens = np.memmap(f"{d}/tokens.i32", np.int32, "r")
        self.gen = np.fromfile(f"{d}/gen.u8", np.uint8).astype(bool)
        self.top = np.memmap(f"{d}/top.bin", np.dtype([("id", "<i4"), ("lp", "<f4")]), "r", shape=(R, TOP)) \
            if os.path.exists(f"{d}/top.bin") else None
        self.rows = np.memmap(f"{d}/rows.f32", np.float32, "r", shape=(R, 3)) if os.path.exists(f"{d}/rows.f32") else None
        self.toff = np.concatenate([[0], np.cumsum([x["tokens"] for x in self.docs])])

    def doc_tokens(self, i): return np.asarray(self.tokens[self.toff[i]:self.toff[i + 1]])


@torch.no_grad()
def ref_rows(lq):
    """The reference's rows from its log-softmax lq [n, V] (float64): top ids / log p, nll later, entropy, tail."""
    lp, ids = lq.topk(TOP, dim=1)
    e = lq.exp()
    ent = -(e * lq).sum(1)
    e.scatter_(1, ids, 0.0)
    return ids.int(), lp.float(), ent, e.sum(1)        # tail summed outside the head, not 1 - head


@torch.no_grad()
def score_rows(lq, ids, lp, p_tail):
    """score.py's Reference.score_rows (kl, top1) on the GPU: lq the candidate's log-softmax [n, V] float64, ids / lp
    the reference's top 512, p_tail its tail. KL = A / S - log S with p renormalised over head + tail cell."""
    lp = lp.double(); p = lp.exp()
    lq_head = lq.gather(1, ids.long())
    e = lq.exp(); e.scatter_(1, ids.long(), 0.0); q_tail = e.sum(1)
    S = p.sum(1) + torch.where(p_tail > 0, p_tail, torch.zeros_like(p_tail))
    A = torch.where(p > 0, p * (lp - lq_head), torch.zeros_like(p)).sum(1)
    tiny = torch.finfo(torch.float64).tiny
    A = A + torch.where(p_tail > 0, p_tail * (torch.log(torch.where(p_tail > 0, p_tail, torch.ones_like(p_tail)))
                                              - torch.log(q_tail.clamp_min(tiny))), torch.zeros_like(p_tail))
    kl = A / S - torch.log(S)
    top1 = lq.argmax(1) == ids[:, 0].long()
    return kl, top1


@torch.no_grad()
def doc_logsoftmax(decoder, head, toks, sf, chunk=256, dev="cuda"):
    """log-softmax (float64) of the scored positions sf .. len - 2 of one document, a chunk of rows at a time"""
    t = torch.from_numpy(toks.astype(np.int64))[None].to(dev)
    h = decoder(input_ids=t).last_hidden_state[0, sf:len(toks) - 1]
    for j in range(0, h.shape[0], chunk):
        yield j, torch.log_softmax(head(h[j:j + chunk]).double(), dim=1)
