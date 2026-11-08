#!/usr/bin/env python3
"""heal90.py MODE OUT --wq DIR [heal.py options] -- w06's heal.py with one change (README.md, merge 11-08 A4 item 2):
env W90_TOPK=k (> 0) replaces the full-vocabulary KL(p_teacher || q_student) of every scored row by the KL of the
coarsened pair: the teacher's top-k tokens plus one lumped tail bin (the rest of the vocabulary), in training and in
the held-out early stop alike, i.e. what a stored top-k + tail teacher target (the challenge golden format) can give.
  KL_k = sum_{i in top_k(p)} p_i (log p_i - log q_i) + p_tail (log p_tail - log q_tail),
  log p_tail = logsumexp(teacher logits outside top_k) - logsumexp(all), q_tail the same with the student's logits.
KL_k <= the full KL (data processing). W90_TOPK unset or 0: heal.py unchanged (same function object)."""
import os, sys
import torch
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, os.environ.get("W06_DIR", "/workspace/rad-exl3-mix/proxy2b/exp/w06-heal"))
import heal

K = int(os.environ.get("W90_TOPK", "0") or 0)


def _tail(logits, idx):
    """log of the probability mass outside idx (per row), from the raw logits"""
    lse = torch.logsumexp(logits, -1)
    m = logits.scatter(-1, idx, float("-inf"))
    return torch.logsumexp(m, -1) - lse


def window_kl_topk(stu, tea, ids, pos, ch=128, grad=True):
    """heal.window_kl with the top-K + tail target (same chunking, same checkpointing, same sum over scored rows)"""
    p_ = torch.from_numpy(pos).to(next(stu.model.parameters()).device)
    ht = tea.hidden(ids)[p_]
    hs = stu.hidden(ids)[p_]

    def chunk(h, t):
        with torch.no_grad():
            zt = tea.logits(t); lp = torch.log_softmax(zt, -1)
            lpk, idx = lp.topk(K, -1)
            lpt = _tail(zt, idx)
        zs = stu.logits(h); lq = torch.log_softmax(zs, -1)
        lqk = lq.gather(-1, idx); lqt = _tail(zs, idx)
        return (lpk.exp() * (lpk - lqk)).sum() + (lpt.exp() * (lpt - lqt)).sum()
    tot = 0.0
    for c in range(0, hs.shape[0], ch):
        tot = tot + (checkpoint(chunk, hs[c:c + ch], ht[c:c + ch], use_reentrant=False) if grad else chunk(hs[c:c + ch], ht[c:c + ch]))
    return tot


if K > 0:
    heal.window_kl = window_kl_topk
    print(f"heal90: loss = top-{K} + tail KL (training and held-out)", flush=True)

if __name__ == "__main__":
    heal.main()
