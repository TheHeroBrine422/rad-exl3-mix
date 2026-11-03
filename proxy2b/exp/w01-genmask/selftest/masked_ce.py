#!/usr/bin/env python3
"""masked_ce.py -- the YAQA_GENMASK loss (collect.py chunk_loss_m under checkpoint, chunks of CH) has the gradient of the
CE over the scored positions alone, at the hidden states (so at every linear above them), and zero for an empty mask."""
import torch
from torch.utils.checkpoint import checkpoint
torch.manual_seed(0); T, H, V, CH = 300, 32, 97, 64
head = torch.nn.Linear(H, V, bias=False)
def chunk_loss_m(h, y, m): return (torch.nn.functional.cross_entropy(head(h).float(), y, reduction="none") * m).sum()
def grads(mask):
    hid = torch.randn(T, H, generator=torch.Generator().manual_seed(1), requires_grad=True)
    y = torch.randint(0, V, (T,), generator=torch.Generator().manual_seed(2))
    loss = 0
    for c0 in range(0, T, CH):
        loss = loss + checkpoint(chunk_loss_m, hid[c0:c0 + CH], y[c0:c0 + CH], mask[c0:c0 + CH], use_reentrant=False)
    loss.backward(); g1 = hid.grad.clone(); hid.grad = None
    sel = mask.bool()
    ref = torch.nn.functional.cross_entropy(head(hid[sel]).float(), y[sel], reduction="sum") if sel.any() else (hid * 0).sum()
    ref.backward(); return g1, hid.grad, float(loss), float(ref)
ok = True
for name, mask in (("blocks", ((torch.arange(T) // 37) % 2).float()), ("random", (torch.rand(T) < 0.3).float()),
                   ("all", torch.ones(T)), ("none", torch.zeros(T))):
    g1, g2, l1, l2 = grads(mask)
    err = float((g1 - g2).abs().max()); good = err < 1e-6 and abs(l1 - l2) < 1e-3 * max(1, abs(l2))
    zero_rows = bool((g1[~mask.bool()].abs().max() == 0) if (~mask.bool()).any() else True)
    ok &= good and zero_rows
    print(f"{'PASS' if good and zero_rows else 'FAIL'}  {name:6s} loss {l1:.4f} vs {l2:.4f}, max |grad diff| {err:.1e}, "
          f"unscored rows' own gradient zero: {zero_rows}")
print("ALL PASS" if ok else "SOME FAILED")
