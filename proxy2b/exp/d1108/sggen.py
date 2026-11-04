#!/usr/bin/env python3
"""sggen.py MODEL PROMPTS.npy OUTDIR [--batch 16] [--max-new 768] [--seed 0] -- d1108 step 2 (GPU): the model's own
continuations of equal-length prompts (sgprep.py), batched without padding: sampling with the model's
generation_config (do_sample, its temperature / top_p / top_k), stopping at <|im_end|>. One file per batch,
OUTDIR/b<i>.npz (gen [B, max_new] int32 padded with -1, len [B], finished [B]); finished batches are skipped
on a rerun. VRAM_CAP_GB caps the allocator (shared 4070)."""
import argparse, os, sys, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("prompts"); ap.add_argument("out")
ap.add_argument("--batch", type=int, default=16); ap.add_argument("--max-new", type=int, default=768)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
cap = float(os.environ.get("VRAM_CAP_GB", 0))
if cap: torch.cuda.set_per_process_memory_fraction(min(1.0, cap * 2**30 / torch.cuda.get_device_properties(0).total_memory))
P = np.load(a.prompts); os.makedirs(a.out, exist_ok=True)
tok = AutoTokenizer.from_pretrained(a.model)
END = tok.convert_tokens_to_ids("<|im_end|>")
m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()
gc = GenerationConfig.from_pretrained(a.model)
gc.do_sample = True; gc.max_new_tokens = a.max_new; gc.eos_token_id = END; gc.pad_token_id = END
print(f"sggen: {len(P)} prompts of {P.shape[1]}, batch {a.batch}, T {gc.temperature} top_p {gc.top_p} top_k {gc.top_k}", flush=True)
t0, ntok = time.time(), 0
for i in range(0, len(P), a.batch):
    f = f"{a.out}/b{i // a.batch:05d}.npz"
    if os.path.exists(f): continue
    torch.manual_seed(a.seed * 1000003 + i)
    x = torch.from_numpy(P[i:i + a.batch].astype(np.int64)).cuda()
    with torch.no_grad():
        y = m.generate(input_ids=x, attention_mask=torch.ones_like(x), generation_config=gc)[:, x.shape[1]:].cpu().numpy()
    G = np.full((len(x), a.max_new), -1, np.int32); L = np.zeros(len(x), np.int32); F = np.zeros(len(x), bool)
    for j, r in enumerate(y):
        e = np.nonzero(r == END)[0]
        n = int(e[0]) + 1 if len(e) else len(r)          # through the first <|im_end|>
        G[j, :n] = r[:n]; L[j] = n; F[j] = bool(len(e))
    np.savez(f + ".part.npz", gen=G, len=L, finished=F); os.replace(f + ".part.npz", f)
    ntok += int(L.sum())
    print(f"sggen: batch {i // a.batch}: {int(L.sum())} tokens, {F.mean():.2f} finished, {ntok / (time.time() - t0):.0f} tok/s", flush=True)
print("SGGEN DONE", flush=True)
