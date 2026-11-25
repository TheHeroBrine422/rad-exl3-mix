#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""testhf231.py -- CPU checks of w231's trainer changes on testhf225's tiny real Qwen3.5 stack (8 layers, two "cards" = two CPU device entries):
  1. --pipe-xfer host (hidden state / gradient through host buffers) = --pipe-xfer p2p = the naive split, bit for bit;
  2. data parallel: 2 ranks (gloo, two processes) x (split 4 + pipeline), V = 4 streams (rank r trains streams s % 2 == r) = the one-rank
     pipeline run within rounding (the gradient sums in another order): KL sums / rows equal, gnorm and parameters close; both ranks'
     parameters identical bit for bit; each rank logs its own segment wall (rank_wall).
Run like testhf225 (box2: HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= QAD_PIPELINE=... /data/q27/venv/bin/python testhf231.py). Exit 0 = all pass."""
import os, sys, tempfile
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "qad225"))
import testhf225 as H                                          # noqa: E402  (the toy golden, Tiny, run, targs)


LR0 = 1e-30                     # "lr 0": the trainer divides by --lr; an Adam step of 1e-30 leaves exp(r) = 1 exactly


def grads(T): return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1).double() for p in T.P])


def rank_main(r, world, port, out):
    os.environ.update(RANK=str(r), WORLD_SIZE=str(world), LOCAL_RANK="0", MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    z = {}
    for lr in (LR0, 1e-3):
        T, o = H.run(H.Tiny(0, split=4), H.targs(pipeline=True, pipe_xfer="host", V=4, lr=lr))
        z[lr] = {"o": o, "state": T.stu.state(), "grad": grads(T), "rank_wall": T.rank_wall, "mine": T.mine}
    torch.save(z, f"{out}/r{r}.pt")
    T.d.dist.destroy_process_group()


def main():
    Ts, os_ = H.run(H.Tiny(0, split=4), H.targs())
    Th, oh = H.run(H.Tiny(0, split=4), H.targs(pipeline=True, pipe_xfer="host"))
    Tp, op = H.run(H.Tiny(0, split=4), H.targs(pipeline=True, pipe_xfer="p2p"))
    H.check("--pipe-xfer host = --pipe-xfer p2p = naive split, bit for bit (KL sums, rows, gnorm, parameters)",
            oh == op == os_ and H.same(Th.stu.state(), Ts.stu.state()) and H.same(Tp.stu.state(), Ts.stu.state()), f"{oh[0]} vs {os_[0]}")

    one = {}
    for lr in (LR0, 1e-3):
        T1, o1 = H.run(H.Tiny(0, split=4), H.targs(pipeline=True, pipe_xfer="host", V=4, lr=lr)); one[lr] = (o1, T1.stu.state(), grads(T1))
    out = tempfile.mkdtemp()
    import torch.multiprocessing as mp
    port = 29000 + os.getpid() % 1000
    mp.start_processes(rank_main, args=(2, port, out), nprocs=2, join=True, start_method="spawn")
    z = [torch.load(f"{out}/r{r}.pt", weights_only=False) for r in range(2)]
    for lr in (LR0, 1e-3): print(f"lr {lr}: one rank {one[lr][0]}\n         two ranks {z[0][lr]['o']}", flush=True)
    H.check("data parallel: rank r trains streams s % 2 == r", z[0][LR0]["mine"] == [0, 2] and z[1][LR0]["mine"] == [1, 3], f"{z[0][LR0]['mine']} {z[1][LR0]['mine']}")
    H.check("data parallel: both ranks' parameters and accounting identical bit for bit", all(H.same(z[0][lr]["state"], z[1][lr]["state"]) and z[0][lr]["o"] == z[1][lr]["o"] for lr in (LR0, 1e-3)))
    o1, _, g1 = one[LR0]; o2, g2 = z[0][LR0]["o"], z[0][LR0]["grad"]
    gr = float((g2 - g1).norm() / g1.norm()); kl_ok = all(abs(a[0] - b[0]) <= 1e-9 * abs(b[0]) and a[1] == b[1] and a[3] == b[3] for a, b in zip(o2, o1))
    gn = max(abs(a[2] / b[2] - 1) for a, b in zip(o2, o1))
    H.check("data parallel = one rank at lr 1e-30 (3 steps; the same parameters each step): KL sums within 1e-9 relative, rows / prefix equal, the reduced gradient within 1e-6 relative (norm of the difference), gnorm within 1e-8",
            kl_ok and gr < 1e-6 and gn < 1e-8, f"grad rel {gr:.2e}, gnorm rel {gn:.2e}")
    o1, s1, _ = one[1e-3]; o2, s2 = z[0][1e-3]["o"], z[0][1e-3]["state"]
    dmax = max(float((s2[k] - s1[k]).abs().max()) for k in s1)
    H.check("data parallel at lr 1e-3: step 1 equal within rounding (KL sum exact, gnorm 1e-8); later steps drift (Adam's first steps move each scale by ~lr whatever the gradient's size, so rounding flips signs of near-zero gradients: info only)",
            o2[0][0] == o1[0][0] and abs(o2[0][2] / o1[0][2] - 1) < 1e-8, f"KL {[a[0] for a in o2]} vs {[b[0] for b in o1]}; parameter max diff after 3 steps {dmax:.2e}")
    H.check("data parallel: rank_wall logged for both ranks", len(z[0][LR0]["rank_wall"]) == 2 and all(x > 0 for x in z[0][LR0]["rank_wall"]), str(z[0][LR0]["rank_wall"]))
    print("\n" + ("ALL PASS" if not H.fails else f"FAILED: {H.fails}"))
    sys.exit(1 if H.fails else 0)


if __name__ == "__main__": main()
