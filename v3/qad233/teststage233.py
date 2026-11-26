#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""teststage233.py -- CPU checks of qad220 --pipe-procs (w233: the pipeline as S stage processes, gloo; S = 2 and 4) on testhf225's tiny REAL Qwen3.5
stack (8 layers, split 4), through the real Trainer.run_group, against today's one-process --pipeline (the reference), bit for bit:
  1. plain (2 processes, 3 steps): KL sums, rows, gnorm, prefix tokens, parameters;
  2. --state-card-gb + --state-budget-gb at a budget that evicts (each stage budgets its own card), also with --budget-evict-first;
     --state-early (a segment's state stored after its forward) on plain / budget / accum at V = 1; --kv-full;
  3. --accum 2 at V = 1;  4. --depth-test 512;
  5. a resume after step 1: the merged state file (snapshot_all) restored in fresh stage processes, every stream's state rebuilt by the pipelined
     prefix rebuild (prefix_stage) = the one-process resume (restore + prefix_state);
  6. fixed eval (Trainer.fixed through eval_stage) and the checkpoint (write_ckpt_all: scales files byte-identical, params.pt tensors equal);
  7. 2 data-parallel ranks x 2 stages (4 processes, V = 4) = the 2-rank one-process pipeline (testhf231's DP run);
  8. branch training (testbranch231's sessions: history walked once, cuts on state copies) = the one-process pipeline on the same arrangement;
  9. four stages (layers 2 / 2 / 2 / 2): plain, budget, accum, restore of the 2-stage file, --pipe-ahead 2 + fixed eval + checkpoint, branch.
Run like testhf225 (box2: HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= QAD_PIPELINE=CODE/pipeline /data/q27/venv/bin/python teststage233.py). Exit 0 = all pass."""
import hashlib, os, sys, tempfile
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "qad225")); sys.path.insert(0, os.path.join(HERE, "..", "qad231"))
import testhf225 as H                                          # noqa: E402  (Tiny, the toy golden C, targs, same)
Q, D = H.Q, H.D
K = 4


BOUNDS = {2: [K], 4: [2, 4, 6]}


def stage_tiny(stage, S=2, seed=0):
    """Tiny(seed, split 4) cut to one stage's layers (the others None), the head only on the last stage: what qmodel.Student(only=) builds"""
    stu = H.Tiny(seed, split=K); b = [0] + BOUNDS[S] + [H.NL]; own = range(b[stage], b[stage + 1])
    for i in range(H.NL):
        if i not in own: stu.layers[i] = None
    stu.P = {n: p for n, p in stu.P.items() if Q.layer_of(stu, n) in own}
    stu.qlin = {k: q for k, q in stu.qlin.items() if Q.layer_of(stu, k + ":r") in own}
    if stage < S - 1: stu.head = None
    stu.devs = [H.CPU]; stu.only = (b[stage], b[stage + 1])
    return stu


def run(stu, a, C, steps, nsteps=3, pos_off=0, restore=None, snap_after=None, out=None):
    """H.run's loop for a given C / plan; restore: a state file (snapshot) restored before the first step; snap_after: write snapshot_all after that
    many optimizer steps to out/snap.pt (stage mode: merged on rank 0) and stop"""
    stu.kvbuf = True; stu.pos_off = pos_off; stu.kv_cap = 0; stu.mask = None; Q.set_first_train(stu, lambda s: None)
    d = Q.Dist()
    if d.stages > 1:
        t_ = torch.tensor([float(stu.first_train)]); d.dist.all_reduce(t_, op=d.dist.ReduceOp.MIN); stu.first_train = stu.first_trainable = int(t_)
    rows_of, cum = Q.rows_cum(C, steps)
    T = Q.Trainer(a, stu, C, steps, rows_of, d, lambda s: None)
    s0 = 0
    if restore:
        z = torch.load(restore, weights_only=False); T.restore(z["snap"]); s0 = z["step"]
    res = []
    for s in range(s0, nsteps * a.accum, a.accum):
        accs, w = T.run_group(list(range(s, s + a.accum)), a.lr)
        res.append((float(accs[0]), float(accs[1]), T.gnorm, float(accs[-1])))
        if snap_after and len(res) == snap_after:
            z_ = T.snapshot_all() if T.stage_mode else T.snapshot()
            if d.rank == 0: torch.save({"snap": z_, "step": s + a.accum}, f"{out}/snap.pt")
            break
    return T, res


def plain_plan(a): return H.C, D.plan(H.C.items, np.arange(len(H.nt)), a.V, a.Q)


def case_args(case):
    if case == "plain": return dict(pipeline=True, pipe_xfer="host"), {}
    if case == "budget": return dict(pipeline=True, pipe_xfer="host", V=4, state_card_gb=1.0, state_budget_gb=3e-4), {}
    if case == "budget_ef": return dict(pipeline=True, pipe_xfer="host", V=4, state_card_gb=1.0, state_budget_gb=3e-4, budget_evict_first=True), {}
    if case == "accum": return dict(pipeline=True, pipe_xfer="host", V=1, accum=2), {}
    if case.startswith("early_"):                                  # --state-early on another case (accum at V = 1: every segment is the same stream's)
        ak, kw = case_args(case[6:]); return dict(ak, state_early=True), kw
    if case == "depth": return dict(pipeline=True, pipe_xfer="host"), {"pos_off": 512}
    if case == "dp": return dict(pipeline=True, pipe_xfer="host", V=4), {}
    raise KeyError(case)


def branch_setup():
    import testbranch231 as B
    gold, cuts = B.build_gold(os.path.join(tempfile.mkdtemp(prefix="ts233-sess-"), "sess"))      # deterministic (seed 7): the same golden in every process
    C, st = B.arrangement(gold, cuts, True)
    return C, st, B.QT


def proc_main(r, world, port, out, case, extra):
    """one process of a stage run (QAD_STAGES=2: rank r = 2 dp + stage) or of a one-process DP reference (QAD_STAGES=1)"""
    stages = extra.get("stages", 2)                               # 1 = a one-process dp reference rank; else the stages a pipeline
    os.environ.update(RANK=str(r), WORLD_SIZE=str(world), LOCAL_RANK="0", MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), QAD_STAGES=str(stages))
    torch.set_num_threads(2)
    if extra.get("budget_cpu"): Q.BUDGET_CPU[0] = True
    if extra.get("kv_full"): Q.KV_FULL[0] = True
    if case == "branch":
        C, steps, QT = branch_setup(); a = H.targs(pipeline=True, pipe_xfer="host", V=2, Q=QT); nsteps = 1; kw = {}
    else:
        ak, kw = case_args(case); a = H.targs(**ak); C, steps = plain_plan(a); nsteps = 3
    stu = stage_tiny(r % stages, stages) if stages > 1 else H.Tiny(0, split=K)
    a.pipe_ahead = extra.get("ahead", 0)
    T, res = run(stu, a, C, steps, nsteps=nsteps, restore=extra.get("restore"), snap_after=extra.get("snap_after"), out=out, **kw)
    z = {"res": res, "state": T.stu.state(), "evict": getattr(T, "n_evict", 0), "mine": T.mine}
    if extra.get("fixed"):
        fe = [(0, 0), (0, 3), (0, 5)]; z["fixed"] = T.fixed(fe).tolist()
        o = os.path.join(out, "ckpt"); T.write_ckpt_all(o, {"case": case})
    torch.save(z, f"{out}/r{r}.pt")
    if T.stage_mode: T.comm.close()
    T.d.barrier(); T.d.dist.destroy_process_group()


def spawn(world, case, extra=None):
    import torch.multiprocessing as mp
    out = tempfile.mkdtemp(prefix=f"ts233-{case}-"); port = 29000 + (os.getpid() * 7 + hash(case) + len(str(extra))) % 2000
    mp.start_processes(proc_main, args=(world, port, out, case, extra or {}), nprocs=world, join=True, start_method="spawn")
    return out, [torch.load(f"{out}/r{r}.pt", weights_only=False) for r in range(world)]


def merged(z, ranks=(0, 1)): return {k: v for r in ranks for k, v in z[r]["state"].items()}


def merged4(z): return merged(z, (0, 1, 2, 3))


def main():
    # 1-4: one dp rank, two stages, vs the one-process pipeline
    for case in ("plain", "budget", "budget_ef", "accum", "depth", "early_plain", "early_budget", "early_accum"):
        ak, kw = case_args(case); a = H.targs(**ak); C, steps = plain_plan(a)
        if "budget" in case: Q.BUDGET_CPU[0] = True
        Tr, ref = run(H.Tiny(0, split=K), a, C, steps, **kw); Q.BUDGET_CPU[0] = False
        _, z = spawn(2, case, {"budget_cpu": "budget" in case})
        ok = z[0]["res"] == ref and z[1]["res"] == ref and H.same(merged(z), Tr.stu.state()) and set(merged(z)) == set(Tr.stu.state())
        info = f"stages {z[0]['res'][0]} vs one process {ref[0]}"
        if "budget" in case: ok = ok and z[0]["evict"] + z[1]["evict"] > 0; info += f"; evictions stage 0 / 1: {z[0]['evict']} / {z[1]['evict']}"
        H.check(f"--pipe-procs = --pipeline bit for bit ({case}: KL sums, rows, gnorm, prefix tokens on both stages; parameters of both stages)", ok, info)
    # 4b: --kv-full (static KV buffers at the document's length) in the stages with --state-early and a budget = the one-process default values
    ak, kw = case_args("early_budget"); a = H.targs(**ak); C, steps = plain_plan(a); Q.BUDGET_CPU[0] = True
    Tr, ref = run(H.Tiny(0, split=K), a, C, steps, **kw); Q.BUDGET_CPU[0] = False
    _, z = spawn(2, "early_budget", {"budget_cpu": True, "kv_full": True})
    H.check("--kv-full + --state-early + budget (2 stages) = the one-process pipeline bit for bit", z[0]["res"] == ref and H.same(merged(z), Tr.stu.state()), f"{z[0]['res'][0]} vs {ref[0]}")
    # 5: resume from the merged state after step 1
    a = H.targs(pipeline=True, pipe_xfer="host"); C, steps = plain_plan(a)
    out1 = tempfile.mkdtemp(prefix="ts233-ref-")
    run(H.Tiny(0, split=K), a, C, steps, snap_after=1, out=out1)
    Tr, ref = run(H.Tiny(0, split=K), a, C, steps, restore=f"{out1}/snap.pt")
    snapdir, _ = spawn(2, "plain", {"snap_after": 1})
    zs = torch.load(f"{snapdir}/snap.pt", weights_only=False)
    _, z = spawn(2, "plain", {"restore": f"{snapdir}/snap.pt"})
    H.check("resume: the merged two-stage state file restored in fresh stages (every stream's state rebuilt through prefix_stage) = the one-process resume bit for bit",
            z[0]["res"] == ref and H.same(merged(z), Tr.stu.state()) and zs["snap"].get("stages") == 2 and ref[0][3] > 0, f"{z[0]['res']} vs {ref}")
    Tx, refx = run(H.Tiny(0, split=K), a, C, steps, restore=f"{snapdir}/snap.pt")
    H.check("resume: a one-process run restores the two-stage state file (Adam state by name) = the one-process resume bit for bit", refx == ref and H.same(Tx.stu.state(), Tr.stu.state()))
    _, z = spawn(2, "plain", {"restore": f"{out1}/snap.pt"})
    H.check("resume: stage processes restore a one-process state file (e.g. --fork-from a one-card state) = the one-process resume bit for bit",
            z[0]["res"] == ref and H.same(merged(z), Tr.stu.state()))
    # 6: fixed eval + checkpoint files
    Tr, ref = run(H.Tiny(0, split=K), a, C, steps)
    fe = [(0, 0), (0, 3), (0, 5)]; fr = Tr.fixed(fe).tolist(); oref = tempfile.mkdtemp(prefix="ts233-ck-")
    Q.write_ckpt(Tr.stu, oref, {"case": "plain"})
    outs, z = spawn(2, "plain", {"fixed": True})
    def files(d_): return {os.path.relpath(os.path.join(r_, f), d_): hashlib.sha1(open(os.path.join(r_, f), "rb").read()).hexdigest() for r_, _, fs in os.walk(d_) for f in fs if f != "params.pt"}
    pa, pb = torch.load(f"{oref}/params.pt", weights_only=False), torch.load(f"{outs}/ckpt/params.pt", weights_only=False)
    H.check("fixed eval through the stages = the one-process fixed eval bit for bit (both stages return the reduced values)", z[0]["fixed"] == fr and z[1]["fixed"] == fr, f"{z[0]['fixed'][:2]} vs {fr[:2]}")
    H.check("checkpoint: scales files byte-identical (stage 1 writes its own), params.pt holds both stages' tensors equal to the one-process file",
            files(oref) == files(f"{outs}/ckpt") and len(files(oref)) > 0 and H.same(pa["state"], pb["state"]) and set(pa["state"]) == set(pb["state"]),
            f"{len(files(oref))} scale files")
    # 7: 2 dp ranks x 2 stages vs the 2-rank one-process pipeline
    _, zr = spawn(2, "dp", {"stages": 1})
    _, z = spawn(4, "dp")
    H.check("2 dp ranks x 2 stages (4 processes) = 2 one-process dp ranks bit for bit (KL, rows, gnorm, parameters; streams s % 2 on dp rank r)",
            all(z[r]["res"] == zr[0]["res"] for r in range(4)) and H.same(merged(z, (0, 1)), zr[0]["state"]) and H.same(merged(z, (2, 3)), zr[1]["state"])
            and z[0]["mine"] == z[1]["mine"] == zr[0]["mine"] and z[2]["mine"] == zr[1]["mine"], f"{z[0]['res'][0]} vs {zr[0]['res'][0]}")
    # 8: branch training
    Cb, sb, QT = branch_setup(); ab = H.targs(pipeline=True, pipe_xfer="host", V=2, Q=QT)
    Tb, refb = run(H.Tiny(0, split=K), ab, Cb, sb, nsteps=1)
    _, z = spawn(2, "branch")
    H.check("branch training (sessions: history walked once, cuts on state copies): stages = one-process pipeline bit for bit", z[0]["res"] == refb and H.same(merged(z), Tb.stu.state()),
            f"{z[0]['res']} vs {refb}")
    # 9: four stages (layers 0-1 / 2-3 / 4-5 / 6-7) on one dp rank, against the same one-process references
    for case in ("plain", "budget", "accum", "early_accum", "early_budget"):
        ak, kw = case_args(case); a = H.targs(**ak); C, steps = plain_plan(a)
        if "budget" in case: Q.BUDGET_CPU[0] = True
        Tr, ref = run(H.Tiny(0, split=K), a, C, steps, **kw); Q.BUDGET_CPU[0] = False
        _, z = spawn(4, case, {"stages": 4, "budget_cpu": "budget" in case})
        ok = all(z[r]["res"] == ref for r in range(4)) and H.same(merged4(z), Tr.stu.state()) and set(merged4(z)) == set(Tr.stu.state())
        info = f"4 stages {z[0]['res'][0]} vs one process {ref[0]}" + (f"; evictions {[z[r]['evict'] for r in range(4)]}" if "budget" in case else "")
        H.check(f"--pipe-procs 4 = --pipeline bit for bit ({case}: KL, rows, gnorm, prefix tokens on every stage; parameters of all four)", ok and ("budget" not in case or sum(z[r]["evict"] for r in range(4)) > 0), info)
    a = H.targs(pipeline=True, pipe_xfer="host"); C, steps = plain_plan(a)
    Tr, ref = run(H.Tiny(0, split=K), a, C, steps, restore=f"{out1}/snap.pt")
    _, z = spawn(4, "plain", {"stages": 4, "restore": f"{snapdir}/snap.pt"})
    H.check("4 stages restore the 2-stage state file (another layout: Adam state by name; prefixes rebuilt through 4 stages) = the one-process resume bit for bit",
            z[0]["res"] == ref and H.same(merged4(z), Tr.stu.state()), f"{z[0]['res']} vs {ref}")
    Tr, ref = run(H.Tiny(0, split=K), a, C, steps); fr = Tr.fixed(fe).tolist()
    outs4, z = spawn(4, "plain", {"stages": 4, "fixed": True, "ahead": 2})
    pb = torch.load(f"{outs4}/ckpt/params.pt", weights_only=False)
    H.check("4 stages with --pipe-ahead 2 (deeper in flight): = one process bit for bit; fixed eval equal; checkpoint scales byte-identical, params.pt equal",
            all(z[r]["res"] == ref for r in range(4)) and z[0]["fixed"] == fr and z[3]["fixed"] == fr and files(oref) == files(f"{outs4}/ckpt") and H.same(pa["state"], pb["state"]))
    _, z = spawn(4, "branch", {"stages": 4})
    H.check("4 stages, branch training = one-process pipeline bit for bit", z[0]["res"] == refb and H.same(merged4(z), Tb.stu.state()), f"{z[0]['res']} vs {refb}")
    print("\n" + ("ALL PASS" if not H.fails else f"FAILED: {H.fails}"))
    sys.exit(1 if H.fails else 0)


if __name__ == "__main__": main()
