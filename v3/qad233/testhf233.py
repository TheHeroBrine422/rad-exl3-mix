#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""testhf233.py -- CPU checks of w233's trainer switches on testhf225's tiny REAL Qwen3.5 stack (8 layers, through Trainer.run_group, 3 steps):
  --keep-act (layers that keep their activations instead of the checkpoint recompute) = the checkpointed run, bit for bit (KL sums, rows, gnorm,
  parameters): one card (all layers / the back half / a frozen first half), the naive split, the pipeline (also with states kept on the cards).
Needs transformers (box2: HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= QAD_PIPELINE=CODE/pipeline /data/q27/venv/bin/python testhf233.py). Exit 0 = all pass."""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "qad225"))
import testhf225 as H                                          # noqa: E402  (the tiny stack, the toy golden, run, targs)

fails = []
def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{info}]" if info else ""), flush=True)
    if not ok: fails.append(name)


def keep(stu, layers): stu.keep_act = frozenset(layers); return stu


def main():
    T0, o0 = H.run(H.Tiny(0), H.targs()); s0 = T0.stu.state()
    T1, o1 = H.run(keep(H.Tiny(0), range(8)), H.targs())
    check("--keep-act all layers, one card: = checkpointed bit for bit", o1 == o0 and H.same(T1.stu.state(), s0), f"{o1[0]} vs {o0[0]}")
    T2, o2 = H.run(keep(H.Tiny(0), range(4, 8)), H.targs())
    check("--keep-act layers 4-7, one card: = checkpointed bit for bit", o2 == o0 and H.same(T2.stu.state(), s0), f"{o2[0]} vs {o0[0]}")
    Tf, of = H.run(H.Tiny(0), H.targs(), train_from=5); Tf1, of1 = H.run(keep(H.Tiny(0), range(8)), H.targs(), train_from=5)
    check("--keep-act with a frozen first half (train 5-7): = checkpointed bit for bit", of1 == of and H.same(Tf1.stu.state(), Tf.stu.state()))
    Ts, os_ = H.run(H.Tiny(0, split=4), H.targs())
    Ts1, os1 = H.run(keep(H.Tiny(0, split=4), range(4, 8)), H.targs())
    check("--keep-act 4-7, naive split: = checkpointed naive split bit for bit", os1 == os_ and H.same(Ts1.stu.state(), Ts.stu.state()))
    Tp, op = H.run(H.Tiny(0, split=4), H.targs(pipeline=True))
    for lay, nm in ((range(4, 8), "4-7 (card B)"), (range(0, 4), "0-3 (card A)"), (range(8), "all")):
        Tk, ok_ = H.run(keep(H.Tiny(0, split=4), lay), H.targs(pipeline=True))
        check(f"--keep-act {nm}, pipeline: = checkpointed pipeline bit for bit", ok_ == op and H.same(Tk.stu.state(), Tp.stu.state()), f"{ok_[0]} vs {op[0]}")
    Tq, oq = H.run(H.Tiny(0, split=4), H.targs(pipeline=True, state_card_gb=1.0))
    Tq1, oq1 = H.run(keep(H.Tiny(0, split=4), range(8)), H.targs(pipeline=True, state_card_gb=1.0))
    check("--keep-act all, pipeline with states kept on the cards: = checkpointed bit for bit", oq1 == oq and H.same(Tq1.stu.state(), Tq.stu.state()))
    H.Q.KL_KEEP[0] = True                                     # w233 --kl-keep: seg_kl without the checkpoint around the head + KL chunks
    Tk, ok_ = H.run(H.Tiny(0, split=4), H.targs(pipeline=True)); H.Q.KL_KEEP[0] = False
    check("--kl-keep, pipeline: = checkpointed head / KL bit for bit", ok_ == op and H.same(Tk.stu.state(), Tp.stu.state()), f"{ok_[0]} vs {op[0]}")
    print("\n" + ("ALL PASS" if not fails else f"FAILED: {fails}"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__": main()
