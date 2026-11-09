#!/usr/bin/env python3
"""settle95.py job LAYER TENSOR SAMPLE OUTDIR DONE -- #126 (../w95.md): i8 vs i4 at the sub-1 levels on w87's two-sided
held-out proxy. For each K in W95_KS (default 0.125, 0.5, 1): settle87.job with SIG = [today's sigma for this tensor
at K] and LEVELS = [i4:K, i8:K], into OUTDIR/k<K>/ (settle87's file per sigma, skipped when present; its own .DONE
per K). DONE is written after all K. settle87 / settle77 are imported unchanged; only their module-level SIG / LEVELS
lists are set here. The i4 rows recompute w87's (same code, scales, Hessians): gate G0 in report95.py."""
import os, sys

_h = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(_h, "..", "w87")]
import settle87  # noqa: E402

KS = os.environ.get("W95_KS", "0.125,0.5,1").split(",")


def main():
    assert sys.argv[1] == "job", __doc__
    layer, tensor, sample, outdir, done = int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5], sys.argv[6]
    for Ks in KS:
        settle87.SIG = [settle87.sigma_current(tensor, float(Ks))]
        settle87.LEVELS = [("i4", Ks), ("i8", Ks)]
        settle87.job(layer, tensor, sample, os.path.join(outdir, f"k{Ks}"))
    if all(os.path.exists(os.path.join(outdir, f"k{Ks}", f"{layer}-{tensor}-s{sample}.DONE")) for Ks in KS):
        open(done, "w").close()


if __name__ == "__main__":
    main()
