#!/usr/bin/env python3
"""check_rocmguard.py [SECONDS] -- rxq.forward's ROCm guard (#168): batched f32 trsm on transformers' GDN shapes
(torch.linalg.solve_triangular, A [1, 48, nc, 64, 64] unit lower triangular, B [1, 48, nc, 64, 128], nc 1..64 at random)
in a loop for SECONDS (default 120) a run, each result compared bit for bit with the same solve made at the start.
On the pip ROCm 7.14 stack the unguarded loop memory-faults in < 30 s (ds8 / box2, 11-12).
  env      importing rxq.forward sets ROCBLAS_USE_HIPBLASLT_BATCHED=0 and keeps a value already set (CPU)
  guard    the production guard (import rxq.forward first + rocm_pool_keep), forced pool trims every 64 calls: clean
  keep     rocm_pool_keep alone (ROCBLAS_USE_HIPBLASLT_BATCHED=1): clean
  control  no guard (ROCBLAS_USE_HIPBLASLT_BATCHED=1, pool trims at syncs): reported, not judged (a fault = this stack
           has the bug the guard covers; clean = it does not show here)
  f64      the guarded solve vs an f64 CPU solve: max relative difference <= 1e-5
GPU parts run only on a ROCm build with a device (else SKIP). Each loop is a child process (a fault kills it).
Last line "ROCMGUARD CHECKS n/m PASS"."""
import os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.dirname(HERE)


def child(arm, seconds):
    """one stress loop; exit 0 clean, 3 on a mismatch (a memory fault aborts the process)"""
    if arm == "guard":
        from rxq import forward as F       # sets ROCBLAS_USE_HIPBLASLT_BATCHED before torch
    import ctypes
    import torch
    dev = torch.device("cuda"); torch.set_grad_enabled(False)
    if arm in ("guard", "keep"):
        from rxq import forward as F
        assert F.rocm_pool_keep(dev)
    hip = ctypes.CDLL(next(ln.split()[-1] for ln in open("/proc/self/maps") if "libamdhip64.so" in ln))
    pool = ctypes.c_void_p(); assert hip.hipDeviceGetDefaultMemPool(ctypes.byref(pool), 0) == 0
    g = torch.Generator().manual_seed(0)
    IN, REF = {}, {}
    for nc in range(1, 65):
        k = torch.nn.functional.normalize(torch.randn(1, 48, nc, 64, 128, generator=g), dim=-1)
        beta = torch.rand(1, 48, nc, 64, generator=g); cum = (-torch.rand(1, 48, nc, 64, generator=g) * 0.1).cumsum(-1)
        pw = (cum.unsqueeze(-1) - cum.unsqueeze(-2)).masked_fill(torch.ones(64, 64, dtype=torch.bool).triu(1), float("-inf")).exp()
        kb = k * beta.unsqueeze(-1)
        IN[nc] = [x.to(dev) for x in ((kb @ k.transpose(-1, -2)) * pw, torch.randn(1, 48, nc, 64, 128, generator=g) * beta.unsqueeze(-1))]
        REF[nc] = torch.linalg.solve_triangular(*IN[nc], upper=False, unitriangular=True)
    torch.cuda.synchronize()
    if arm == "f64":
        A, B = IN[64]
        r = torch.linalg.solve_triangular(A.double().cpu(), B.double().cpu(), upper=False, unitriangular=True)
        d = float((REF[64].double().cpu() - r).abs().max() / r.abs().max())
        print(f"f64: max rel {d:.2e}", flush=True); sys.exit(0 if d <= 1e-5 else 3)
    rng = torch.Generator().manual_seed(1); n = 0; t0 = time.time()
    while time.time() - t0 < seconds:
        nc = int(torch.randint(1, 65, (1,), generator=rng))
        x = torch.linalg.solve_triangular(*IN[nc], upper=False, unitriangular=True)
        n += 1
        if not torch.equal(x, REF[nc]): print(f"MISMATCH call {n} nc {nc}", flush=True); sys.exit(3)
        if arm == "guard" and n % 64 == 0:
            torch.cuda.synchronize(); assert hip.hipMemPoolTrimTo(pool, ctypes.c_size_t(0)) == 0
    torch.cuda.synchronize()
    print(f"{arm}: {n} calls in {time.time() - t0:.0f} s, clean", flush=True)
    sys.exit(0)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child": child(sys.argv[2], float(sys.argv[3])); return
    seconds = sys.argv[1] if len(sys.argv) > 1 else "120"
    res = []

    def report(name, ok, why):
        res.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'} {name}: {why}", flush=True)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([PIPE, os.environ.get("PYTHONPATH", "")]))
    env.pop("ROCBLAS_USE_HIPBLASLT_BATCHED", None)
    code = "from rxq import forward; import os; print(os.environ['ROCBLAS_USE_HIPBLASLT_BATCHED'])"
    a = subprocess.run([sys.executable, "-c", code], env=dict(env, CUDA_VISIBLE_DEVICES=""), capture_output=True, text=True)
    b = subprocess.run([sys.executable, "-c", code], env=dict(env, CUDA_VISIBLE_DEVICES="", ROCBLAS_USE_HIPBLASLT_BATCHED="1"),
                       capture_output=True, text=True)
    report("env", a.stdout.strip() == "0" and b.stdout.strip() == "1", f"default -> {a.stdout.strip()!r}, preset 1 -> {b.stdout.strip()!r}")

    import torch
    if not (torch.version.hip and torch.cuda.is_available()):
        print("SKIP gpu parts: not a ROCm build with a device", flush=True)
    else:
        def run(arm, **extra):
            t = time.time()
            p = subprocess.run([sys.executable, __file__, "--child", arm, seconds], env=dict(env, **extra), capture_output=True, text=True)
            tail = (p.stdout + p.stderr).strip().splitlines()
            msg = next((ln for ln in tail if "Memory Fault" in ln or "MISMATCH" in ln or "clean" in ln or "f64:" in ln), tail[-1] if tail else "")
            return p.returncode, f"rc {p.returncode} after {time.time() - t:.0f} s: {msg[:160]}"
        rc, why = run("guard"); report("guard", rc == 0, why)
        rc, why = run("keep", ROCBLAS_USE_HIPBLASLT_BATCHED="1"); report("keep", rc == 0, why)
        rc, why = run("f64"); report("f64", rc == 0, why)
        rc, why = run("control", ROCBLAS_USE_HIPBLASLT_BATCHED="1")
        print(f"INFO control (unguarded): {'FAULTED: this stack has the #168 bug' if rc else 'clean here'} ({why})", flush=True)
    print(f"ROCMGUARD CHECKS {sum(res)}/{len(res)} {'PASS' if all(res) else 'FAIL'}", flush=True)
    sys.exit(0 if all(res) else 1)


if __name__ == "__main__":
    main()
