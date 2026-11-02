#!/usr/bin/env python3
"""mutate.py RHV3_DIR -- mutation check of exl3i's selftest: build the plugin with one deliberate bug at a time and
confirm exl3i_gpu_test FAILS, and in the sections that should see it. A mutant the test passes is a hole in the test.
Each mutant: (name, file, old, new, sections expected to fail). Run on iceman, HOLDING THE CARD'S LOCK (mutants are
broken kernels: never beside another job on the card -- 2026-11-02 the host went down during an unlocked run):
  exec 5>/root/card1.lock; flock 5; exec 7>/root/ram.lock; flock -s 7; ROCR_VISIBLE_DEVICES=1 python -I mutate.py /root/rhv3
Results of the first run (6 of 10, cut by the crash): all caught."""
import os, re, shutil, subprocess, sys
R = sys.argv[1]; SRC = f"{R}/plugin"; MUT = "/data/plugcheck-mut"   # not on iceman's overlay root: copies there read EIO
MUTANTS = [
    ("relayout-no-transpose", "nt.c",
     "memcpy(out + (tk * nt + tn) * W, rec + (tn * kt + tk) * W, (size_t)W * 4);",
     "memcpy(out + (tn * kt + tk) * W, rec + (tn * kt + tk) * W, (size_t)W * 4);", "BCDE"),
    ("sub1-no-salt", "exl3i_rate.hip.h", "if constexpr (MODE == 1) st[i] ^= xr[r][i];", "", "AB"),
    ("i8-clamp-126", "exl3i_rate.hip.h", "const float q0 = fminf(fmaxf(rintf((float)v[0] * inv), -127.f), 127.f);",
     "const float q0 = fminf(fmaxf(rintf((float)v[0] * inv), -126.f), 126.f);", "A"),
    ("mode2-mid-off", "exl3i_rate.hip.h", "__builtin_amdgcn_ubfe(mid, off[r][i] - 16u, 16)", "__builtin_amdgcn_ubfe(mid, off[r][i] - 15u, 16)", "AB"),
    ("t2-acc-bump", "exl3i_nt.hip.h", "    __builtin_amdgcn_sched_barrier(0);   // the store stays after the WMMAs",
     "    if (c == 0) acc[0][0][0] += 0.25f;\n    __builtin_amdgcn_sched_barrier(0);   // the store stays after the WMMAs", "D"),
    ("frac-words-off", "exl3i_rate.hip.h", "static __device__ __forceinline__ int words(int m) { return m / 2; }",
     "static __device__ __forceinline__ int words(int m) { return m / 2 + (m % 16 ? 0 : 0) + (m > 80 ? 1 : 0); }", "AB"),
    ("seam-ignored", "nt.c", "d->cls[0].suh1 = ts1 ? ts1->data : 0;", "d->cls[0].suh1 = 0;", "CE"),
    ("act-unchecked", "nt.c", "if (g->act.kind != EXL3_ACT_F16) return RAD_E_UNSUPPORTED;", "", "F"),
    ("rate-unchecked-layout", "nt.c", "if (g.m != em || g.cb != ecb) return RAD_E_DTYPE;", "if (g.cb != ecb) return RAD_E_DTYPE;", "F"),
    ("int-i8-dropped", "exl3i_rate.hip.h", "for (int i = 0; i < 4; i++) x.u[i] = exl3i_i8_pair(x.u[i]);", "", "AB"),
]
only = sys.argv[2:] 
env = dict(os.environ, PATH="/opt/rocm/core-10.1/bin:" + os.environ["PATH"], ROCM_PATH="/opt/rocm/core-10.1")
holes = 0
for name, f, old, new, want in MUTANTS:
    if only and name not in only: continue
    if os.path.exists(MUT): shutil.rmtree(MUT)
    os.makedirs(f"{MUT}/plugin")
    for d in ("exl3import", "exl3r4d"): os.symlink(f"{SRC}/{d}", f"{MUT}/plugin/{d}")
    shutil.copytree(f"{SRC}/exl3i", f"{MUT}/plugin/exl3i", copy_function=shutil.copyfile)
    p = f"{MUT}/plugin/exl3i/{f}"; s = open(p).read()
    assert s.count(old) == 1, (name, s.count(old)); open(p, "w").write(s.replace(old, new))
    b = subprocess.run(f"cmake -S {MUT}/plugin/exl3i -B {MUT}/build -G Ninja -DCMAKE_PREFIX_PATH=/root/install-115 "
                       f"-DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DEXL3I_BUILD_ID=mutant-{name} && "
                       f"cmake --build {MUT}/build -j6", shell=True, capture_output=True, text=True, env=env)
    if b.returncode: print(f"{name:<24} BUILD FAIL\n{b.stdout[-1500:]}"); holes += 1; continue
    t = subprocess.run([f"{MUT}/build/exl3i_gpu_test", f"{MUT}/build/radiance_home/kernels/exl3i.so",
                        f"{R}/build-exl3r4d/radiance_home/kernels/exl3_r4d.so"], capture_output=True, text=True, timeout=900)
    failed = sorted({m.group(1)[0] for m in re.finditer(r"^FAIL: ([A-F])", t.stderr, re.M)} |
                    {l[0] for l in t.stdout.splitlines() if l[:1] in "ABCDEF" and re.search(r"[1-9]\d* (failed|differ|not byte|wrong)", l)})
    caught = t.returncode != 0
    miss = sorted(set(want) - set(failed))
    holes += (not caught)
    print(f"{name:<24} {'caught' if caught else 'NOT CAUGHT'}  failing sections {''.join(failed) or '-'}"
          f"  (expected {want}{', missed ' + ''.join(miss) if miss else ''})", flush=True)
shutil.rmtree(MUT, ignore_errors=True)
print("MUTATION", "PASS" if holes == 0 else f"FAIL ({holes} not caught)")
