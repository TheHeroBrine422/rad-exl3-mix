#!/usr/bin/env python3
"""check_rxrun.py FORK [WORK] [--part pool|run|all] -- the local runner (rxrun, w105) on the CPU stand-in (rxq/cpukern.py,
the fork's real quantize.py), on check_cpujob's tiny two-layer model (layer 2 GDN + MLP, layer 3 attention + MLP;
random weights and Hessians), under the V3 settings (H_out, tile CD on attn_qg, C9 on the writers):
  pool   K2.5 and K0.5 i8 group jobs (8 a level) run by rxrun.pool as one `python -m rxq job` process a job (N = 1: the
         reference), by persistent workers N = 1 and N = 2 (one worker runs both levels), and through a shared queue on
         two devices: every t- / c9- / diag- file byte-identical to the reference; a worker killed mid-job and the run
         restarted: the rerun redoes only what is missing and is identical; the hold file stops new jobs
  run    rxrun.run on a two-level tiny ladder (layers 2-3, steps jobs + assemble; swj / post stubbed out): levels
         assembled, duplicate-seed jobs present beside the jobs (never in the level), a rerun skips everything, the
         status line has an ETA, rxrun.seedsd reports the pairs
  multi  rxrun.multi (w120) over two local hostd "hosts" (1 and 2 devices; A k_max 3, B k_min 3.25 and 2 levels a unit) and
         a third with a changed tree: every level built on exactly one host within its limits, the changed-tree host
         refused, eval units (stub pack / mix / score) on the host that holds their level with the results fields,
         a mix over levels on two hosts blocked, a rerun assigns nothing, a coordinator killed mid-unit and restarted
         records the unit once; a level equal to rxrun.run's (part run) byte for byte
Last line: "RXRUN CHECKS n/m PASS"."""
import argparse, json, os, re, shutil, signal, subprocess, sys, tempfile, threading, time

import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, PIPE)
from rxq import config as C, qwen35  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("fork"); ap.add_argument("work", nargs="?"); ap.add_argument("--part", default="all")
a = ap.parse_args()
W = a.work or tempfile.mkdtemp(prefix="w105rxrun-")
os.makedirs(W, exist_ok=True)
ENV = {"RXQ_CPU_STANDIN": "1", "RXQ_EXL3_SRC": f"{a.fork}/exllamav3/exllamav3_ext", "OMP_NUM_THREADS": "2",
       "PYTHONPATH": PIPE + os.pathsep + os.environ.get("PYTHONPATH", "")}
os.environ.update(ENV)
from rxrun import cmp, pool  # noqa: E402
res = []


def report(name, ok, why):
    res.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'} {name}: {why}", flush=True)


# ---- the tiny model (= check_cpujob.py's)
SH = {"ssm_qkv": (384, 256), "ssm_z": (256, 256), "ssm_out": (256, 256), "attn_qg": (256, 256), "attn_k": (128, 256),
      "attn_v": (128, 256), "attn_output": (256, 256), "ffn_gate": (384, 256), "ffn_up": (384, 256), "ffn_down": (256, 384)}
ck, hs, ho = f"{W}/ckpt", f"{W}/hess", f"{W}/hout/bn1920"
if not os.path.exists(f"{ck}/model.safetensors"):
    g = torch.Generator().manual_seed(86)
    for d in (ck, hs, ho): os.makedirs(d, exist_ok=True)
    tens = {}
    for l in (2, 3):
        for hname, ks in qwen35.layer_plan(l):
            k_in = SH[qwen35.family(ks[0])][1]
            A = torch.randn(4 * k_in, k_in, generator=g) * torch.linspace(0.3, 2.0, k_in)
            H = (A.T @ A).float(); iu = torch.triu_indices(k_in, k_in)
            save_file({"hessian_upper": H[iu[0], iu[1]].contiguous()}, f"{hs}/{hname}.safetensors", metadata={"k": str(k_in), "rows": str(4 * k_in)})
            for key in ks:
                n, k = SH[qwen35.family(key)]
                tens[f"{key}.weight"] = (torch.randn(n, k, generator=g) * 0.02).to(torch.bfloat16)
                save_file({"hout_diag": (torch.rand(n, generator=g) + 0.1).float()}, f"{ho}/{key}.safetensors")
    save_file(tens, f"{ck}/model.safetensors.part"); os.replace(f"{ck}/model.safetensors.part", f"{ck}/model.safetensors")
CFG = os.path.join(PIPE, "config", "v3.toml")
cfg = C.load(CFG); ST = C.settings_stamp(cfg)


def specs(K, tag, seed=0):
    d = f"{W}/specs-{tag}"; os.makedirs(d, exist_ok=True); out = []
    for l in (2, 3):
        for gi, (_, ks) in enumerate(qwen35.layer_plan(l)):
            sp = {"type": "rxq.quant", "version": 1, "level": f"t-k{K}-i8", "K": K, "codebook": "i8", "seed_offset": seed,
                  "keys": list(ks), "settings_stamp": ST, "config": CFG,
                  "inputs": {"ckpt": ck, "hessians": hs, "hin": None, "wpre": None, "hout": ho}}
            n = f"{tag}-L{l:02d}-g{gi}"; p = f"{d}/{n}.json"; json.dump(sp, open(p, "w")); out.append((n, p))
    return out


def jobs_of(sp, outdir):
    return [pool.make_job(n, p, f"{outdir}/{n}", cfg=cfg) for n, p in sp]


def dev(name="cpu0", n=1, **k):
    return {"name": name, "device": "cpu", "lock": f"{W}/card.lock", "ram_lock": f"{W}/ram.lock", "workers": n,
            "vram_gib": 28.0, "nice": 0, "env": {}, **k}


def same(ref, b, what):
    n, diff, _ = cmp.compare(ref, b, quiet=True)
    return not diff, f"{what}: {n} files, {len(diff)} differ{(' ' + str(diff[:2])) if diff else ''}"


if a.part in ("pool", "all"):
    S25, S05 = specs(2.5, "k2.5"), specs(0.5, "k0.5")
    t = time.time(); r = pool.run_jobs(jobs_of(S25, f"{W}/proc1/k2.5") + jobs_of(S05, f"{W}/proc1/k0.5"), dev(), f"{W}/proc1", mode="proc")
    rp = pool.run_jobs(jobs_of(S25, f"{W}/proc1b/k2.5"), dev(), f"{W}/proc1b", mode="proc")
    tp = time.time() - t
    nf = len(cmp.files(f"{W}/proc1"))
    raw = cmp.compare(f"{W}/proc1/k2.5", f"{W}/proc1b/k2.5", quiet=True, raw=True)[1]
    ok, why = same(f"{W}/proc1/k2.5", f"{W}/proc1b/k2.5", "proc vs proc")
    report("proc-ref", r["done"] == 16 and nf > 0 and ok, f"16 jobs (two levels) one process a job: {r}, {nf} t / c9 / diag files, "
           f"{tp:.0f} s; a second proc run of K2.5: {why} (whole-file sha256: {len(raw)} differ, the header's metadata order)")
    for arm, n in (("persist1", 1), ("persist2", 2)):
        t = time.time(); r = pool.run_jobs(jobs_of(S25, f"{W}/{arm}/k2.5") + jobs_of(S05, f"{W}/{arm}/k0.5"), dev(n=n), f"{W}/{arm}", mode="persist")
        ok, why = same(f"{W}/proc1", f"{W}/{arm}", arm)
        recs = [json.loads(x) for x in open(f"{W}/{arm}/jobs.jsonl")]
        conc = max(x["concurrent"] for x in recs)
        wl = [f for f in os.listdir(f"{W}/{arm}") if f.startswith("worker-")]
        report(arm, ok and r["done"] == 16 and conc <= n and len(wl) == n, f"{why}; done {r['done']}, max concurrent {conc}, "
               f"{len(wl)} worker logs, {time.time() - t:.0f} s (proc {tp:.0f} s)")
    # two devices on one shared queue
    q = pool.JobQueue(jobs_of(S25, f"{W}/twodev/k2.5"))
    rr = {}
    ths = [threading.Thread(target=lambda d: rr.__setitem__(d["name"], pool.run_jobs(q, d, f"{W}/twodev/{d['name']}", mode="persist")),
                            args=(dev(nm, lock=f"{W}/{nm}.lock"),)) for nm in ("cpu0", "cpu1")]
    for th in ths: th.start()
    for th in ths: th.join()
    ok, why = same(f"{W}/proc1/k2.5", f"{W}/twodev/k2.5", "two devices, one queue")
    report("two-devices", ok and sum(x["done"] for x in rr.values()) == 8 and all(x["done"] for x in rr.values()),
           f"{why}; per device {[(k, v['done']) for k, v in sorted(rr.items())]}")
    # kill a worker mid-run (SIGKILL to the worker process group), restart: only the missing work is redone
    K = f"{W}/kill"; shutil.rmtree(K, ignore_errors=True)
    stop = threading.Event()

    def killer():
        while not stop.is_set():
            lp = [p for p in (f"{K}/k2.5/{n}/log.jsonl" for n, _ in S25) if os.path.exists(p)]
            if len(lp) >= 3:
                out = subprocess.run(["pgrep", "-f", "rxrun.worker"], capture_output=True, text=True).stdout.split()
                for pid in out:
                    try: os.killpg(int(pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError): pass
                return
            time.sleep(0.05)
    th = threading.Thread(target=killer); th.start()
    try: pool.run_jobs(jobs_of(S25, f"{K}/k2.5"), dev(), K, mode="persist"); first = "finished"
    except pool.PoolError as e: first = f"PoolError ({str(e)[:60]})"
    stop.set(); th.join()
    part = sum(pool.job_done(json.load(open(p)), f"{K}/k2.5/{n}") for n, p in S25)
    r = pool.run_jobs(jobs_of(S25, f"{K}/k2.5"), dev(), K, mode="persist")
    ok, why = same(f"{W}/proc1/k2.5", f"{K}/k2.5", "after kill + rerun")
    report("kill-resume", ok and r["done"] == 8 - part and part < 8, f"first run {first}, {part}/8 jobs done; rerun ran {r['done']}; {why}")
    # hold file: no job starts once it exists; the run returns held
    H = f"{W}/hold"; open(f"{W}/HOLD", "w").close()
    r = pool.run_jobs(jobs_of(S25, f"{H}/k2.5"), dev(), H, mode="persist", hold=f"{W}/HOLD")
    os.remove(f"{W}/HOLD")
    report("hold", r["held"] and r["done"] == 0, f"{r}")

if a.part in ("run", "all"):
    R = f"{W}/ladder"
    conf = {"root": R, "settings": CFG, "python": sys.executable, "layers": "2-3", "passes": ["1"], "levels": ["3.5-i8", "3.25-i8"],
            "steps": ["jobs", "assemble"], "allow_standin": True, "dup_frac": 0.12, "template": None,
            "inputs": {"ckpt": ck, "hessians": hs, "hin": None, "hout": ho},
            "env": ENV, "devices": [{"name": "cpu0", "device": "cpu", "lock": f"{W}/card.lock", "ram_lock": f"{W}/ram.lock",
                                     "workers": 2, "vram_gib": 28.0, "nice": 0, "kinds": ["quant", "swj"]}]}
    json.dump(conf, open(f"{W}/ladder.json", "w"), indent=1)
    run = [sys.executable, "-m", "rxrun.run", f"{W}/ladder.json"]
    p = subprocess.run(run, capture_output=True, text=True, env=dict(os.environ))
    tail = (p.stdout + p.stderr).strip().splitlines()[-3:]
    from rxq import ladder
    lvs = [v for v in ladder.levels(cfg) if f"{ladder.fmt_k(v['K'])}-{v['codebook']}" in conf["levels"]]
    ok_lv = all(os.path.exists(f"{R}/{v['name']}/level/stamp.json") for v in lvs)
    shards = sorted(os.listdir(f"{R}/{lvs[0]['name']}/level")) if ok_lv else []
    dups = {v["name"]: sorted(os.listdir(f"{R}/{v['name']}/dup")) if os.path.isdir(f"{R}/{v['name']}/dup") else [] for v in lvs}
    dup_ok = all(dups.values()) and all(pool.job_done(json.load(open(f"{R}/{v}/dup/{d}/spec.json")), f"{R}/{v}/dup/{d}")
                                        for v, ds in dups.items() for d in ds)
    seed_ok = all(json.load(open(f"{R}/{v}/dup/{d}/spec.json"))["seed_offset"] == 1 for v, ds in dups.items() for d in ds)
    st = open(f"{R}/status.txt").read().strip() if os.path.exists(f"{R}/status.txt") else ""
    report("run-levels", p.returncode == 0 and ok_lv and "q-L02.safetensors" in shards and "q-L03.safetensors" in shards,
           f"rc {p.returncode}; {len(lvs)} levels assembled {ok_lv}: {shards[:4]}; {tail[-1:]}")
    report("run-dups", dup_ok and seed_ok, f"dup jobs {({k[-12:]: v for k, v in dups.items()})}, seed_offset 1 {seed_ok}")
    report("run-status", "ETA" in st and "jobs/h" in st, st[:200])
    t = time.time(); p2 = subprocess.run(run, capture_output=True, text=True, env=dict(os.environ))
    out2 = p2.stdout + p2.stderr
    report("run-rerun", p2.returncode == 0 and "skip" in out2 and time.time() - t < 30,
           f"rc {p2.returncode}, {time.time() - t:.1f} s: {[x for x in out2.splitlines() if 'skip' in x][:2]}")
    p3 = subprocess.run([sys.executable, "-m", "rxrun.seedsd", R], capture_output=True, text=True, env=dict(os.environ))
    report("seedsd", p3.returncode == 0 and "pairs" in p3.stdout, (p3.stdout.strip().splitlines() or [p3.stderr[-200:]])[-1][:200])
if a.part in ("multi", "all"):
    # rxrun.multi (w120): two local "hosts" (hostd processes, own roots and devices) on one queue; eval units through stub
    # pack / mix / score commands; a host with another tree stamp; a coordinator killed mid-unit and restarted
    from rxq import ladder
    MW = f"{W}/multi"; shutil.rmtree(MW, ignore_errors=True); os.makedirs(MW)
    names = {f"{ladder.fmt_k(v['K'])}-{v['codebook']}": v["name"] for v in ladder.levels(cfg)}
    LV = [names[k] for k in ("3.5-i8", "3.25-i8", "3.75-i8", "3-i8")]
    stub = f"{MW}/stub.py"
    open(stub, "w").write(r"""import hashlib, json, sys
import numpy as np
op = sys.argv[1]
if op == "pack": open(sys.argv[3], "w").write("pack of " + open(sys.argv[2]).read()[:200])
elif op == "mix": open(sys.argv[2], "w").write(open(sys.argv[3]).read() + open(sys.argv[4]).read() + "x" * 1000)
else:
    mix, g, o = sys.argv[2:5]; m = json.load(open(g + "/kld.json")); n = m["rows"]
    h = int(hashlib.sha256(open(mix, "rb").read()).hexdigest()[:8], 16)
    k = np.random.default_rng(h).gamma(0.5, 0.2, n).astype(np.float32); r = np.zeros((n, 4), np.float32); r[:, 0] = k
    r.tofile(o + ".json.rows")
    json.dump({"all": {"positions": n, "kld": {"mean": float(k.mean())}, "top1_agreement": 0.9}}, open(o + ".json", "w"))
""")
    for part, nd in (("x", 5), ("c", 3)):
        gd = f"{MW}/golden-t-{part}"; os.makedirs(gd); docs, r = [], 0
        for i in range(nd): docs.append({"tokens": 101 + 10 * i, "score_from": 0, "row0": r, "orig": i}); r += 100 + 10 * i
        json.dump({"docs": docs, "rows": r}, open(f"{gd}/kld.json", "w")); np_ones = bytes([1]) * r; open(f"{gd}/gen.u8", "wb").write(np_ones)
    open(f"{MW}/base.rad", "w").write("base")
    PYX = sys.executable
    from rxrun import hostd
    tree2 = f"{MW}/tree2"
    for d_ in hostd.TREE_DIRS:
        if os.path.isdir(os.path.join(PIPE, "..", d_)): shutil.copytree(os.path.join(PIPE, "..", d_), f"{tree2}/{d_}", ignore=shutil.ignore_patterns("__pycache__"))
    with open(f"{tree2}/pipeline/rxq/config.py", "a") as f: f.write("\n# a changed tree\n")

    def host(nm, devs, unit_levels, limits, tree=os.path.join(PIPE, "..")):
        H = f"{MW}/{nm}"; os.makedirs(H, exist_ok=True)
        rc = {"root": f"{H}/ladder", "settings": CFG, "python": PYX, "layers": "2-3", "passes": ["1"], "steps": ["jobs", "assemble"],
              "allow_standin": True, "dup_frac": 0.0, "template": None, "inputs": {"ckpt": ck, "hessians": hs, "hin": None, "hout": ho},
              "env": ENV, "devices": [{"name": f"{nm}-{d}", "device": "cpu", "lock": f"{H}/card{d}.lock", "ram_lock": f"{H}/ram.lock",
                                       "workers": 1, "vram_gib": 28.0, "nice": 0, "kinds": ["quant", "swj"]} for d in range(devs)]}
        json.dump(rc, open(f"{H}/run.json", "w"), indent=1)
        hc = {"name": nm, "root": H, "run_conf": f"{H}/run.json", "tree": tree, "slots": {"level": 1, "eval": 1}, "limits": limits,
              "eval": {"packs": f"{H}/packs", "aliases": {"BASE": f"{MW}/base.rad", "GX": f"{MW}/golden-t-x", "GC": f"{MW}/golden-t-c"},
                       "card_lock": f"{H}/card0.lock", "ram_lock": f"{H}/ram.lock", "conv_lock": f"{H}/conv.lock",
                       "pack_cmd": [PYX, stub, "pack", "{recipe}", "{out}"], "mix_cmd": [PYX, stub, "mix", "{out}", "{base}", "{mixfile}"],
                       "score_cmd": [PYX, stub, "score", "{mix}", "{golden}", "{out}"]}}
        json.dump(hc, open(f"{H}/host.json", "w"), indent=1)
        return {"name": nm, "cmd": [], "python": PYX, "hostconf": f"{H}/host.json", "unit_levels": unit_levels, "reconnect_s": 5}
    # level units here have no post / recipe step: the stub pack reads final.recipe, so give each built level one
    hA = host("hA", 1, 1, {"kinds": ["level", "eval"], "k_max": 3.0})
    hB = host("hB", 2, 2, {"kinds": ["level", "eval"], "k_min": 3.25})
    hC = host("hC", 1, 1, {"kinds": ["level", "eval"]}, tree=tree2)
    ev = [{"tag": "e-u3", "mix": {"base": "BASE", "trunk": LV[3]}, "slices": [{"name": "t", "parts": {"x": "GX", "c": "GC"}}]},
          {"tag": "e-u35", "mix": {"base": "BASE", "trunk": LV[0]}, "slices": [{"name": "t", "parts": {"x": "GX", "c": "GC"}}]},
          {"tag": "e-split", "mix": {"base": "BASE", "parity": {"even": LV[3], "odd": LV[0]}}, "slices": [{"name": "t", "parts": {"x": "GX"}}]}]
    open(f"{MW}/evals.jsonl", "w").write("".join(json.dumps(e) + "\n" for e in ev))
    CO = {"root": f"{MW}/coord", "settings": CFG, "passes": ["1"], "levels": LV, "layers": "2-3", "hosts": [hA, hB, hC],
          "evals": f"{MW}/evals.jsonl", "retries": 0}
    json.dump(CO, open(f"{MW}/multi.json", "w"), indent=1)
    # final.recipe stand-in: written by a watcher as levels get DONE (steps jobs + assemble have no recipe step)
    stopw = threading.Event()

    def recipes():
        while not stopw.is_set():
            for p_ in __import__("glob").glob(f"{MW}/h?/ladder/*/DONE"):
                rp_ = os.path.join(os.path.dirname(p_), "final.recipe")
                if not os.path.exists(rp_): open(rp_, "w").write(os.path.dirname(p_))
            time.sleep(0.2)
    thw = threading.Thread(target=recipes, daemon=True); thw.start()
    t = time.time()
    p = subprocess.run([sys.executable, "-m", "rxrun.multi", f"{MW}/multi.json"], capture_output=True, text=True, env=dict(os.environ), timeout=3600)
    mlog = open(f"{MW}/coord/multi.log").read() if os.path.exists(f"{MW}/coord/multi.log") else p.stdout + p.stderr
    where = {n: [h for h in ("hA", "hB", "hC") if os.path.exists(f"{MW}/{h}/ladder/{n}/DONE")] for n in LV}
    once = all(len(v) == 1 for v in where.values())
    kv = {v["name"]: v["K"] for v in ladder.levels(cfg)}
    limit_ok = all((kv[n] <= 3.0) == (v == ["hA"]) for n, v in where.items())
    st = [json.loads(x) for x in open(f"{MW}/coord/state.jsonl")]
    units = [e for e in st if e["ev"] == "assign" and e["kind"] == "level"]
    two = any(e["host"] == "hB" and len(e["levels"]) == 2 for e in units)
    report("multi-levels", p.returncode == 0 and once and limit_ok and two and not any(e["host"] == "hC" for e in st if e["ev"] == "assign"),
           f"rc {p.returncode}, {time.time() - t:.0f} s; level -> host {({n[-12:]: v for n, v in where.items()})}; limits (hA K <= 3, hB K >= 3.25) {limit_ok}; "
           f"a 2-level unit on hB {two}; units {[(e['host'], len(e['levels'])) for e in units]}")
    report("multi-tree-check", "hC: REFUSED" in mlog and "tree_stamp" in mlog, [x for x in mlog.splitlines() if "REFUSED" in x][:1])
    res_ = [json.loads(x) for x in open(f"{MW}/coord/results.jsonl")] if os.path.exists(f"{MW}/coord/results.jsonl") else []
    good = [r for r in res_ if not r.get("fail")]
    fields = all(all(k in r for k in ("tag", "bytes", "positions", "kld", "p99", "top1", "slice", "host")) for r in good)
    hosts_ok = all(r["host"] == where[LV[3] if r["tag"] == "e-u3" else LV[0]][0] for r in good)
    docs_ok = all(os.path.exists(f"{MW}/coord/docs/{r['tag']}-{r['slice']}.json") for r in good)
    pos_ok = all(r["positions"] == sum(100 + 10 * i for i in range(5)) + sum(100 + 10 * i for i in range(3)) for r in good)
    split_blocked = "e-split BLOCKED" in mlog and not any(r["tag"] == "e-split" for r in res_)
    report("multi-evals", len(good) == 2 and fields and hosts_ok and docs_ok and pos_ok and split_blocked,
           f"{[(r['tag'], r['host'], round(r['kld'], 4), r['positions']) for r in good]}; fields {fields}, on the level's host {hosts_ok}, "
           f"docs {docs_ok}; parity mix over two hosts blocked {split_blocked}")
    n_assign = len([e for e in st if e["ev"] == "assign"])
    t = time.time(); p2 = subprocess.run([sys.executable, "-m", "rxrun.multi", f"{MW}/multi.json"], capture_output=True, text=True, env=dict(os.environ), timeout=600)
    st2 = [json.loads(x) for x in open(f"{MW}/coord/state.jsonl")]
    report("multi-rerun", p2.returncode == 0 and len([e for e in st2 if e["ev"] == "assign"]) == n_assign, f"rc {p2.returncode}, {time.time() - t:.1f} s, "
           f"assignments {n_assign} -> {len([e for e in st2 if e['ev'] == 'assign'])}")
    # the coordinator killed while a level unit runs: the unit goes on (own process), the restart picks up its result
    LV2 = [names["4-i8"]]
    CO2 = dict(CO, root=f"{MW}/coord2", levels=LV2, hosts=[hB], evals=None)
    json.dump(CO2, open(f"{MW}/multi2.json", "w"), indent=1)
    pk = subprocess.Popen([sys.executable, "-m", "rxrun.multi", f"{MW}/multi2.json"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                          env=dict(os.environ), start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < 600 and not (os.path.exists(f"{MW}/coord2/state.jsonl") and "assign" in open(f"{MW}/coord2/state.jsonl").read()): time.sleep(0.2)
    time.sleep(2); os.killpg(pk.pid, signal.SIGKILL); pk.wait()
    alive_after = any(open(f).read().strip() and os.path.exists(f"/proc/{open(f).read().strip()}") for f in __import__("glob").glob(f"{MW}/hB/units/*/pid"))
    p3 = subprocess.run([sys.executable, "-m", "rxrun.multi", f"{MW}/multi2.json"], capture_output=True, text=True, env=dict(os.environ), timeout=1800)
    st3 = [json.loads(x) for x in open(f"{MW}/coord2/state.jsonl")]
    a3 = [e for e in st3 if e["ev"] == "assign"]; r3 = [e for e in st3 if e["ev"] == "result"]
    report("multi-restart", p3.returncode == 0 and len(a3) == 1 and len(r3) == 1 and r3[0]["rc"] == 0 and os.path.exists(f"{MW}/hB/ladder/{LV2[0]}/DONE"),
           f"unit alive after the coordinator's kill {alive_after}; assignments {len(a3)}, results {[(e['id'], e['rc']) for e in r3]}, rc {p3.returncode}")
    stopw.set()
    if os.path.isdir(f"{W}/ladder/{LV[0]}/level") and os.path.isdir(f"{MW}/{where[LV[0]][0]}/ladder/{LV[0]}/level"):
        ok, why = same(f"{W}/ladder/{LV[0]}/level", f"{MW}/{where[LV[0]][0]}/ladder/{LV[0]}/level", "multi level vs rxrun.run level")
        report("multi-identity", ok, why)
print(f"RXRUN CHECKS {sum(res)}/{len(res)} PASS")
sys.exit(0 if all(res) else 1)
