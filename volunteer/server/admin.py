#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""admin.py --db QUEUE.sqlite [--data DATAROOT] [--results DIR] COMMAND ...  (on the server host; no network API)
(--db / --data / --results default to $RXMQ_DB / $RXMQ_DATA / $RXMQ_RESULTS, as in the server container)

  invite [--uses N]                      an invite code for worker registration (printed once)
  add-worker NAME                        a worker token directly (printed once)
  workers                                registered workers, last seen, declared caps
  disable-worker ID [--enable]           revoke (or restore) a worker's token
  datapack MODEL REV CKPT HESS [--hout SNAP DIR] [--no-head]
                                         cut CKPT into per-layer weight files under DATAROOT/models/MODEL@REV/ and link
                                         the Hessians (and an H_out snapshot); then run `manifest`
  manifest [--jobs N]                    sha256 + size of every file under DATAROOT (incremental) -> manifest.json
  enqueue-level --model M --rev R --K K --codebook CB --level NAME [settings] [--layers A-B] [--head]
                                         one quant_layer job per layer (+ quant_head): a ladder level
  enqueue-noop N [--seconds S] [--size B] [--inputs a,b] [--tag T]
  list [--state S] [--level L] [--type T]   jobs
  events [JOB]                           the audit log
  requeue IDS|--level L|--state S [--force]   back to the queue (accepted ones only with --force)
  cancel IDS|--level L
  export --level L OUT [--partial]       accepted shards of a level in ourq.py's layout (q-LNN.safetensors + index)
  check --level L [--rows-frac F]        run checker/check.py on the level's accepted results, store the verdict"""
import argparse, concurrent.futures as cf, json, os, shutil, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.util import read_json, write_json_atomic, sha256_file, st_header, safe_relpath, fmt_bytes
from common.st import copy_tensors
import jobs
from jobs import qwen, quant_common as QC
from server.store import Store


def man_files(data):
    return (read_json(os.path.join(data, "manifest.json"), {}) or {}).get("files", {})


def cmd_manifest(a):
    old = man_files(a.data); todo, out = [], {}
    for dp, dn, fn in os.walk(a.data, followlinks=True):
        dn[:] = sorted(d for d in dn if not d.startswith("."))
        for f in sorted(fn):
            if f.startswith(".") or f.endswith(".part"): continue
            p = os.path.join(dp, f); rel = os.path.relpath(p, a.data)
            if rel == "manifest.json": continue
            if not safe_relpath(rel): print(f"skip (name not allowed): {rel}"); continue
            st = os.stat(p); o = old.get(rel)
            if o and o["size"] == st.st_size and o.get("mtime") == st.st_mtime: out[rel] = o
            else: todo.append((rel, p, st))
    t0 = time.time(); nb = sum(s.st_size for _, _, s in todo)
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        for (rel, p, st), hx in zip(todo, ex.map(lambda t: sha256_file(t[1]), todo)):
            out[rel] = {"sha256": hx, "size": st.st_size, "mtime": st.st_mtime}
    write_json_atomic(os.path.join(a.data, "manifest.json"), {"files": dict(sorted(out.items())), "built": time.time()})
    print(f"manifest: {len(out)} files, hashed {len(todo)} ({fmt_bytes(nb)}) in {time.time() - t0:.1f}s")


def cmd_datapack(a):
    root = os.path.join(a.data, "models", f"{a.model}@{a.rev}")
    for d in ("weights", "hessians"): os.makedirs(os.path.join(root, d), exist_ok=True)
    cfg = json.load(open(os.path.join(a.ckpt, "config.json")))
    cfg["quantization_config"] = {"quant_method": "exl3"}                   # ourq's template (as proxy2b/q2b.sh)
    write_json_atomic(os.path.join(root, "config.json"), cfg)
    wm = json.load(open(os.path.join(a.ckpt, "model.safetensors.index.json")))["weight_map"]
    nl = (cfg.get("text_config") or cfg)["num_hidden_layers"]
    t0 = time.time()
    lo, hi = map(int, (a.layers or f"0-{nl - 1}").split("-"))
    for l in range(lo, hi + 1):
        out = os.path.join(root, "weights", f"L{l:02d}.safetensors")
        if os.path.exists(out): continue
        copy_tensors(a.ckpt, wm, [f"{k}.weight" for k in qwen.keys(qwen.layer_plan(a.arch, l))], out)
    if not a.no_head:
        out = os.path.join(root, "weights", "head.safetensors")
        if "lm_head.weight" not in wm: print("no lm_head.weight in the checkpoint (tied embeddings?): no head file")
        elif not os.path.exists(out): copy_tensors(a.ckpt, wm, ["lm_head.weight"], out)
    for f in sorted(os.listdir(a.hess)):
        if f.endswith(".safetensors") and not os.path.lexists(os.path.join(root, "hessians", f)):
            os.symlink(os.path.abspath(os.path.join(a.hess, f)), os.path.join(root, "hessians", f))
    if a.hout:
        snap, d = a.hout; od = os.path.join(root, "hout", snap); os.makedirs(od, exist_ok=True)
        for f in sorted(os.listdir(d)):
            if f.endswith(".safetensors") and not os.path.lexists(os.path.join(od, f)):
                os.symlink(os.path.abspath(os.path.join(d, f)), os.path.join(od, f))
    print(f"datapack {a.model}@{a.rev}: layers {lo}-{hi} in {time.time() - t0:.0f}s -> {root}; now run `manifest`")


def settings_from(a):
    if a.settings_json: return json.load(open(a.settings_json))
    return {"sigma_reg": a.sigma_reg, "sigma_reg_in": a.sigma_reg_in, "sigma_reg_out": a.sigma_reg_out,
            "scaled_hessian": a.scaled_hessian, "qtf2": a.qtf2, "hout": a.hout_snap, "seed_offset": a.seed_offset}


def add_job(S, db, man, jtype, p, level, ukey, priority=0):
    m = jobs.check_params(jtype, p)
    req, opt = m.inputs(p)
    import re
    names = list(req) + sorted(n for n in man if any(re.fullmatch(rx, n) for rx in opt))
    miss = [n for n in req if n not in man]
    if miss: sys.exit(f"{jtype} {ukey}: inputs not in the manifest: {miss[:4]}")
    jobs.check_inputs(jtype, p, names)
    return S.enqueue(db, jtype, m.VERSION, p, names, m.requires(p), sum(man[n]["size"] for n in names),
                     level=level, ukey=ukey, priority=priority)


def cmd_enqueue_level(a, S):
    man = man_files(a.data)
    base = {"model": a.model, "revision": a.rev, "arch": a.arch, "K": int(a.K) if float(a.K).is_integer() else float(a.K),
            "codebook": a.codebook, "level": a.level, "settings": settings_from(a)}
    base["stamp"] = QC.settings_stamp(base)
    cfg = read_json(os.path.join(a.data, "models", f"{a.model}@{a.rev}", "config.json"))
    nl = (cfg.get("text_config") or cfg)["num_hidden_layers"]
    lo, hi = map(int, (a.layers or f"0-{nl - 1}").split("-"))
    n = 0
    with S.tx() as db:
        for l in range(lo, hi + 1):
            n += add_job(S, db, man, "quant_layer", {**base, "layer": l}, a.level, f"quant_layer/v{jobs.get('quant_layer').VERSION}:{base['stamp']}:L{l:02d}") is not None
        if a.head:
            n += add_job(S, db, man, "quant_head", base, a.level, f"quant_head/v{jobs.get('quant_head').VERSION}:{base['stamp']}:head", priority=1) is not None
    print(f"level {a.level} (stamp {base['stamp']}): {n} new jobs")


def cmd_enqueue_noop(a, S):
    man = man_files(a.data); ids = []
    with S.tx() as db:
        for i in range(a.n):
            p = {"seconds": a.seconds, "size": a.size, "seed": a.seed + i}
            if a.inputs: p["inputs"] = [x for x in a.inputs.split(",") if x]
            if a.tag: p["tag"] = a.tag
            if a.fail: p["fail"] = True
            ids.append(add_job(S, db, man, "noop", p, a.tag or None, None))
    print("enqueued", ids)


def cmd_list(a, S):
    w, args = [], []
    for k in ("state", "level", "type"):
        if getattr(a, k): w.append(f"{k}=?"); args.append(getattr(a, k))
    rows = S.q("SELECT * FROM jobs" + (" WHERE " + " AND ".join(w) if w else "") + " ORDER BY id", args)
    for r in rows:
        p = json.loads(r["params"]); what = f"L{p['layer']:02d}" if "layer" in p else ("head" if r["type"] == "quant_head" else "")
        print(f"{r['id']:5d} {r['type']:11s} {what:5s} {r['state']:9s} att {r['attempts']}/{r['max_attempts']} "
              f"lvl {r['level'] or '-':12s} by {r['accepted_by'] or '-'} chk {r['check_state'] or '-'} {(r['error'] or '')[-80:]}")
    print(f"{len(rows)} jobs")


def ids_or(a, S, where):
    if a.ids: return [int(x) for x in a.ids]
    if a.level: return [r[0] for r in S.q(f"SELECT id FROM jobs WHERE level=? AND {where}", (a.level,))]
    if getattr(a, "state", None): return [r[0] for r in S.q(f"SELECT id FROM jobs WHERE state=? AND {where}", (a.state,))]
    sys.exit("give job ids, --level or --state")


def cmd_requeue(a, S):
    ids = ids_or(a, S, "1"); n = 0
    with S.tx() as db:
        for i in ids:
            j = db.execute("SELECT * FROM jobs WHERE id=?", (i,)).fetchone()
            if j is None or j["state"] == "queued": continue
            if j["state"] == "accepted" and not a.force: print(f"{i}: accepted (use --force)"); continue
            if j["state"] == "accepted" and a.results:
                d = os.path.join(a.results, str(i))
                if os.path.exists(d): os.replace(d, d + f".old-{int(time.time())}")
            if j["lease_id"]: db.execute("UPDATE leases SET state='cancelled', ended=? WHERE id=?", (time.time(), j["lease_id"]))
            db.execute("UPDATE jobs SET state='queued', attempts=0, lease_id=NULL, lease_owner=NULL, lease_expires=NULL, "
                       "accepted_lease=NULL, accepted_by=NULL, accepted_at=NULL, check_state=NULL, updated=? WHERE id=?", (time.time(), i))
            S.event(db, "requeue", i); n += 1
    print(f"requeued {n}")


def cmd_cancel(a, S):
    ids = ids_or(a, S, "state IN ('queued','leased')")
    with S.tx() as db:
        for i in ids:
            j = db.execute("SELECT * FROM jobs WHERE id=?", (i,)).fetchone()
            if j is None or j["state"] not in ("queued", "leased"): continue
            if j["lease_id"]: db.execute("UPDATE leases SET state='cancelled', ended=? WHERE id=?", (time.time(), j["lease_id"]))
            db.execute("UPDATE jobs SET state='cancelled', lease_id=NULL, lease_owner=NULL, lease_expires=NULL, updated=? WHERE id=?",
                       (time.time(), i))
            S.event(db, "cancel", i)
    print(f"cancelled {len(ids)}")


def level_jobs(S, level):
    return [r for r in S.q("SELECT * FROM jobs WHERE level=? AND type IN ('quant_layer','quant_head') ORDER BY id", (level,))]


def cmd_export(a, S):
    rows = level_jobs(S, a.level)
    acc = [r for r in rows if r["state"] == "accepted"]
    if len(acc) != len(rows) and not a.partial: sys.exit(f"{len(acc)}/{len(rows)} accepted (--partial to export anyway)")
    if a.checked and any(r["check_state"] != "pass" for r in acc): sys.exit("not every accepted job passed the checker")
    os.makedirs(a.out, exist_ok=True)
    stamps, proxy, wm, K = [], [], {}, None
    def order(r): p = json.loads(r["params"]); return p.get("layer", 10 ** 6)
    for r in sorted(acc, key=order):
        p = json.loads(r["params"]); K = p["K"]; src = os.path.join(a.results, str(r["id"]))
        shard = f"q-L{p['layer']:02d}.safetensors" if r["type"] == "quant_layer" else "q-head.safetensors"
        dst = os.path.join(a.out, shard)
        if os.path.exists(dst): os.remove(dst)
        try: os.link(os.path.join(src, shard), dst)
        except OSError: shutil.copyfile(os.path.join(src, shard), dst + ".part"); os.replace(dst + ".part", dst)
        for k in st_header(dst):
            if k != "__metadata__": wm[k] = shard
        proxy += [l for l in open(os.path.join(src, "proxy.jsonl")) if l.strip()]
        for s in json.load(open(os.path.join(src, "stamp.json"))):
            if s["stamp"] not in [x["stamp"] for x in stamps]: stamps.append(s)
    with open(os.path.join(a.out, "proxy.jsonl"), "w") as f: f.writelines(proxy)
    json.dump(stamps, open(os.path.join(a.out, "stamp.json"), "w"), indent=1)
    # as ourq.py write_index(): index over every shard, config = the template with the level's bits
    json.dump({"metadata": {}, "weight_map": dict(sorted(wm.items()))}, open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=1)
    p0 = json.loads(acc[0]["params"])
    c = read_json(os.path.join(a.data, "models", f"{p0['model']}@{p0['revision']}", "config.json"))
    c["quantization_config"].update({"bits": float(K), "head_bits": K, "calibration": {"source": "challenge calib Hessians"}})
    json.dump(c, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    print(f"exported {len(acc)} shards of {a.level} to {a.out} (ourq stamps: {sorted({s['stamp'] for s in stamps})})")


def cmd_check(a, S):
    from checker import check
    rows = [r for r in level_jobs(S, a.level) if r["state"] == "accepted" and (a.recheck or not r["check_state"])]
    for r in rows:
        p = json.loads(r["params"]); res = os.path.join(a.results, str(r["id"]))
        rep = check.check_job(r["type"], p, res, lambda rel: os.path.join(a.data, rel), rows_frac=a.rows_frac, tol=a.tol)
        with S.tx() as db:
            db.execute("UPDATE jobs SET check_state=?, check_detail=? WHERE id=?", ("pass" if rep["pass"] else "fail", json.dumps(rep), r["id"]))
            S.event(db, "check", r["id"], detail={"pass": rep["pass"]})
        print(f"{r['id']:5d} {'PASS' if rep['pass'] else 'FAIL'} {rep['summary']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    e = lambda k: os.environ.get("RXMQ_" + k) or None                          # same env as server.py (the container)
    ap.add_argument("--db", default=e("DB")); ap.add_argument("--data", default=e("DATA")); ap.add_argument("--results", default=e("RESULTS"))
    sp = ap.add_subparsers(dest="cmd", required=True)
    x = sp.add_parser("invite"); x.add_argument("--uses", type=int, default=1); x.add_argument("--note", default="")
    x = sp.add_parser("add-worker"); x.add_argument("name")
    sp.add_parser("workers")
    x = sp.add_parser("disable-worker"); x.add_argument("id", type=int); x.add_argument("--enable", action="store_true")
    x = sp.add_parser("manifest"); x.add_argument("--jobs", type=int, default=4)
    x = sp.add_parser("datapack"); x.add_argument("model"); x.add_argument("rev"); x.add_argument("ckpt"); x.add_argument("hess")
    x.add_argument("--arch", default="qwen3_5"); x.add_argument("--hout", nargs=2, metavar=("SNAP", "DIR")); x.add_argument("--no-head", action="store_true")
    x.add_argument("--layers", help="A-B: only these layers' weight files (tests)")
    x = sp.add_parser("enqueue-level")
    for k in ("model", "rev", "K", "codebook", "level"): x.add_argument(f"--{k}", required=True)
    x.add_argument("--arch", default="qwen3_5"); x.add_argument("--layers"); x.add_argument("--head", action="store_true")
    x.add_argument("--settings-json"); x.add_argument("--sigma-reg", type=float, default=0.025)
    x.add_argument("--sigma-reg-in", type=float); x.add_argument("--sigma-reg-out", type=float)
    x.add_argument("--scaled-hessian", action="store_true"); x.add_argument("--qtf2", action="store_true")
    x.add_argument("--hout-snap"); x.add_argument("--seed-offset", type=int, default=0)
    x = sp.add_parser("enqueue-noop"); x.add_argument("n", type=int); x.add_argument("--seconds", type=float, default=1)
    x.add_argument("--size", type=int, default=1024); x.add_argument("--seed", type=int, default=0)
    x.add_argument("--inputs"); x.add_argument("--tag"); x.add_argument("--fail", action="store_true")
    x = sp.add_parser("list"); x.add_argument("--state"); x.add_argument("--level"); x.add_argument("--type")
    x = sp.add_parser("events"); x.add_argument("job", nargs="?", type=int)
    for c in ("requeue", "cancel"):
        x = sp.add_parser(c); x.add_argument("ids", nargs="*"); x.add_argument("--level"); x.add_argument("--state")
        x.add_argument("--force", action="store_true")
    x = sp.add_parser("export"); x.add_argument("--level", required=True); x.add_argument("out")
    x.add_argument("--partial", action="store_true"); x.add_argument("--checked", action="store_true", help="refuse unless every job passed the checker")
    x = sp.add_parser("check"); x.add_argument("--level", required=True); x.add_argument("--rows-frac", type=float, default=1.0)
    x.add_argument("--tol", type=float, default=None); x.add_argument("--recheck", action="store_true")
    a = ap.parse_args()
    if not a.db: ap.error("--db (or RXMQ_DB) is required")
    S = Store(a.db)
    if a.cmd == "invite": print(S.add_invite(a.uses, a.note))
    elif a.cmd == "add-worker": wid, tok = S.add_worker(a.name); print(f"worker {wid} token {tok}")
    elif a.cmd == "workers":
        for r in S.q("SELECT * FROM workers ORDER BY id"):
            print(r["id"], r["name"], "disabled" if r["disabled"] else "", time.strftime("%Y-%m-%d %H:%M", time.gmtime(r["last_seen"] or 0)), r["caps"])
    elif a.cmd == "disable-worker":
        with S.tx() as db: db.execute("UPDATE workers SET disabled=? WHERE id=?", (0 if a.enable else 1, a.id))
        print(f"worker {a.id} {'enabled' if a.enable else 'disabled'}")
    elif a.cmd == "manifest": cmd_manifest(a)
    elif a.cmd == "datapack": cmd_datapack(a)
    elif a.cmd == "enqueue-level": cmd_enqueue_level(a, S)
    elif a.cmd == "enqueue-noop": cmd_enqueue_noop(a, S)
    elif a.cmd == "list": cmd_list(a, S)
    elif a.cmd == "events":
        rows = S.q("SELECT * FROM events" + (" WHERE job_id=?" if a.job else "") + " ORDER BY id", (a.job,) if a.job else ())
        for r in rows: print(f"{r['t']:.1f} job {r['job_id']} w {r['worker_id']} {r['kind']:14s} {(r['lease_id'] or '')[:8]} {r['detail'] or ''}")
    elif a.cmd == "requeue": cmd_requeue(a, S)
    elif a.cmd == "cancel": cmd_cancel(a, S)
    elif a.cmd == "export": cmd_export(a, S)
    elif a.cmd == "check": cmd_check(a, S)


if __name__ == "__main__":
    main()
