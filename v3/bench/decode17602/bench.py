#!/usr/bin/env python3
"""bench.py -- #176 end-to-end decode client (w138): what a user of the OpenAI chat API sees, batch 1.

  bench.py --port P --prompts DIR --out FILE.jsonl --tag TAG [--ctx 0,4k,16k,32k] [--warm 2] [--runs 5] [--gen 512]
           [--server-log LOG] [--meta JSON]

Per context: warm + runs streamed /v1/chat/completions requests, one at a time (greedy: temperature 0; ignore_eos so
every request generates exactly --gen tokens). Each request starts with a fresh nonce line, so no prompt prefix is
ever served from the prefix cache. Per request it records:
  client: TTFT (send -> first content chunk), first -> last chunk wall, ms per token = that / (tokens - 1), tok/s;
  server: /metrics deltas around the request: engine decode steps, seconds inside decode steps (ms per decode step =
          engine_step_seconds_decode / engine_steps_decode), decode tokens per decode step (> 1 with speculation),
          drafted / accepted tokens, the dashboard's decode-step gauges (best / recent);
          the request's own 'I req ... done' line from the server log (ttft, tok/s, seconds) when --server-log is given.
One JSON line per request (warmups marked). w159 (#176-02) added the median / p25 inter-chunk gap (client_med_gap_ms,
client_p25_gap_ms; first gap skipped): robust to host-contention spikes that inflate a run's mean. Only the standard library: run with python3 -I.
Prompt files: DIR/ctx-<name>.txt (filler cut to the context's token count by mkprompts.py), DIR/ctx-0.txt short.
"""
import argparse, json, os, re, time, uuid, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--prompts", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--ctx", default="0,4k,16k,32k")
ap.add_argument("--warm", type=int, default=2)
ap.add_argument("--runs", type=int, default=5)
ap.add_argument("--gen", type=int, default=512)
ap.add_argument("--server-log")
ap.add_argument("--meta", default="{}")
a = ap.parse_args()
U = f"http://127.0.0.1:{a.port}"
META = json.loads(a.meta)

MKEYS = ["radiance:engine_steps_decode_total", "radiance:engine_step_seconds_decode_total",
         "radiance:engine_steps_total", "radiance:engine_step_seconds_total", "radiance:decode_tokens_total",
         "radiance:prefill_tokens_total", "radiance:engine_steps_mixed_total", "radiance:engine_steps_prefill_total",
         "radiance:draft_tokens_total", "radiance:draft_accepted_total", "radiance:attn_cached_tokens_total",
         "radiance:decode_step_seconds_best", "radiance:decode_step_seconds_recent"]


def metrics():
    txt = urllib.request.urlopen(U + "/metrics", timeout=10).read().decode()
    out = {}
    for line in txt.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name = line.split("{")[0].split()[0]
        if name in MKEYS and name not in out:
            out[name] = float(line.rsplit(None, 1)[-1])
    return out


def log_size():
    return os.path.getsize(a.server_log) if a.server_log else 0


def log_done_line(off):
    if not a.server_log:
        return None
    with open(a.server_log, "rb") as f:
        f.seek(off)
        new = f.read().decode(errors="replace")
    for line in new.splitlines():
        m = re.search(r"done .*prompt=(\d+) \(cache (\d+)\) gen=(\d+)(?: reason=(\d+))?.*? ttft=([\d.]+)s ([\d.]+) tok/s ([\d.]+)s", line)
        if m:
            return {"prompt": int(m.group(1)), "cache": int(m.group(2)), "gen": int(m.group(3)),
                    "reason": int(m.group(4)) if m.group(4) else 0,
                    "ttft_s": float(m.group(5)), "tok_s": float(m.group(6)), "total_s": float(m.group(7))}
    return None


def one(ctxname, text, i, warm):
    nonce = f"[session {uuid.uuid4().hex}]\n"
    body = {"model": "q", "messages": [{"role": "user", "content": nonce + text}], "max_tokens": a.gen,
            "temperature": 0, "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    m0, off = metrics(), log_size()
    req = urllib.request.Request(U + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    times, usage, text_out, nchunks = [], None, [], 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]":
                break
            j = json.loads(d)
            if j.get("usage"):
                usage = j["usage"]
            for ch in j.get("choices", []):
                dl = ch.get("delta", {})
                piece = (dl.get("content") or "") + (dl.get("reasoning_content") or dl.get("reasoning") or "")
                if piece:
                    times.append(time.perf_counter())
                    nchunks += 1
                    if len(text_out) < 400:
                        text_out.append(piece)
    t_end = time.perf_counter()
    time.sleep(0.3)  # the engine publishes its metrics copy after the step
    m1 = metrics()
    d = {k: m1.get(k, 0) - m0.get(k, 0) for k in MKEYS if not k.startswith("radiance:decode_step_seconds")}
    ntok = (usage or {}).get("completion_tokens", nchunks)
    rec = {"tag": a.tag, "ctx": ctxname, "run": i, "warmup": warm, "utc": time.strftime("%FT%TZ", time.gmtime()),
           "prompt_tokens": (usage or {}).get("prompt_tokens"), "completion_tokens": ntok, "chunks": nchunks,
           "client_ttft_ms": 1e3 * (times[0] - t0) if times else None,
           "client_total_s": t_end - t0,
           "client_ms_per_token": 1e3 * (times[-1] - times[0]) / (ntok - 1) if len(times) > 1 and ntok > 1 else None,
           "client_ms_per_chunk": 1e3 * (times[-1] - times[0]) / (nchunks - 1) if nchunks > 1 else None}
    rec["client_tok_s"] = 1e3 / rec["client_ms_per_token"] if rec["client_ms_per_token"] else None
    gaps = sorted(1e3 * (b - x) for x, b in zip(times[1:], times[2:]))   # w159: robust to host-contention spikes
    rec["client_med_gap_ms"] = gaps[len(gaps) // 2] if gaps else None
    rec["client_p25_gap_ms"] = gaps[len(gaps) // 4] if gaps else None
    sd, ss = d["radiance:engine_steps_decode_total"], d["radiance:engine_step_seconds_decode_total"]
    rec["srv_decode_steps"] = sd
    rec["srv_ms_per_decode_step"] = 1e3 * ss / sd if sd else None
    rec["srv_decode_tokens"] = d["radiance:decode_tokens_total"]
    rec["srv_tokens_per_decode_step"] = d["radiance:decode_tokens_total"] / sd if sd else None
    rec["srv_ms_per_token_in_steps"] = 1e3 * ss / d["radiance:decode_tokens_total"] if d["radiance:decode_tokens_total"] else None
    rec["srv_mixed_steps"] = d["radiance:engine_steps_mixed_total"]
    rec["srv_prefill_tokens"] = d["radiance:prefill_tokens_total"]
    rec["srv_cached_tokens"] = d["radiance:attn_cached_tokens_total"]
    rec["srv_draft_tokens"] = d["radiance:draft_tokens_total"]
    rec["srv_draft_accepted"] = d["radiance:draft_accepted_total"]
    rec["gauge_step_best_ms"] = 1e3 * m1["radiance:decode_step_seconds_best"] if "radiance:decode_step_seconds_best" in m1 else None
    rec["gauge_step_recent_ms"] = 1e3 * m1["radiance:decode_step_seconds_recent"] if "radiance:decode_step_seconds_recent" in m1 else None
    rec["srv_log"] = log_done_line(off)
    rec["text_head"] = "".join(text_out)[:200]
    rec.update(META)
    return rec


with open(a.out, "a") as f:
    for c in a.ctx.split(","):
        text = open(os.path.join(a.prompts, f"ctx-{c}.txt")).read()
        for i in range(a.warm + a.runs):
            r = one(c, text, i, i < a.warm)
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"{a.tag} ctx {c} run {i}{' (warm)' if r['warmup'] else ''}: prompt {r['prompt_tokens']} "
                  f"ttft {r['client_ttft_ms']:.0f} ms, client {r['client_ms_per_token']:.2f} ms/tok, "
                  f"server {r['srv_ms_per_decode_step'] or 0:.2f} ms/step x {r['srv_tokens_per_decode_step'] or 0:.2f} tok",
                  flush=True)
