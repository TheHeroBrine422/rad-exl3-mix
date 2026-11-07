#!/usr/bin/env python3
"""rrun.py -- run a RULER selection against one model config, greedy, one result file per sample (resumable).

    python3 -I rrun.py --set SETDIR (--select select/fast-v1.json | --cells task:L:a-b,...) --config CFG.json
        --out RUNDIR [--template chat|base] [--tokenizer tokenizer.json] [--budget-s S] [--limit N]

CFG.json (a candidate = one file; see configs/ for examples):
  {"name": "v2-400-bf16", "backend": "radiance",          # start a radiance server of its own (loopback)
   "radiance": "/data/install-116/bin/radiance", "home": "/data/swap27/home", "model": ".../x.rad",
   "kv_cache_dtype": "bf16", "flags": ["--rope-scaling", "yarn:4:262144", ...], "env": {"ROCR_VISIBLE_DEVICES": "0"},
   "max_model_len": 0 (0 = the selection's longest cell + 256), "load_timeout_s": 600,
   "mem_start_gb": 7.0, "mem_kill_gb": 3.5}                # host MemAvailable gates (shared boxes)
  {"name": ..., "backend": "openai", "url": "http://127.0.0.1:PORT"}           # a server someone else runs
  {"name": ..., "backend": "hf", "model": "/data/models/qwen35-08b", "dtype": "bfloat16", "vram_cap_gb": 7.2,
   "prefill_chunk": 0}                                    # > 0: prefill long prompts in chunks (VRAM)
Every key but "name", "notes", "url", "port", "load_timeout_s", "mem_*" is part of the run's identity: a RUNDIR made
with another config, set (name / seed / format / generator), template or tokenizer is refused (start a new RUNDIR),
and so is a RUNDIR whose finished rows came from other content of their cell (the cell's sha256 is in each record).

RUNDIR: run.json (identity), pred/<task>_<L>/<index>.json (one sample: reply, score, timings; written .part ->
rename), server-*.log. A killed run resumes where it stopped. Exit 0 = every selected sample done, 3 = the
--budget-s ran out first (rerun to continue), 4 = killed by the host-memory guard (rerun later), 1 = error.
Score and compare with rreport.py.
"""
import argparse, json, os, signal, socket, subprocess, sys, threading, time, urllib.error, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rlib

VERSION = 1
VOLATILE = ('name', 'url', 'port', 'load_timeout_s', 'mem_start_gb', 'mem_kill_gb', 'notes')


def mem_available_gb():
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 1048576
    return float('inf')


class MemKilled(Exception):
    pass


# ---------------------------------------------------------------- backends
class OpenAI:
    """An OpenAI-style /v1/completions server; greedy (temperature 0), token-id prompts."""
    def __init__(self, url):
        self.url = url.rstrip('/')

    def post(self, path, body, timeout):
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def complete(self, ids, max_tokens, timeout):
        body = {'prompt': ids, 'max_tokens': max_tokens, 'temperature': 0, 'top_k': 1, 'top_p': 1.0, 'seed': 0,
                'stream': False}
        r = self.post('/v1/completions', body, timeout)
        c = (r.get('choices') or [{}])[0]
        u = r.get('usage') or {}
        return {'text': c.get('text', ''), 'finish_reason': c.get('finish_reason'),
                'prompt_tokens': u.get('prompt_tokens'), 'completion_tokens': u.get('completion_tokens')}

    def alive(self):
        return True

    def close(self):
        pass


class Radiance(OpenAI):
    """A radiance server of our own: started in its own process group, watched for host memory, stopped at exit."""
    def __init__(self, cfg, max_model_len, logpath):
        self.cfg = cfg
        port = cfg.get('port') or free_port()
        super().__init__(f'http://127.0.0.1:{port}')
        argv = [cfg['radiance'], '--model', cfg['model'], '--tp', str(cfg.get('tp', 1)),
                '--kv-cache-dtype', cfg.get('kv_cache_dtype', 'bf16'), '--max-model-len', str(max_model_len),
                '--max-num-seqs', '1', '--host', '127.0.0.1', '--port', str(port)] + list(cfg.get('flags', []))
        env = dict(os.environ)
        if cfg.get('home'):
            env['RADIANCE_HOME'] = cfg['home']
        env.update({k: str(v) for k, v in cfg.get('env', {}).items()})
        self.kill_gb = float(cfg.get('mem_kill_gb', 3.5))
        start_gb = float(cfg.get('mem_start_gb', 7.0))
        a = mem_available_gb()
        if a < start_gb:
            raise MemKilled(f'MemAvailable {a:.2f} GB < mem_start_gb {start_gb}: not starting the engine')
        self.log = open(logpath, 'ab')
        self.log.write(('$ ' + ' '.join(argv) + '\n').encode())
        self.log.flush()
        self.p = subprocess.Popen(argv, stdout=self.log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
                                  start_new_session=True)
        self.mem_killed = None
        self.mem_min = None
        self.mem_min_t = 0.0
        threading.Thread(target=self._watch, daemon=True).start()
        deadline = time.time() + float(cfg.get('load_timeout_s', 600))
        while time.time() < deadline:
            self._check()
            try:
                with urllib.request.urlopen(self.url + '/health', timeout=5) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(1.0)
        self.close()
        raise RuntimeError('radiance did not come up before load_timeout_s')

    def _watch(self):
        t0 = time.time()
        while self.p is not None and self.p.poll() is None:
            a = mem_available_gb()
            if self.mem_min is None or a < self.mem_min:
                self.mem_min, self.mem_min_t = a, time.time() - t0
            if a < self.kill_gb:
                self.mem_killed = a
                try:
                    os.killpg(self.p.pid, signal.SIGKILL)
                except OSError:
                    pass
                return
            time.sleep(0.2)

    def _check(self):
        if self.mem_killed is not None:
            raise MemKilled(f'MemAvailable {self.mem_killed:.2f} GB < {self.kill_gb} GB: engine killed')
        if self.p.poll() is not None:
            raise RuntimeError(f'radiance exited (rc {self.p.returncode}); see {self.log.name}')

    def complete(self, ids, max_tokens, timeout):
        try:
            out = super().complete(ids, max_tokens, timeout)
        except (urllib.error.URLError, OSError, ConnectionError):
            self._check()
            raise
        self._check()
        return out

    def close(self):
        if self.p is not None and self.p.poll() is None:
            try:
                os.killpg(self.p.pid, signal.SIGTERM)
                self.p.wait(timeout=30)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self.p.pid, signal.SIGKILL)
                except OSError:
                    pass
                self.p.wait()
        self.p = None


class HF:
    """transformers on CUDA (the 0.8B proxy on the 4070: radiance has no CUDA path). Greedy generate()."""
    def __init__(self, cfg, tok):
        import torch
        from transformers import AutoModelForCausalLM
        self.torch = torch
        dev = cfg.get('device', 'cuda')
        caps = [float(x) for x in (cfg.get('vram_cap_gb'), os.environ.get('RULER_VRAM_CAP_GB')) if x]
        if dev.startswith('cuda') and caps:       # the allocator's ceiling: an overrun fails here, not in a neighbour
            total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
            torch.cuda.set_per_process_memory_fraction(min(1.0, min(caps) / total))
        kw = {'dtype': getattr(torch, cfg.get('dtype', 'bfloat16'))}
        if cfg.get('attn'):
            kw['attn_implementation'] = cfg['attn']
        self.m = AutoModelForCausalLM.from_pretrained(cfg['model'], **kw).to(dev).eval()
        self.dev = dev
        self.tok = tok
        self.eos = [int(x) for x in cfg.get('eos', [248046, 248044])]
        self.chunk = int(cfg.get('prefill_chunk', 0))     # > 0: prefill prompts longer than this in chunks

    def complete(self, ids, max_tokens, timeout):
        torch = self.torch
        x = torch.tensor([ids], device=self.dev)
        kw = {}
        with torch.no_grad():
            if self.chunk and len(ids) > self.chunk:
                # all but the last token through the model in chunks (the cache holds the attention keys and the
                # delta-net states); generate() then runs the last token and decodes
                cache = None
                for i in range(0, len(ids) - 1, self.chunk):
                    out = self.m(input_ids=x[:, i:min(i + self.chunk, len(ids) - 1)], past_key_values=cache,
                                 use_cache=True, logits_to_keep=1)
                    cache = out.past_key_values
                    del out
                kw['past_key_values'] = cache
            y = self.m.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=max_tokens, do_sample=False,
                                temperature=None, top_p=None, top_k=None, eos_token_id=self.eos,
                                pad_token_id=self.eos[0], **kw)
        new = y[0, len(ids):].tolist()
        stop = bool(new) and new[-1] in self.eos
        text = self.tok.decode([t for t in new if t not in self.eos], skip_special_tokens=True)
        peak = torch.cuda.max_memory_allocated() / 2 ** 30 if self.dev.startswith('cuda') else None
        return {'text': text, 'finish_reason': 'stop' if stop else 'length', 'prompt_tokens': len(ids),
                'completion_tokens': len(new), 'completion_ids': new, 'vram_peak_gb': peak}

    def alive(self):
        return True

    def close(self):
        pass


def free_port():
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    p = s.getsockname()[1]
    s.close()
    return p


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--set', required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--select')
    g.add_argument('--cells')
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--template', default='chat', choices=['chat', 'base'])
    ap.add_argument('--tokenizer', default=os.environ.get('RULER_TOKENIZER'))
    ap.add_argument('--budget-s', type=float, default=0, help='stop before a sample that may not finish in time')
    ap.add_argument('--limit', type=int, default=0, help='at most N new samples this invocation')
    a = ap.parse_args()

    from tokenizers import Tokenizer
    if not a.tokenizer:
        sys.exit('--tokenizer (or RULER_TOKENIZER): the Qwen3.8 tokenizer.json')
    tok = Tokenizer.from_file(a.tokenizer)
    cfg = json.load(open(a.config))
    if a.select:
        sel = rlib.load_selection(a.select)
        sel_name, cells = sel['name'], sel['cells']
    else:
        sel_name, cells = 'cells:' + a.cells, rlib.parse_cells(a.cells)
    samples = rlib.select(a.set, cells)
    ident = {'version': VERSION, 'set': rlib.manifest_id(a.set), 'template': a.template,
             'tokenizer_sha256': rlib.sha256_file(a.tokenizer),
             'config': {k: v for k, v in cfg.items() if k not in VOLATILE}}
    os.makedirs(os.path.join(a.out, 'pred'), exist_ok=True)
    rj = os.path.join(a.out, 'run.json')
    if os.path.exists(rj):
        old = json.load(open(rj))
        if old['identity'] != ident:
            sys.exit(f'{rj}: made with another identity (config / set / template / tokenizer); use a new --out\n'
                     f'old {json.dumps(old["identity"], sort_keys=True)}\nnew {json.dumps(ident, sort_keys=True)}')
        old.setdefault('selections', [])
        if sel_name not in old['selections']:
            old['selections'].append(sel_name)
            rlib.write_json(rj, old)
    else:
        rlib.write_json(rj, {'identity': ident, 'name': cfg.get('name'), 'selections': [sel_name],
                             'created': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})

    def path_of(s):
        return os.path.join(a.out, 'pred', s['cell'], f"{s['index']:04d}.json")
    stale = [r['key'] for r in rlib.run_results(a.out) if r['cell'] in cells and
             r.get('cell_sha256') != rlib.cell_sha256(a.set, r['cell'])]
    if stale:
        sys.exit(f'{len(stale)} finished samples were run on other rows of their cell ({stale[:3]}...): '
                 'the set changed under this RUNDIR; use a new --out')
    todo = [s for s in samples if not os.path.exists(path_of(s))]
    print(f'{len(samples)} selected, {len(samples) - len(todo)} done, {len(todo)} to run', flush=True)
    if not todo:
        return 0
    t_start = time.time()
    deadline = t_start + a.budget_s if a.budget_s else float('inf')
    enc = {}
    for s in todo:
        enc[s['key']] = tok.encode(rlib.render(s, a.template), add_special_tokens=False).ids
    need = max(len(enc[s['key']]) + s['tokens_to_generate'] for s in todo)
    be = cfg['backend']
    if be == 'radiance':
        mml = int(cfg.get('max_model_len') or 0) or max(len(enc[s['key']]) + s['tokens_to_generate'] for s in todo) + 256
        if mml < need:
            sys.exit(f'max_model_len {mml} < {need} needed')
        try:
            eng = Radiance(cfg, mml, os.path.join(a.out, f'server-{time.strftime("%Y%m%dT%H%M%S")}.log'))
        except MemKilled as e:
            print(f'memory: {e}', flush=True)
            return 4
    elif be == 'openai':
        eng = OpenAI(cfg['url'])
    elif be == 'hf':
        eng = HF(cfg, tok)
    else:
        sys.exit(f'unknown backend {be}')
    load_s = time.time() - t_start
    print(f'engine up in {load_s:.0f} s', flush=True)
    if be in ('radiance', 'openai'):
        # does the server's own tokenization of the rendered text give our ids? (a text-prompt client such as
        # kv_judge's T2 relies on it); recorded once per run directory
        s0 = todo[0]
        try:
            r = eng.post('/tokenize', {'content': rlib.render(s0, a.template)}, 120)
            got = [int(x) for x in (r.get('tokens') or r.get('ids') or [])]
            same = got == enc[s0['key']]
            print(f"server /tokenize of {s0['key']}: {'identical to' if same else 'DIFFERS from'} our ids "
                  f"({len(got)} vs {len(enc[s0['key']])})", flush=True)
            rlib.write_json(os.path.join(a.out, 'tokenize-check.json'),
                            {'key': s0['key'], 'identical': same, 'server': len(got), 'ours': len(enc[s0['key']])})
        except Exception as e:
            print(f'server /tokenize check failed: {e}', flush=True)
    # seconds per prompt token (prefill-dominated), the worst seen, starting from this run's earlier samples
    prev = [r['seconds'] / max(1, r['prompt_tokens']) for r in rlib.run_results(a.out) if r.get('seconds')]
    rate = max(prev) if prev else None
    done = 0
    rc = 0
    try:
        for s in todo:
            ids = enc[s['key']]
            if rate is not None and time.time() + 1.3 * rate * len(ids) + 5 > deadline:
                print(f'budget: stopping before {s["key"]} ({len(ids)} tokens)', flush=True)
                rc = 3
                break
            if a.limit and done >= a.limit:
                rc = 3
                break
            t0 = time.time()
            out = eng.complete(ids, s['tokens_to_generate'], timeout=max(60.0, min(3600.0, deadline - t0)))
            dt = time.time() - t0
            rate = max(rate or 0.0, dt / max(1, len(ids)))
            rec = {'key': s['key'], 'task': s['task'], 'length': s['max_length'], 'index': s['index'],
                   'cell': s['cell'], 'cell_sha256': s['cell_sha256'], 'prompt_tokens': len(ids), 'pred': out['text'],
                   'finish_reason': out.get('finish_reason'), 'completion_tokens': out.get('completion_tokens'),
                   'outputs': s['outputs'], 'score': rlib.score(s, out['text']), 'seconds': round(dt, 3),
                   'config': cfg.get('name')}
            if out.get('prompt_tokens') not in (None, len(ids)):
                rec['server_prompt_tokens'] = out['prompt_tokens']
            for k in ('completion_ids', 'vram_peak_gb'):
                if out.get(k) is not None:
                    rec[k] = out[k]
            os.makedirs(os.path.dirname(path_of(s)), exist_ok=True)
            rlib.write_json(path_of(s), rec)
            done += 1
            print(f"{s['key']} {len(ids)} tok {dt:.1f}s score {rec['score']:.2f} | {out['text'][:70]!r}", flush=True)
            if time.time() > deadline:
                rc = 3
                break
    except MemKilled as e:
        print(f'memory: {e}', flush=True)
        rc = 4
    finally:
        eng.close()
        if be == 'radiance' and getattr(eng, 'mem_min', None) is not None:
            print(f'host MemAvailable minimum while the engine ran: {eng.mem_min:.2f} GB, '
                  f'{eng.mem_min_t:.1f} s after its start', flush=True)
    left = sum(1 for s in samples if not os.path.exists(path_of(s)))
    print(f'{done} samples this run, {left} left, {time.time() - t_start:.0f} s', flush=True)
    return rc if left else 0


if __name__ == '__main__':
    sys.exit(main())
