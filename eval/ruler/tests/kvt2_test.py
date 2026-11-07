#!/usr/bin/env python3
"""kvt2_test.py AR_TREE SETDIR -- exercise the proposed kv_judge T2 path (patches/autoresearch-v3-kv-t2.patch applied
in AR_TREE, e.g. a `git archive kv-task` copy) without a card: kvjudge.Runner.t2_rlib with the hold / RAM / argv
helpers stubbed and a fake OpenAI-style server that answers each RULER prompt with its reference (anchors) or, for
the candidate (--kv-cache-dtype k6_v4 in its argv), drops every other niah answer. Expect: a correct candidate
passes, the broken one is rejected by the paired gate."""
import json, os, subprocess, sys, tempfile, time
from pathlib import Path

ar, setdir = sys.argv[1:3]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(ar, 'tasks', 'kv'), os.path.join(ar, 'tasks', 'kernel')]
import kvjudge  # noqa: E402

FAKE = r'''
import json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.path.insert(0, sys.argv[1]); import rlib
setdir, sel, broken = sys.argv[2], json.loads(sys.argv[3]), sys.argv[4] == '1'
port = int(sys.argv[sys.argv.index('--port') + 1])
ans = {}
for i, s in enumerate(rlib.select(setdir, sel)):
    wrong = broken and s['task'].startswith('niah') and i % 2 == 0
    ans[rlib.render(s, 'chat')] = ' nothing' if wrong else ' ' + ', '.join(s['outputs'])
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _j(self, o):
        b = json.dumps(o).encode(); self.send_response(200); self.send_header('Content-Length', str(len(b)))
        self.end_headers(); self.wfile.write(b)
    def do_GET(self): self._j({'status': 'ok'})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self._j({'choices': [{'text': ans.get(body['prompt'], ' ?')}], 'usage': {}})
ThreadingHTTPServer(('127.0.0.1', port), H).serve_forever()
'''


def runner(tmp, sel, broken_cand):
    r = kvjudge.Runner.__new__(kvjudge.Runner)
    fake = Path(tmp) / 'fake.py'
    fake.write_text(FAKE)
    r.model = {'ruler': {'set': setdir, 'select': sel, 'template': 'chat', 'anchors': ['fp8'], 'tol_points': 0.0,
                         'tol_length_points': 10.0}, 'anchors': [{'name': 'fp8', 'setup': {'kv_cache_dtype': 'fp8'}}],
               'max_depth_fit': 1 << 30, 'load_timeout_s': 30}
    r.setup = {'kv_cache_dtype': 'k6_v4'}
    r.root, r.pid, r.bid = Path(tmp), 'p', 'b'
    r.jd = r.run = Path(tmp)
    r.host = {'mem_kill_gb': 0}
    r.holds = []
    r.log = print

    def argv(prod, setup, extra, rdir):
        b = '1' if (broken_cand and setup.get('kv_cache_dtype') == 'k6_v4') else '0'
        return [sys.executable, str(fake), HERE + '/..', setdir, json.dumps(sel['cells']), b] + extra
    r.engine_argv = argv
    r.env = lambda prod: dict(os.environ)
    r.mem_ok = lambda what, engine=False: None
    r.hold = lambda what, fn: fn(time.time() + 120)
    r.retry_mem = lambda what, fn: fn()
    return r


sel = {'name': 'test', 'cells': {'niah_single_1_4096': 4, 'niah_multikey_2_4096': 4, 'vt_4096': 4, 'qa_squad_4096': 4}}
ok = True
for broken in (False, True):
    with tempfile.TemporaryDirectory() as tmp:
        r = runner(tmp, sel, broken)
        out = r.t2_rlib(Path(tmp), Path(tmp), {'key': 'k'})
        print(f"broken={broken}: ok={out['ok']} cand {out['reports']['cand']['overall']:.1f} "
              f"fp8 {out['reports']['fp8']['overall']:.1f} failed={out.get('failed')}")
        ok &= out['ok'] == (not broken)
print('PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
