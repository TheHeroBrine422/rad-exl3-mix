#!/bin/bash
# killtest.sh SETDIR TOKENIZER.json WORK -- preemption test of rrun.py (CPU only): a fake /v1/completions server
# (deterministic reply from the prompt ids, 0.15 s a request); one clean run, then a run SIGKILLed at random moments
# 5 times and resumed each time; the finished records must equal the clean run's (reply, score, key) one for one,
# and no .part file may be trusted (none left as a final name).
set -u
X="$(cd "$(dirname "$0")/.." && pwd)"; SET=$1 TOK=$2 W=$3
mkdir -p "$W"; rm -rf "$W"/clean "$W"/killed
cat > "$W/fake.py" <<'EOF'
import hashlib, json, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        if self.path != '/v1/completions':
            self.send_error(404); return
        b = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        time.sleep(0.15)
        h = hashlib.sha256(json.dumps(b['prompt']).encode()).hexdigest()
        o = json.dumps({'choices': [{'text': ' ' + h[:12], 'finish_reason': 'stop'}], 'usage': {}}).encode()
        self.send_response(200); self.send_header('Content-Length', str(len(o))); self.end_headers(); self.wfile.write(o)
ThreadingHTTPServer(('127.0.0.1', int(sys.argv[1])), H).serve_forever()
EOF
PORT=$(( 20000 + RANDOM % 20000 ))
python3 -I "$W/fake.py" $PORT & FP=$!
echo "{\"name\": \"fake\", \"backend\": \"openai\", \"url\": \"http://127.0.0.1:$PORT\"}" > "$W/cfg.json"
sleep 1
CELLS=niah_single_1:4096:25,vt:4096:25,cwe:8192:25
python3 -I "$X/rrun.py" --set "$SET" --cells $CELLS --config "$W/cfg.json" --out "$W/clean" --tokenizer "$TOK" > "$W/clean.log" 2>&1
for k in 1 2 3 4 5; do
  python3 -I "$X/rrun.py" --set "$SET" --cells $CELLS --config "$W/cfg.json" --out "$W/killed" --tokenizer "$TOK" >> "$W/killed.log" 2>&1 &
  P=$!; sleep "$(python3 -c "import random; print(round(random.uniform(2.2, 4.0), 2))")"; kill -9 $P 2>/dev/null; wait $P 2>/dev/null
  echo "kill $k: $(find "$W/killed/pred" -name '*.json' | wc -l) done, $(find "$W/killed" -name '*.part' | wc -l) .part"
done
python3 -I "$X/rrun.py" --set "$SET" --cells $CELLS --config "$W/cfg.json" --out "$W/killed" --tokenizer "$TOK" >> "$W/killed.log" 2>&1
echo "final rc=$?"
kill $FP
python3 -I - "$W" <<'EOF'
import json, os, sys
w = sys.argv[1]
def recs(d):
    out = {}
    for c in sorted(os.listdir(f'{w}/{d}/pred')):
        for f in sorted(os.listdir(f'{w}/{d}/pred/{c}')):
            if f.endswith('.json'):
                r = json.load(open(f'{w}/{d}/pred/{c}/{f}'))
                out[r['key']] = (r['pred'], r['score'], r['prompt_tokens'], r['cell_sha256'])
    return out
a, b = recs('clean'), recs('killed')
print(f'clean {len(a)}, killed+resumed {len(b)}, identical: {a == b}')
sys.exit(0 if a == b and len(a) == 75 else 1)
EOF
