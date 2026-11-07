#!/usr/bin/env python3
"""fidelity.py RULER_CLONE HAYSTACKS TOKENIZER_DIR DEPS NLTK_DATA WORK CELL... -- run NVIDIA's own generator scripts
(read and reviewed first: scripts/data/synthetic/*.py, tokenizer.py, manifest_utils.py) and rgen.py's port on the same
cells, and compare the rows field by field. CELL = task:length:samples. The official scripts run in a child
`python3 -I` with a shim for `wonderwords` (the vendored word lists) and `tenacity` (unused by the HF tokenizer).
The clone's json/ directory must hold the haystack files (copied from HAYSTACKS here)."""
import json, os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
clone, hay, tokdir, deps, nltk_data, work = sys.argv[1:7]
cells = sys.argv[7:]

SHIM = '''
import sys, os, types, runpy
deps, nltk_data, words_dir, syn, script = sys.argv[1:6]
sys.path[:0] = [deps, syn, os.path.dirname(syn)]
os.environ["NLTK_DATA"] = nltk_data
import nltk; nltk.data.path.insert(0, nltk_data)
ww = types.ModuleType("wonderwords"); rw = types.ModuleType("wonderwords.random_word")
def _get(name):
    with open(os.path.join(words_dir, name), encoding="utf-8") as f:
        return [w.rstrip() for w in f.readlines()]
rw._get_words_from_text_file = _get; ww.random_word = rw
sys.modules["wonderwords"] = ww; sys.modules["wonderwords.random_word"] = rw
ten = types.ModuleType("tenacity")
class _W:
    def __init__(self, *a, **k): pass
    def __add__(self, o): return self
ten.retry = lambda *a, **k: (lambda f: f)
ten.stop_after_attempt = ten.wait_fixed = ten.wait_random = _W
sys.modules["tenacity"] = ten
sys.argv = [script] + sys.argv[6:]
runpy.run_path(script, run_name="__main__")
'''

import rgen
syn = os.path.join(clone, 'scripts', 'data', 'synthetic')
for f in ('PaulGrahamEssays.json', 'english_words.json', 'squad.json'):
    if os.path.exists(os.path.join(hay, f)):
        shutil.copyfile(os.path.join(hay, f), os.path.join(syn, 'json', f))
shim = os.path.join(work, 'shim.py')
open(shim, 'w').write(SHIM)
if deps:
    sys.path.insert(0, deps)
os.environ['NLTK_DATA'] = nltk_data
import nltk
nltk.data.path.insert(0, nltk_data)
tok = rgen.Tok(os.path.join(tokdir, 'tokenizer.json'))
yaml_name = {'qa_squad': 'qa_1', 'qa_hotpot': 'qa_2'}
bad = 0
for c in cells:
    task, L, n = c.split(':'); L, n = int(L), int(n)
    kind, args = rgen.TASKS[task]
    out = tempfile.mkdtemp(dir=work)
    cmd = [sys.executable, '-I', shim, deps, nltk_data, rgen.WORDS_DIR, syn, os.path.join(syn, kind + '.py'),
           '--save_dir', out, '--save_name', task, '--subset', 'validation', '--tokenizer_path', tokdir,
           '--tokenizer_type', 'hf', '--max_seq_length', str(L),
           '--tokens_to_generate', str(rgen.TEMPLATES[kind]['tokens_to_generate']), '--num_samples', str(n),
           '--random_seed', '42']
    for k, v in args.items():
        cmd += [f'--{k}', str(v)]
    if kind == 'qa':
        cmd += ['--pre_samples', '0']
    cmd += ['--template', rgen.template_of(kind)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(f"{c}: official generator failed\n{r.stderr[-2000:]}")
        bad += 1
        continue
    off = [json.loads(l) for l in open(os.path.join(out, task, 'validation.jsonl'))]
    port = rgen.generate(task, L, n, 42, tok, hay, log=lambda s: None)
    diffs = []
    for i, (a, b) in enumerate(zip(off, port)):
        for k in ('input', 'outputs', 'answer_prefix', 'length'):
            if a.get(k) != b.get(k):
                diffs.append(f"row {i} {k}")
    if len(off) != len(port):
        diffs.append(f"rows {len(off)} vs {len(port)}")
    bad += bool(diffs)
    print(f"{c}: {'IDENTICAL' if not diffs else 'DIFF ' + ', '.join(diffs[:6])} ({len(off)} rows, "
          f"prompt tokens {min(x['length'] for x in off)}-{max(x['length'] for x in off)} incl. generation)", flush=True)
    shutil.rmtree(out, ignore_errors=True)
print(f"{len(cells)} cells, {bad} differ")
sys.exit(1 if bad else 0)
