#!/usr/bin/env python3
"""lsa_hash.py SETDIR HASHES.json [CELL...] -- cross-check a set against the LSA repo's pinned prompt-set hashes for
Qwen3.8-27B (src/lsa/evaluation/accuracy/data/hashes_qwen38_27b.json: key ruler_<task>_<length>:first<N> = the first
16 hex of sha256 over the concatenated 16-hex sha256 of each prompt). Their prompt is lm-eval's context for a row:
the input (lm-eval strips its end), a space, the answer prefix stripped (their `gen_prefix`). A match means the rows are the ones their
harness serves (same generator, seed 42, tokenizer, haystacks, word lists, punkt)."""
import hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rlib

setdir, hashes = sys.argv[1:3]
want = json.load(open(hashes))
m = rlib.manifest(setdir)
cells = sys.argv[3:] or sorted(m['cells'])
variants = {
    'input+" "+gen_prefix': lambda r: r['input'] + ' ' + r['answer_prefix'].strip(),
    'input+answer_prefix': lambda r: r['input'] + r['answer_prefix'],
    'input+"\\n"+gen_prefix': lambda r: r['input'] + '\n' + r['answer_prefix'].strip(),
    'input.rstrip()+" "+gen_prefix': lambda r: r['input'].rstrip() + ' ' + r['answer_prefix'].strip(),
}
alias = {'qa_squad': 'qa_squad', 'qa_hotpot': 'qa_hotpot'}
hit = miss = 0
for cell in cells:
    rows = rlib.cell_rows(setdir, cell)
    for key, h in want.items():
        if not key.startswith(f'ruler_{cell}:first'):
            continue
        n = int(key.rsplit('first', 1)[1])
        if n > len(rows):
            print(f'{key}: set has {len(rows)} rows, skipped')
            continue
        got = {name: hashlib.sha256(''.join(hashlib.sha256(f(r).encode()).hexdigest()[:16] for r in rows[:n])
                                    .encode()).hexdigest()[:16] for name, f in variants.items()}
        match = [name for name, g in got.items() if g == h]
        hit += bool(match)
        miss += not match
        print(f'{key}: {"MATCH (" + match[0] + ")" if match else "no match"}')
print(f'{hit} match, {miss} differ')
