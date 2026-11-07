#!/usr/bin/env python3
"""mksel.py -- write the versioned selection files of the ruler-qwen38-v1 set (rerun after editing; never change a
released selection's content: make a -v2 instead)."""
import json, os
D = os.path.dirname(os.path.abspath(__file__))
CORE = ['niah_single_1', 'niah_single_2', 'niah_single_3', 'niah_multikey_1', 'niah_multikey_2', 'niah_multikey_3',
        'niah_multivalue', 'niah_multiquery', 'vt', 'cwe', 'fwe', 'qa_squad']
LONG = ['niah_single_2', 'niah_multikey_1', 'niah_multikey_2', 'niah_multikey_3', 'niah_multivalue', 'vt', 'cwe']
K = lambda L: L // 1024


def w(name, about, cells):
    json.dump({'name': name, 'set': 'ruler-qwen38-v1', 'about': about, 'cells': cells},
              open(os.path.join(D, name + '.json'), 'w'), indent=1)
    print(name, sum(len(range(v)) if isinstance(v, int) else len(v) for v in cells.values()), 'samples')


w('all-v1', 'every row of the set', {**{f'{t}_{L}': 100 for t in CORE for L in (4096, 8192, 16384, 32768, 65536, 131072)},
                                      **{f'{t}_{L}': 20 for t in LONG for L in (262144, 524288, 786432)}})
w('full-v1', 'final checks: 12 tasks, 20 rows a cell at 4k-64k, 10 at 128k (~11 h of a 27B on one R9700)',
  {**{f'{t}_{L}': 20 for t in CORE for L in (4096, 8192, 16384, 32768, 65536)}, **{f'{t}_131072': 10 for t in CORE}})
w('fast-v1', 'gate probe (~11 min of the 27B V2 4.00 on one R9700, bf16 KV: ~5 s a sample at 8k, ~22 s at 32k): '
  'the 12 tasks at 8k and 32k, 2 rows each', {**{f'{t}_8192': 2 for t in CORE}, **{f'{t}_32768': 2 for t in CORE}})
w('dev08-v1', '0.8B validation grid: 12 tasks x 4k-32k x 5 rows',
  {f'{t}_{L}': 5 for t in CORE for L in (4096, 8192, 16384, 32768)})
w('dev08-long-v1', "0.8B at its trained context: 12 tasks at 128k and the 7 probe tasks at 256k, 2 rows a cell",
  {**{f'{t}_131072': 2 for t in CORE}, **{f'{t}_262144': 2 for t in LONG}})
w('lsa-v1', 'D-LSA gate probes (YaRN past 256k): 7 tasks x 128k / 256k / 512k / 768k x 8 rows',
  {f'{t}_{L}': 8 for t in LONG for L in (131072, 262144, 524288, 786432)})
# the LSA repo's configs/data/targets/qwen38_27b/ruler_probes_128k.yaml rows (same rows as ours at 128k when the
# prompt-set hashes match: tests/lsa_hash.py)
w('lsa-repo-128k', "the LSA repo's Qwen3.8-27B 128k probe rows",
  {'niah_single_2_131072': [29, 30, 34, 37, 52, 73, 78, 80], 'niah_multikey_1_131072': [33, 40, 61, 68, 81, 88, 90, 91],
   'niah_multikey_2_131072': [21, 36, 45, 67, 69, 76, 79, 98], 'niah_multikey_3_131072': [24, 40, 41, 42, 49, 66, 69, 88],
   'cwe_131072': [24, 30, 33, 44, 51, 80, 84, 86]})
w('kv-long-v1', 'KV finalists at long context: the 7 probe tasks at 128k, 4 rows a cell',
  {f'{t}_131072': 4 for t in LONG})
