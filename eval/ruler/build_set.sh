#!/bin/bash
# build_set.sh SETDIR -- build the ruler-qwen38-v1 set (idempotent: finished cells are skipped). Needs: python3 with
# numpy, scipy, tokenizers; DEPS (pip --target dir with nltk==3.10.3, defusedxml), NLTK_DATA with punkt_tab,
# HAY (PaulGrahamEssays.json, english_words.json, squad.json; sha256 in README), TOK (Qwen3.8 tokenizer.json).
# Ran on colo (2026-11-07): DEPS=/data/work/w66/deps NLTK_DATA=/data/work/w66/nltk_data HAY=/data/work/w66/assets
#   TOK=/data/models/qwen38-27b-bf16/tokenizer.json
set -e
D=${1:?SETDIR}; X="$(dirname "$(readlink -f "$0")")"
: "${DEPS:?}" "${NLTK_DATA:?}" "${HAY:?}" "${TOK:?}" "${JOBS:=3}"
export RAYON_NUM_THREADS=2 OMP_NUM_THREADS=1
G() { nice -n 19 python3 -I "$X/rgen.py" build "$D" --name ruler-qwen38-v1 --seed 42 --tokenizer "$TOK" \
        --haystacks "$HAY" --deps "$DEPS" --nltk-data "$NLTK_DATA" --jobs "$JOBS" "$@"; }
CORE=niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery,vt,cwe,fwe,qa_squad
# core grid: 12 tasks x 4k..128k, 100 rows a cell
G --tasks $CORE --lengths 4096,8192,16384,32768,65536,131072 --samples 100
# long-context probe cells (LSA gate, YaRN): the LSA repo's probe tasks + multivalue + vt, 20 rows a cell
G --tasks niah_single_2,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,vt,cwe \
  --lengths 262144,524288,786432 --samples 20
python3 -I "$X/rgen.py" check "$D"
