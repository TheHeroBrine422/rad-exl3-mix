#!/usr/bin/env python3
"""mk8k.py TOKENIZER_DIR SRC_PROMPT OUT -- an 8k prompt for bench.py (#175, w137): w138's ctx-16k text (public radiance
docs, mkprompts.py) cut to 8192 - 48 tokens + the same request, so the chat prompt is ~8192 tokens. Timing only."""
import sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
ask = "\n\nSummarize the text above in detail, section by section."
src = open(sys.argv[2]).read()
assert src.endswith(ask)
ids = tok(src[: -len(ask)], add_special_tokens=False)["input_ids"]
t = tok.decode(ids[: 8192 - 48]) + ask
open(sys.argv[3], "w").write(t)
print(len(tok(t, add_special_tokens=False)["input_ids"]))
