#!/usr/bin/env python3
"""mktest.py MODEL OUT.jsonl -- a small public test corpus for mkgold27.py (no corpus data): the model's own README and
LICENSE, transformers' Qwen3.5 modeling source, a rendered chat (control tokens, score_from 24) and a short line, cut
to a spread of lengths (~100 to ~8k tokens)."""
import json, os, sys
import transformers.models.qwen3_5.modeling_qwen3_5 as q35
from transformers import AutoTokenizer

model, out = sys.argv[1:3]
tok = AutoTokenizer.from_pretrained(model)
src = open(q35.__file__).read()
readme = open(f"{model}/README.md").read() if os.path.exists(f"{model}/README.md") else src[::-1]
chat = tok.apply_chat_template([
    {"role": "system", "content": "You are a careful assistant."},
    {"role": "user", "content": "Explain in a few paragraphs how a gated delta network differs from softmax attention."},
    {"role": "assistant", "content": "A gated delta network keeps a fixed-size state that it updates at every token "
     "with a delta rule: it reads the value the state predicts for the current key, and writes back a correction "
     "towards the true value, scaled by a learned step size. A gate decays the old state first. Softmax attention "
     "instead keeps every past key and value and compares the current query against all of them. " * 4},
    {"role": "user", "content": "And what does that mean for long documents?"}], tokenize=False, add_generation_prompt=True)


def cut(text, ntok):
    ids = tok(text, add_special_tokens=False)["input_ids"][:ntok]
    return tok.decode(ids)


docs = [(cut(src, 8000), "test/modeling", 0), (cut(readme, 3000), "test/readme", 0), (chat, "test/chat", 24),
        (cut(src[40000:], 1200), "test/modeling2", 0), ("The quick brown fox jumps over the lazy dog. " * 10, "test/short", 0)]
with open(out, "w") as f:
    for p, s, sf in docs: f.write(json.dumps({"prompt": p, "source": s, "score_from": sf}) + "\n")
print(out, [len(tok(p, add_special_tokens=False)["input_ids"]) for p, _, _ in docs])
