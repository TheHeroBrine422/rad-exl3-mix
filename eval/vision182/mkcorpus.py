#!/usr/bin/env python3
"""mkcorpus.py SET GEN TOKENIZER OUT -- the #182 KL corpus (w156): every item with its model-written answer (GEN/ans-img.jsonl,
GEN/ans-text.jsonl) rendered with the checkpoint's own chat template (one user turn: the images, then the question;
generation prompt, thinking off), each image part's `<|vision_start|><|image_pad|><|vision_end|>` replaced by radiance's
media marker; then the answer and `<|im_end|>` (left off where the answer was cut at max_tokens).
Scored ("score_tail"): every token after the last image (text items: after `<|im_start|>user\\n`), i.e. the question
text and the answer. OUT/corpus.jsonl (radiance --kld-corpus with "media": absolute image paths) and OUT/meta.jsonl
{id, kind, source, n_images, tail, ans}: the doc's scored rows are its `tail`, the last `ans` of them the answer's
(the gen mask). Image items first (items.jsonl order), then the text control."""
import json, os, sys
from transformers import AutoTokenizer
SET, GEN, TOKP, OUT = sys.argv[1:5]
tok = AutoTokenizer.from_pretrained(TOKP)
TRIPLE, MARK = "<|vision_start|><|image_pad|><|vision_end|>", "<__media__>"
os.makedirs(OUT, exist_ok=True)
n = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
cor, meta = open(f"{OUT}/corpus.jsonl", "w"), open(f"{OUT}/meta.jsonl", "w")
for items, ans in (("items.jsonl", "ans-img.jsonl"), ("text-items.jsonl", "ans-text.jsonl")):
    A = {a["id"]: a for a in map(json.loads, open(f"{GEN}/{ans}"))}
    for it in map(json.loads, open(f"{SET}/{items}")):
        a = A.get(it["id"])
        if a is None: print("no answer", it["id"]); continue
        content = [{"type": "image"} for _ in it["images"]] + [{"type": "text", "text": it["question"]}]
        head = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        assert head.count(TRIPLE) == len(it["images"]), (it["id"], head[:300])
        head = head.replace(TRIPLE, MARK)
        body = a["text"] + ("<|im_end|>" if a["finish"] == "stop" else "")
        cut = head.rfind(MARK) + len(MARK) if it["images"] else head.index("<|im_start|>user\n") + len("<|im_start|>user\n")
        tail_text = head[cut:]
        tail = n(tail_text + body)
        ansn = tail - n(tail_text)
        cor.write(json.dumps({"source": f"{it['kind']}:{it['id']}", "prompt": head + body,
                              "media": [os.path.abspath(f"{SET}/img/{f}") for f in it["images"]], "score_tail": tail}) + "\n")
        meta.write(json.dumps({"id": it["id"], "kind": it["kind"], "source": it["source"], "n_images": len(it["images"]),
                               "tail": tail, "ans": ansn, "finish": a["finish"]}) + "\n")
cor.close(); meta.close()
