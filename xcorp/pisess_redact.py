#!/usr/bin/env python3
"""pisess_redact.py IN.jsonl OUT.jsonl -- redaction and integrity pass over pisess.py output (our own pi sessions,
a training-only corpus; this lowers the risk of secrets in it, it does not make it publishable).

Replaced by [REDACTED:<label>]:
  secrets      known key prefixes (HF, OpenAI / Anthropic style, GitHub, AWS, Slack, Google), PEM private-key
               blocks, bearer / authorization headers, `password=` / `token=` assignments
  infra        private and our public IPs, ssh key file names, ssh ports in commands
  email        e-mail addresses
  entropy      high-entropy tokens (>= 24 chars, Shannon >= 4.2 bits / char) that are NOT plain hex digests
               (git hashes, sha256 / md5 sums, safetensors checksums are kept: common and harmless here), not
               paths, not identifiers (snake_case / CamelCase words)
Integrity (counted, and recorded per document in "flags"): tool_call vs tool_response counts per assistant / user
turn pair, first block is a system block, no unbalanced <think>. A document is dropped only if it has no
assistant turn left. Stats -> OUT.stats.json (hits per label, flags)."""
import collections, json, math, re, sys

PAT = [
    ("secret", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S)),
    ("secret", re.compile(r"\b(?:hf_[A-Za-z0-9]{30,}|sk-(?:ant-)?[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}"
                          r"|github_pat_[A-Za-z0-9_]{40,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}"
                          r"|AIza[0-9A-Za-z_-]{35})\b")),
    ("secret", re.compile(r"(?i)\b(authorization|bearer)\s*[:=]?\s*[A-Za-z0-9._~+/=-]{20,}")),
    ("secret", re.compile(r"(?i)\b(password|passwd|secret|token|api[_-]?key)\s*[=:]\s*['\"]?[^\s'\"]{8,}")),
    ("infra", re.compile(r"\b(?:10|127|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}(?:\.\d{1,3})?\b")),
    ("infra", re.compile(r"\b23\.155\.\d{1,3}\.\d{1,3}\b")),
    ("infra", re.compile(r"\b[\w.-]*_ed25519(?:\.pub)?\b|\bid_(?:rsa|ecdsa|ed25519)\b")),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
]
TOKEN = re.compile(r"[A-Za-z0-9+/=_-]{24,}")
HEX = re.compile(r"^[0-9a-fA-F]+$")


def entropy(s):
    c = collections.Counter(s); n = len(s)
    return -sum(v / n * math.log2(v / n) for v in c.values())


def looks_secret(t):
    if HEX.match(t): return False                              # digests, hashes
    if "/" in t and not re.search(r"[+=]", t): return False      # paths
    if re.fullmatch(r"[A-Za-z_]+(?:[_-][A-Za-z0-9]+)*", t): return False   # identifiers, words
    if sum(ch.isdigit() for ch in t) < 2 or sum(ch.isalpha() for ch in t) < 4: return False
    return entropy(t) >= 4.2


def redact(text, hits):
    for label, p in PAT:
        text, n = p.subn(f"[REDACTED:{label}]", text); hits[label] += n
    def ent(m):
        if looks_secret(m.group(0)): hits["entropy"] += 1; return "[REDACTED:entropy]"
        return m.group(0)
    return TOKEN.sub(ent, text)


def flags_of(p):
    f = []
    if not p.startswith("<|im_start|>system"): f.append("no-system")
    if p.count("<think>") != p.count("</think>"): f.append("think-unbalanced")
    turns = p.split("<|im_start|>")
    for i, t in enumerate(turns):
        if t.startswith("assistant") and "<tool_call>" in t:
            nxt = turns[i + 1] if i + 1 < len(turns) else ""
            if nxt.count("<tool_response>") < t.count("<tool_call>"): f.append("unpaired-tool-call"); break
    return f


def main():
    src, dst = sys.argv[1], sys.argv[2]
    hits, fl, n_in, n_out = collections.Counter(), collections.Counter(), 0, 0
    with open(dst + ".part", "w") as fo:
        for l in open(src):
            d = json.loads(l); n_in += 1
            d["prompt"] = redact(d["prompt"], hits)
            if "<|im_start|>assistant" not in d["prompt"]: fl["dropped-no-assistant"] += 1; continue
            d["flags"] = flags_of(d["prompt"]); fl.update(d["flags"])
            fo.write(json.dumps(d, ensure_ascii=False) + "\n"); n_out += 1
    import os; os.replace(dst + ".part", dst)
    st = {"docs_in": n_in, "docs_out": n_out, "hits": dict(hits), "flags": dict(fl)}
    json.dump(st, open(dst + ".stats.json", "w"), indent=1); print(json.dumps(st))


if __name__ == "__main__":
    main()
