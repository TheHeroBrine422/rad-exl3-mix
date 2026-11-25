"""xgap-v1 shared pieces: paths, tokenizer, Qwen3.8 chat rendering (thinking on / off per document), hashing, length
bands, the dedup index (calib + public + xcorp), licence screen, restartable writers."""
import hashlib, json, os, pickle, re, sys

ROOT = os.environ.get("XGAP_ROOT", "/data/xgap/v1")
TOKDIR = os.environ.get("XGAP_TOKENIZER", "/data/models/qwen38-27b-bf16")
XCORP = os.environ.get("XGAP_XCORP", "/data/xcorp/v1")
PUBLIC = os.environ.get("XGAP_PUBLIC", "/data/xcorp/ref/corpus/public.jsonl")   # read for hashes only; never copied
CALIB = f"{ROOT}/raw/challenge/corpus/calib.jsonl"
SRC_DIR = f"{ROOT}/work/src"            # per-source build outputs (before the global dedup / assembly)
INDEX_PKL = f"{ROOT}/work/dedup-index.pkl"
SEED = 20261125
ORIGIN_BIT = {"calib": 1, "public": 2, "xcorp": 4}
BIT_ORIGIN = {v: k for k, v in ORIGIN_BIT.items()}

_tok = None


def tok():
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(TOKDIR)
        _tok.model_max_length = 10 ** 9
    return _tok


def ntok(text):
    return len(tok()(text, add_special_tokens=False)["input_ids"])


def decode(ids):
    return tok().decode(ids)


# ------------------------------------------------------------------ rendering
def has_reasoning(msgs):
    return any((m.get("reasoning_content") or "").strip() for m in msgs if m.get("role") == "assistant")


def render_chat(msgs, tools=None):
    """The model dir's chat_template.jinja. Any assistant turn with real reasoning -> thinking on (template default:
    xhigh preamble, every turn keeps its think block). No reasoning anywhere -> enable_thinking=False (no effort
    preamble; assistant turns carry the template's empty think block). Returns (prompt, "on" | "off")."""
    think = has_reasoning(msgs)
    kw = {} if think else {"enable_thinking": False}
    prompt = tok().apply_chat_template(msgs, tools=tools or None, tokenize=False, **kw)
    return prompt, "on" if think else "off"


SPECIAL = re.compile(r"<\|(?:im_start|im_end|endoftext|vision_start|vision_end|image_pad|video_pad|object_ref_start|"
                     r"object_ref_end|box_start|box_end|quad_start|quad_end|fim_prefix|fim_middle|fim_suffix|fim_pad|"
                     r"repo_name|file_sep)\|>")


def has_special(s):
    return SPECIAL.search(s) is not None


# ------------------------------------------------------------------ hashing / bands
_WS = re.compile(r"\s+")


def norm_hash(s):
    return hashlib.blake2b(_WS.sub(" ", s.lower()).strip().encode("utf-8", "replace"), digest_size=16).hexdigest()


def lhash(line):
    return int.from_bytes(hashlib.blake2b(line.encode("utf-8", "replace"), digest_size=8).digest(), "little")


def long_lines(text):
    out = []
    for l in text.split("\n"):
        l = l.strip()
        if len(l) >= 50: out.append(l)
    return out


BANDS = [(1024, "<1k"), (2048, "1-2k"), (4096, "2-4k"), (8192, "4-8k"), (16384, "8-16k"), (32768, "16-32k"),
         (65536, "32-64k"), (131072, "64-128k")]
BAND_ORDER = [b for _, b in BANDS] + ["128k+"]
KS = {"8-16k": (8192, 16384), "16-32k": (16384, 32768), "32-64k": (32768, 65536), "64-128k": (65536, 131072)}


def band(n):
    for hi, name in BANDS:
        if n <= hi: return name
    return "128k+"


def repetitive(text):
    ls = [l for l in text.split("\n") if l.strip()]
    if len(ls) >= 10 and len(set(ls)) / len(ls) < 0.5: return True
    w = text.split()
    return len(w) >= 200 and len(set(w)) / len(w) < 0.1


NONPERMISSIVE = re.compile(r"non-?commercial|(?<![a-z])nc(?![a-z])|research[- ]only|proprietary", re.I)


def licence_ok(lic):
    """Record-level screen for sources that carry a per-record licence string."""
    return not (lic and NONPERMISSIVE.search(str(lic)))


# ------------------------------------------------------------------ dedup index
class Index:
    """docs: normalized-content hash -> origin bits. lines: hash of every line >= 50 chars -> origin bits | count << 3.
    Boilerplate (a line in >= 3 indexed documents) never counts as evidence of a shared document."""
    def __init__(self, docs, lines, ndocs):
        self.docs, self.lines, self.ndocs = docs, lines, ndocs
        self.boiler = {h for h, v in lines.items() if (v >> 3) >= 3}

    def check(self, text):
        """-> None (fresh) or (kind, origin) for the first reason the document is already represented."""
        o = self.docs.get(norm_hash(text))
        if o: return ("exact", BIT_ORIGIN[o & -o])
        ll = [h for h in map(lhash, long_lines(text)) if h not in self.boiler]
        hits = {}
        for h in ll:
            v = self.lines.get(h)
            if v:
                for b, name in BIT_ORIGIN.items():
                    if v & b: hits[name] = hits.get(name, 0) + 1
        if not hits: return None
        n = max(hits.values()); dom = max(hits, key=hits.get)
        # >= 20 % of its long lines, or >= 25 of them: a whole original that contains an earlier 8k window
        if n >= 2 and (n >= 0.2 * len(ll) or n >= 25): return ("lines", dom)
        return None


def load_index():
    with open(INDEX_PKL, "rb") as f: d = pickle.load(f)
    return Index(d["docs"], d["lines"], d["ndocs"])


# ------------------------------------------------------------------ output
class Out:
    """Writes SRC_DIR/<sid>.jsonl.part (+ .meta.jsonl.part, line-aligned tags), renames on close."""
    def __init__(self, sid):
        os.makedirs(SRC_DIR, exist_ok=True)
        self.sid = sid
        self.p = f"{SRC_DIR}/{sid}.jsonl"; self.pm = f"{SRC_DIR}/{sid}.meta.jsonl"
        self.f = open(self.p + ".part", "w"); self.fm = open(self.pm + ".part", "w"); self.n = 0

    def done(self):
        return os.path.exists(f"{SRC_DIR}/{self.sid}.stats.json")

    def write(self, line, meta):
        self.f.write(json.dumps(line, ensure_ascii=False) + "\n"); self.fm.write(json.dumps(meta, ensure_ascii=False) + "\n"); self.n += 1

    def close(self, stats):
        self.f.close(); self.fm.close()
        os.replace(self.p + ".part", self.p); os.replace(self.pm + ".part", self.pm)
        with open(f"{SRC_DIR}/{self.sid}.stats.json.part", "w") as g: json.dump(stats, g, indent=1)
        os.replace(f"{SRC_DIR}/{self.sid}.stats.json.part", f"{SRC_DIR}/{self.sid}.stats.json")


def finished(sid):
    return os.path.exists(f"{SRC_DIR}/{sid}.stats.json")
