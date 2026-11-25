"""sources.py -- the xgap-v1 source registry. Budgets are Qwen tokens at scale 1 (the full build; the pilot scales them
down). cells: long-docs, code-langs, agent-sessions, multilingual-text, multilingual-chat, long-chat, tools-other.

Keyers: "band" (budget key = length band, 8k..128k), "code" (< 8k tokens -> cell code-langs key "cl"; 8k..128k ->
cell long-docs, key = band), "all" (one budget; sessions / SWE also have an "xl" budget for whole sessions > 128k)."""

BAND_TOT = 6_250_000            # long-docs: 25M over 8-16k / 16-32k / 32-64k / 64-128k, equal token mass
CODE_LANG_TOT = 5_000_000       # code-langs
A, B, C, D = "8-16k", "16-32k", "32-64k", "64-128k"
CP = "common-pile/"

S = []


def add(**kw):
    kw.setdefault("split", "train"); kw.setdefault("config", None); kw.setdefault("gen", "stream")
    kw.setdefault("min_tok", 64); kw.setdefault("max_tok", 32768)
    S.append(kw)


# ------------------------------------------------------------------ long-docs (whole originals, 8-128k)
def long_src(sid, repo, adapter, cat, w, **kw):
    add(id=sid, source=sid.split("__")[0].replace("--", "/"), cell="long-docs", category=cat, repo=repo, adapter=adapter, keyer="band",
        budgets={b: int(BAND_TOT * f) for b, f in w.items()}, min_tok=8192, max_tok=131072, per_file=300_000, **kw)


BOOKS = dict(window=True)
long_src("pg19", "emozilla/pg19", "pg19", "books", {D: .30, C: .15, B: .05}, **BOOKS)
long_src("gutenberg", CP + "project_gutenberg_filtered", "commonpile", "books", {D: .25, C: .15, B: .10}, english_only=True, **BOOKS)
long_src("doab", CP + "doab_filtered", "commonpile", "books", {D: .25, C: .25, B: .10, A: .05}, **BOOKS)
long_src("arxiv", CP + "arxiv_papers_filtered", "commonpile", "science", {D: .10, C: .20, B: .20, A: .10})
long_src("libretexts", CP + "libretexts_filtered", "commonpile", "textbook", {D: .05, C: .10, B: .10, A: .10})
long_src("caselaw", CP + "caselaw_access_project_filtered", "commonpile", "legal", {D: .05, C: .05, B: .10, A: .10})
long_src("peS2o", CP + "peS2o_filtered", "commonpile", "science", {C: .10, B: .15, A: .15})
long_src("wikipedia--en", "wikimedia/wikipedia", "wikipedia", "encyclopedic", {B: .10, A: .20}, config="20231101.en", lang="en")
for lg in ("spa_Latn", "deu_Latn", "fra_Latn", "rus_Cyrl", "por_Latn"):
    long_src(f"fineweb2-hq--{lg}", "epfml/FineWeb2-HQ", "fineweb2", "web", {A: .03}, config=lg, lang=lg, columns=["text", "language_score"])

# ------------------------------------------------------------------ code (whole files): code-langs (1k-8k) + long-docs code part (8k+)
# per language: smol-xl directory (data/<dir>/data.json) or a stack-edu shard set
CODE_LANGS = [("python", "python"), ("javascript", "javascript"), ("typescript", "typescript"), ("rust", "rust"), ("go", "go"),
              ("c", "c"), ("cpp", "c++"), ("java", "java"), ("csharp", "c-sharp"), ("kotlin", "kotlin"), ("swift", None),
              ("ruby", "ruby"), ("php", "php"), ("shell", "shell"), ("sql", "sql"), ("haskell", "haskell"), ("ocaml", "ocaml"),
              ("scala", "scala"), ("lua", "lua"), ("zig", "zig"), ("julia", "julia"), ("r", "r"),
              ("dart", "dart"), ("elixir", "elixir"), ("perl", "perl"), ("lean", "lean"), ("erlang", "erlang"), ("fortran", "fortran")]
NL = 22        # per-language budgets are CODE_LANG_TOT / 22 (the first 22 languages); the 6 added ones use the same rule
for name, d in CODE_LANGS:
    budgets = {"cl": CODE_LANG_TOT // NL, A: int(BAND_TOT * .15) // NL, B: int(BAND_TOT * .10) // NL}    # no 32k+ code: one 32-64k file overshoots a per-language budget
    if d:
        add(id=f"code--{name}", source=f"the-stack-smol-xl/{name}", cell="code-langs", category="code", repo="bigcode/the-stack-smol-xl",
            adapter="stack", keyer="code", budgets=budgets, min_tok=1024, max_tok=131072, data_files=[f"data/{d}/data.json"],
            lang=name, per_file=None, window=False)
    else:   # Swift: not in the-stack-smol-xl; stack-edu shards of that language (shard list from stackedu_langs.json)
        add(id=f"code--{name}", source=f"stackv2-edu/{name}", cell="code-langs", category="code", repo=CP + "stackv2_edu_filtered",
            adapter="stackedu", keyer="code", budgets=budgets, min_tok=1024, max_tok=131072, stackedu_lang="Swift", lang="Swift", per_file=None)

# ------------------------------------------------------------------ multilingual-text
for lg, b in (("cmn_Hani", 700_000), ("jpn_Jpan", 700_000), ("rus_Cyrl", 600_000), ("arb_Arab", 600_000), ("fas_Arab", 500_000),
              ("ell_Grek", 500_000), ("vie_Latn", 500_000)):
    add(id=f"mlt--fineweb2-hq--{lg}", source=f"fineweb2-hq/{lg}", cell="multilingual-text", category="multilingual", repo="epfml/FineWeb2-HQ",
        config=lg, adapter="fineweb2", keyer="all", budgets={"all": b}, min_tok=128, max_tok=32768, lang=lg,
        columns=["text", "language_score"], per_file=b // 4)
for lg in ("kor_Hang", "hin_Deva", "tha_Thai", "heb_Hebr", "ukr_Cyrl", "ben_Beng", "tam_Taml", "urd_Arab", "kat_Geor", "hye_Armn",
           "bul_Cyrl", "srp_Cyrl", "kaz_Cyrl"):
    add(id=f"mlt--fineweb2--{lg}", source=f"fineweb-2/{lg}", cell="multilingual-text", category="multilingual", repo="HuggingFaceFW/fineweb-2",
        config=lg, adapter="fineweb2", keyer="all", budgets={"all": 300_000}, min_tok=128, max_tok=32768, lang=lg,
        columns=["text", "language_score"], per_file=100_000, min_lang_score=0.95)
for lg in ("zh", "ja", "ko", "ru", "ar", "hi", "fa", "th", "he", "el", "uk", "bn", "vi", "ta", "ur", "ka", "hy", "kk", "ml", "te", "mr", "sr"):
    add(id=f"mlt--wikipedia--{lg}", source=f"wikipedia/{lg}", cell="multilingual-text", category="multilingual", repo="wikimedia/wikipedia",
        config=f"20231101.{lg}", adapter="wikipedia", keyer="all", budgets={"all": 177_000}, min_tok=128, max_tok=32768, lang=lg,
        per_file=60_000)

# ------------------------------------------------------------------ multilingual-chat (thinking off; no system prompts in these sources)
add(id="mlc--aya", source="aya_dataset", cell="multilingual-chat", category="chat", repo="CohereLabs/aya_dataset", adapter="aya",
    keyer="all", budgets={"all": 1_000_000}, min_tok=32, max_tok=8192, per_file=None)
add(id="mlc--oasst2", source="oasst2", cell="multilingual-chat", category="chat", repo="OpenAssistant/oasst2", gen="oasst",
    keyer="all", budgets={"all": 1_200_000}, min_tok=32, max_tok=1199, non_english=True)
add(id="mlc--wildchat", source="wildchat", cell="multilingual-chat", category="chat", repo="allenai/WildChat-4.8M", adapter="wildchat",
    keyer="all", budgets={"all": 2_800_000}, min_tok=64, max_tok=2999, non_english=True, per_file=500_000,
    columns=["conversation", "language", "toxic", "redacted"])

# ------------------------------------------------------------------ long-chat (whole threads)
add(id="lc--wildchat", source="wildchat", cell="long-chat", category="chat", repo="allenai/WildChat-4.8M", adapter="wildchat",
    keyer="all", budgets={"all": 4_500_000}, min_tok=3000, max_tok=131072, min_msgs=4, per_file=900_000,
    columns=["conversation", "language", "toxic", "redacted"])
add(id="lc--oasst2", source="oasst2", cell="long-chat", category="chat", repo="OpenAssistant/oasst2", gen="oasst",
    keyer="all", budgets={"all": 500_000}, min_tok=1200, max_tok=131072)

# ------------------------------------------------------------------ tools-other
add(id="tools--glaive", source="glaive-fc-v2", cell="tools-other", category="tools", repo="glaiveai/glaive-function-calling-v2",
    adapter="glaive", keyer="all", budgets={"all": 1_000_000}, min_tok=64, max_tok=16384, per_file=None)
add(id="tools--hermes-func", source="hermes-fc/func_calling_singleturn", cell="tools-other", category="tools", repo="NousResearch/hermes-function-calling-v1",
    config="func_calling_singleturn", adapter="hermes", keyer="all", budgets={"all": 600_000}, min_tok=64, max_tok=16384, per_file=None)
add(id="tools--hermes-multi", source="hermes-fc/func_calling", cell="tools-other", category="tools", repo="NousResearch/hermes-function-calling-v1",
    config="func_calling", adapter="hermes", keyer="all", budgets={"all": 400_000}, min_tok=64, max_tok=16384, per_file=None)
add(id="tools--hermes-json", source="hermes-fc/json_mode_singleturn", cell="tools-other", category="tools", repo="NousResearch/hermes-function-calling-v1",
    config="json_mode_singleturn", adapter="hermes", keyer="all", budgets={"all": 500_000}, min_tok=64, max_tok=16384, need_tools=False, per_file=None)
add(id="tools--hermes-json-agentic", source="hermes-fc/json_mode_agentic", cell="tools-other", category="tools", repo="NousResearch/hermes-function-calling-v1",
    config="json_mode_agentic", adapter="hermes", keyer="all", budgets={"all": 200_000}, min_tok=64, max_tok=16384, need_tools=False, per_file=None)
add(id="tools--when2call", source="when2call", cell="tools-other", category="tools", repo="nvidia/When2Call", config="train_sft",
    adapter="when2call", keyer="all", budgets={"all": 600_000}, min_tok=64, max_tok=16384, per_file=None)

# ------------------------------------------------------------------ agent-sessions
SESSION_DIRS = {   # dir under the raw snapshot root: (tokens, licence)
    "Infatoshi__kernelbench-hard-traces": 2_200_000, "Infatoshi__kernelbench-mega-traces": 1_100_000,
    "thomasmustier__pi-for-excel-sessions": 1_100_000, "thomasmustier__pi-mono-sessions": 800_000,
    "thomasmustier__pi-extensions-sessions": 600_000, "thomasmustier__pine-of-glass-sessions": 800_000,
    "thomasmustier__pinet-sessions": 500_000, "thomasmustier__pi-nes-sessions": 100_000,
    "championswimmer__pi-coding-sessions": 300_000, "AlinCiocan__fable-5-claude-code-traces": 500_000,
    "armand0e__claude-fable-5-claude-code": 1_600_000, "trace-commons__agent-traces": 1_200_000}
LIC = {"AlinCiocan__fable-5-claude-code-traces": "cc-by-4.0", "trace-commons__agent-traces": "cc-by-4.0"}
for d, b in SESSION_DIRS.items():
    add(id=f"agent--{d}", source=d.replace("__", "/", 1), cell="agent-sessions", category="agentic", gen="sessions", dir=d,
        lic=LIC.get(d, "mit"), repo=d.replace("__", "/", 1), keyer="all", budgets={"all": b, "xl": b // 4},
        min_tok=4096, max_tok=131072, xl_max=1_000_000)
for sid, repo, cfg, split, lic in (("swe-rebench-openhands", "nebius/SWE-rebench-openhands-trajectories", None, "train", "cc-by-4.0"),
                                   ("nemotron-swe", "nvidia/Nemotron-SWE-v1", None, "r2e_gym", None),
                                   ("swe-smith", "SWE-bench/SWE-smith-trajectories", None, "tool", "mit")):
    add(id=f"agent--{sid}", source=sid, cell="agent-sessions", category="agentic", repo=repo, config=cfg, split=split, adapter="swe", lic=lic,
        keyer="all", budgets={"all": 1_000_000, "xl": 100_000}, min_tok=4096, max_tok=131072, xl_max=524288, per_file=250_000)

BY_ID = {s["id"]: s for s in S}
