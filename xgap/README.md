# xgap-v1 — a superset corpus of documents (calib + xcorp-v1 + new gap documents)

Documents only: no model-generated text, no goldens. Built on colo (`/data/xgap/v1`); the code runs there
(`rad-local/bin/xgap-sync.sh` pushes this directory + `xcorp/build.py` to `/data/xgap/code`).

## Layout of `/data/xgap/v1`
| path | content |
|---|---|
| `raw/challenge/corpus/calib.jsonl` | the challenge calib split (`StillDeadcode/qwen3.8-27b-quant-challenge`, revision `2dcf0061cb683152742b904e2965ce7cd92b2da6`; identical to `/data/xcorp/ref` copy). Never the public split. |
| `inputs/` | `pools-v3.json`, `j217-calib-ids.json`, `jlong-ids.json`, `contpool.json` (from `rad-local/workers/data/w229`) |
| `corpus/calib/calib.jsonl`, `corpus/xcorp-v1/*.jsonl`, `corpus/xgap-v1/<cell>/<source>.jsonl` | lines `{id, source, category, kind, tokens, prompt}` (calib's format + the manifest id) |
| `manifest.jsonl` | one row per document (all origins), `file` + `line` point into `corpus/` |
| `sources.json`, `summary.md`, `dedup.json` | datasets / configs / splits / revisions / licences; token tables; dedup counts |
| `work/` | `dedup-index.pkl`, per-source build outputs `work/src/<id>.{jsonl,meta.jsonl,stats.json}`, pinned revisions `work/rev/` |

## Manifest fields
`id, origin (calib | xcorp-v1 | xgap-v1), cell, category, source, dataset, config, split, revision, licence, kind (text|chat),
tokens (Qwen3.8 tokenizer, whole rendered prompt), length_band, holdout, file, line`, plus
- calib: `calib_idx, pool (fit/search/qad/verify), judge [verify, j217, jc228, jlong], cont228_train`
- xcorp-v1: `xcorp_file` (every `pisess*` file is refused)
- xgap-v1: `lang` (ISO 639-1 where one exists, else 639-3, script suffix only for multi-script languages: `sr-Cyrl`, `kk-Cyrl`; programming
  language name for code) and `lang_src` (the source's own value), `src_id`, `budget_key`, and where they apply `thinking (on|off)`, `has_system`, `length: xl` (whole session > 128k),
  `windowed`/`orig_tokens` (book longer than 128k: a paragraph-aligned window), `piece/pieces` (session cut at user turns), `repo`, `session`, `turns`,
  agent sessions: `format` (claude-code | pi), `tools_inferred` (any tool of the session has an inferred definition), `tools_real`, `tools_inferred_names`.
- chat docs of every origin carry `thinking` and `has_system` (parsed from the rendered prompt for calib / xcorp).

## Rules
- **Rendering** (chats): the model dir's `chat_template.jinja` via `AutoTokenizer.apply_chat_template`. Any assistant turn with real reasoning
  -> thinking on (template default: xhigh preamble, every turn keeps its `<think>` block). No reasoning in the whole chat ->
  `enable_thinking=False` (no effort preamble; the template's empty think blocks). Tools go through the template's Qwen tool-call format
  from structured sources. Plain text: the raw text (code: `# path\n` + file, as calib).
- **Tool definitions of agent sessions**: the logs carry none. Known harness + tool name + every argument key seen in the session is a parameter of the real
  definition -> the real definition: Claude Code (`data/claude-code-tools.json`, from `@anthropic-ai/claude-agent-sdk` `sdk-tools.d.ts` via `extract_cc_tools.py`: real
  names, parameter schemas and parameter descriptions; the tool-level description is a one-line summary, the harness's long tool prompts are not published as data)
  and pi (`data/pi-tools.json`, from the MIT `@earendil-works/pi-coding-agent` package via `extract_pi_tools.mjs`, v1.5.2: name, description, schema). OpenHands
  tools (SWE-rebench / Nemotron-SWE) and SWE-smith / hermes / glaive / When2Call tools come with the records (real). Everything else (other harnesses, MCP and extension
  tools, older pi `edit` with oldText / newText) is inferred from the calls (typed parameters, `required` = keys in every call) and tagged `tools_inferred`.
- **System prompts**: the source's real system text is kept (SWE-agent harnesses, glaive, hermes JSON mode) and tagged `has_system`; Hermes'
  function-calling prompt is pure format boilerplate replaced by the template and is dropped; nothing is invented.
- **Dedup**: normalized-content hash (lowercase, whitespace collapsed), then the line rule over non-boilerplate lines >= 50 chars (a line in >= 3 indexed
  documents is boilerplate): text documents are rejected at >= 20 % or >= 25 of their lines already in calib / public / xcorp (catches a whole original that
  contains an earlier 8k window); chats and agent sessions at >= 50 % (different trajectories over the same repository share file contents and tool schemas) —
  except against public, where the text rule always applies. Public: hashes only, read from `/data/xcorp/ref/corpus/public.jsonl`, never copied.
  Within xgap, `assemble.py` repeats exact + line rule across sources (later source loses; documents of the same session are exempt from the line rule).
- **Holdout**: `assemble.py`: per (cell, length band) stratum the documents ranked by `sha256("xgap-v1-holdout-20261125:" + id)`, the first
  round(10 %) get `holdout: true`. Fixed once the manifest is written.
- **Budget keys**: a document is accepted while its key (length band / language / one budget) is below budget; it may cross the
  budget by at most half its size.

## Rebuild
```
rad-local/bin/xgap-sync.sh                                       # on the VM
# on colo, from an empty cwd such as /data/xgap/work:
export XGAP_ROOT=/data/xgap/v1 CALIB_REVISION=2dcf0061cb683152742b904e2965ce7cd92b2da6 HF_HUB_DISABLE_PROGRESS_BARS=1
python3 /data/xgap/code/xgap/build_index.py                      # dedup index (calib + public hashes + xcorp-v1), ~25 s
python3 /data/xgap/code/xgap/stackedu_langs.py $XGAP_ROOT/work/stackedu_langs.json   # shard -> language map for the Swift source
python3 /data/xgap/code/xgap/tag.py                              # calib + xcorp tagging, copies into corpus/
bash /data/xgap/code/xgap/run.sh SCALE 16 3000 [--pilot]         # all sources (SCALE 1 = the full ~80M tokens; pilot 0.027 with --pilot)
python3 /data/xgap/code/xgap/assemble.py                         # dedup, holdout, manifest, sources.json, summary.md
python3 /data/xgap/code/xgap/samples.py 1                        # sample documents per source
python3 /data/xgap/code/xgap/verify.py 300                       # id / line / token-count / duplicate checks
python3 /data/xgap/code/xgap/report.py > report-tables.md        # tables for the report
```
Restartable: a source with a `.stats.json` is skipped; outputs are written `.part` and renamed. Delete a source's three files to rebuild it.
Source registry (budgets, repos, adapters): `sources.py`; adapters: `adapters.py`; generators (file shuffle, per-file token cap): `gens.py`;
agent-trace converters: `sessfmt.py` (from `xcorp/longtraces.py`); the build loop: `build_source.py`.
