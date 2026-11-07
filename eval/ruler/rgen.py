#!/usr/bin/env python3
# Port of NVIDIA RULER's synthetic generators (github.com/NVIDIA/RULER, scripts/data/synthetic/*.py and
# scripts/data/synthetic/constants.py at db43f92ff634bbce1c33d1fb15858aec4bba509a).
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved. Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). Changes: the five generator scripts are folded into one module, every
# cell runs in-process with its own seeded random.Random / np.random.RandomState (the streams the official
# per-process global seeding gives, call for call), the tokenizer is the `tokenizers` library's count of the HF
# tokenizer.json, cwe's binary search bound is clamped to its word list (as in the LSA repo's ruler.patch; no change
# below ~4M tokens), and the files are written in this harness's layout with a manifest of checksums.
"""rgen.py -- build RULER sample sets (the official task definitions, seeded, deterministic, versioned files).

    python3 -I rgen.py build SETDIR --tasks T,... --lengths L,... --samples N [--samples-at L=N,...]
        --tokenizer TOKENIZER.json --haystacks DIR [--deps DIR] [--nltk-data DIR] [--seed 42] [--jobs J]
    python3 -I rgen.py check SETDIR          -- re-hash every file against SETDIR/manifest.json

One file per cell: SETDIR/<task>_<length>.jsonl.gz (gzip, mtime 0), rows {index, input, outputs, answer_prefix,
length, max_length, tokens_to_generate}; `input + answer_prefix` is RULER's prompt for a base model. A cell's rows are
one seeded stream after the haystack-size search, so the first k rows of an N-row file are the k-row file's rows.
Finished cells are skipped (atomic writes: .part, fsync, rename); manifest.json is rewritten after each cell.

Tasks (RULER's synthetic.yaml): niah_single_1/2/3, niah_multikey_1/2/3, niah_multivalue, niah_multiquery, vt, cwe,
fwe, qa_squad (qa_1), qa_hotpot (qa_2, needs hotpotqa.json). Haystack files in --haystacks: PaulGrahamEssays.json
(niah essay tasks), english_words.json (cwe), squad.json (qa_squad); the word lists are vendored in assets/.
"""
import argparse, gzip, hashlib, heapq, json, os, re, string, sys, time, uuid
import random

HERE = os.path.dirname(os.path.abspath(__file__))
UPSTREAM = "NVIDIA/RULER db43f92ff634bbce1c33d1fb15858aec4bba509a"
FORMAT = 1                                   # bump when a row's fields or a generator's output changes
WORDS_DIR = os.path.join(HERE, "assets", "wonderwords-3.0.1")

# scripts/data/synthetic/constants.py (the generators' templates and answer prefixes)
TEMPLATES = {
    'niah': {
        'tokens_to_generate': 128,
        'template': """Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat are all the special magic {type_needle_v} for {query} mentioned in the provided text?""",
        'answer_prefix': """ The special magic {type_needle_v} for {query} mentioned in the provided text are"""
    },
    'variable_tracking': {
        'tokens_to_generate': 30,
        'template': """Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n{context}\nQuestion: Find all variables that are assigned the value {query} in the text above.""",
        'answer_prefix': """ Answer: According to the chain(s) of variable assignment in the text above, {num_v} variables are assigned the value {query}, they are: """
    },
    'common_words_extraction': {
        'tokens_to_generate': 120,
        'template': """Below is a numbered list of words. In these words, some appear more often than others. Memorize the ones that appear most often.\n{context}\nQuestion: What are the 10 most common words in the above list?""",
        'answer_prefix': """ Answer: The top 10 words that appear most often in the list are:"""
    },
    'freq_words_extraction': {
        'tokens_to_generate': 50,
        'template': """Read the following coded text and track the frequency of each coded word. Find the three most frequently appeared coded words. {context}\nQuestion: Do not provide any explanation. Please ignore the dots '....'. What are the three most frequently appeared words in the above coded text?""",
        'answer_prefix': """ Answer: According to the coded text above, the three most frequently appeared words are:"""
    },
    'qa': {
        'tokens_to_generate': 32,
        'template': """Answer the question based on the given documents. Only give me the answer and do not output any other words.\n\nThe following are given documents.\n\n{context}\n\nAnswer the question based on the given documents. Only give me the answer and do not output any other words.\n\nQuestion: {query}""",
        'answer_prefix': """ Answer:""",
    },
}

# scripts/synthetic.yaml (our names for qa_1 / qa_2)
TASKS = {
    'niah_single_1': ('niah', dict(type_haystack='noise', type_needle_k='words', type_needle_v='numbers', num_needle_k=1, num_needle_v=1, num_needle_q=1)),
    'niah_single_2': ('niah', dict(type_haystack='essay', type_needle_k='words', type_needle_v='numbers', num_needle_k=1, num_needle_v=1, num_needle_q=1)),
    'niah_single_3': ('niah', dict(type_haystack='essay', type_needle_k='words', type_needle_v='uuids', num_needle_k=1, num_needle_v=1, num_needle_q=1)),
    'niah_multikey_1': ('niah', dict(type_haystack='essay', type_needle_k='words', type_needle_v='numbers', num_needle_k=4, num_needle_v=1, num_needle_q=1)),
    'niah_multikey_2': ('niah', dict(type_haystack='needle', type_needle_k='words', type_needle_v='numbers', num_needle_k=1, num_needle_v=1, num_needle_q=1)),
    'niah_multikey_3': ('niah', dict(type_haystack='needle', type_needle_k='uuids', type_needle_v='uuids', num_needle_k=1, num_needle_v=1, num_needle_q=1)),
    'niah_multivalue': ('niah', dict(type_haystack='essay', type_needle_k='words', type_needle_v='numbers', num_needle_k=1, num_needle_v=4, num_needle_q=1)),
    'niah_multiquery': ('niah', dict(type_haystack='essay', type_needle_k='words', type_needle_v='numbers', num_needle_k=1, num_needle_v=1, num_needle_q=4)),
    'vt': ('variable_tracking', dict(type_haystack='noise', num_chains=1, num_hops=4)),
    'cwe': ('common_words_extraction', dict(freq_cw=30, freq_ucw=3, num_cw=10)),
    'fwe': ('freq_words_extraction', dict(alpha=2.0)),
    'qa_squad': ('qa', dict(dataset='squad')),
    'qa_hotpot': ('qa', dict(dataset='hotpotqa')),
}
TASK_ORDER = list(TASKS)


def template_of(kind):
    """prepare.py: the 'base' model template around the task template, then the answer prefix."""
    return TEMPLATES[kind]['template'] + TEMPLATES[kind]['answer_prefix']


class Tok:
    """HFTokenizer.text_to_tokens(text) counted: the tokenizer.json's pieces, no special tokens added."""
    def __init__(self, path):
        from tokenizers import Tokenizer
        self.t = Tokenizer.from_file(path)

    def count(self, text):
        return len(self.t.encode(text, add_special_tokens=False).ids)


def _words(name):
    with open(os.path.join(WORDS_DIR, name), encoding="utf-8") as f:
        return [w.rstrip() for w in f.readlines()]


def _depths():
    import numpy as np
    return list(np.round(np.linspace(0, 100, num=40, endpoint=True)).astype(int))


# ---------------------------------------------------------------- niah.py
def gen_niah(L, n, seed, tg, tok, hay_dir, type_haystack, type_needle_k, type_needle_v, num_needle_k, num_needle_v,
             num_needle_q, template, log):
    import numpy as np
    from nltk.tokenize import sent_tokenize
    rnd = random.Random(seed)
    np.random.seed(seed)                      # the script seeds numpy too; niah draws nothing from it
    num_needle_k = max(num_needle_k, num_needle_q)
    needle = "One of the special magic {type_needle_v} for {key} is: {value}."
    if type_haystack == 'essay':
        essay = json.load(open(os.path.join(hay_dir, "PaulGrahamEssays.json")))['text']
        haystack = re.sub(r'\s+', " ", essay).split(" ")
    elif type_haystack == 'noise':
        haystack = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    elif type_haystack == 'needle':
        haystack = needle
    nouns, adjs = _words("nounlist.txt"), _words("adjectivelist.txt")
    words = sorted(list(set(f"{adj}-{noun}" for adj in adjs for noun in nouns)))
    DEPTHS = _depths()

    def generate_random(t):
        if t == 'numbers':
            return str(rnd.randint(10 ** 6, 10 ** 7 - 1))
        if t == 'words':
            return rnd.choice(words)
        if t == 'uuids':
            return str(uuid.UUID(int=rnd.getrandbits(128), version=4))
        raise NotImplementedError(t)

    def generate_input_output(num_haystack):
        keys, values, needles = [], [], []
        for _ in range(num_needle_k):
            keys.append(generate_random(type_needle_k))
            value = []
            for _ in range(num_needle_v):
                value.append(generate_random(type_needle_v))
                needles.append(needle.format(type_needle_v=type_needle_v, key=keys[-1], value=value[-1]))
            values.append(value)
        random.Random(seed).shuffle(needles)
        if type_haystack == 'essay':
            if num_haystack <= len(haystack):
                text = " ".join(haystack[:num_haystack])
            else:
                repeats = (num_haystack + len(haystack) - 1) // len(haystack)
                text = " ".join((haystack * repeats)[:num_haystack])
            document_sents = sent_tokenize(text.strip())
            insertion_positions = [0] + sorted([int(len(document_sents) * (depth / 100))
                                                for depth in rnd.sample(DEPTHS, len(needles))]) + [len(document_sents)]
            document_sents_list = []
            for i in range(1, len(insertion_positions)):
                document_sents_list.append(" ".join(document_sents[insertion_positions[i - 1]:insertion_positions[i]]))
                if i - 1 < len(needles):
                    document_sents_list.append(needles[i - 1])
            context = " ".join(document_sents_list)
        else:
            if type_haystack == 'noise':
                sentences = [haystack] * num_haystack
            else:
                sentences = [haystack.format(type_needle_v=type_needle_v, key=generate_random(type_needle_k),
                                             value=generate_random(type_needle_v)) for _ in range(num_haystack)]
            indexes = sorted(rnd.sample(range(num_haystack), len(needles)), reverse=True)
            for index, element in zip(indexes, needles):
                sentences.insert(index, element)
            context = "\n".join(sentences)
        indices = rnd.sample(range(num_needle_k), num_needle_q)
        queries = [keys[i] for i in indices]
        answers = [a for i in indices for a in values[i]]
        query = ', '.join(queries[:-1]) + ', and ' + queries[-1] if len(queries) > 1 else queries[0]
        tpl, tnv = template, type_needle_v
        if num_needle_q * num_needle_v == 1:
            tpl = tpl.replace('Some', 'A').replace('are all', 'is').replace('are', 'is').replace('answers', 'answer')
            tnv = tnv[:-1]
        return tpl.format(type_needle_v=tnv, context=context, query=query), answers

    incremental = {'essay': 500, 'noise': 25, 'needle': 25}[type_haystack]
    if type_haystack != 'essay' and L < 4096:
        incremental = 5
    sample_input_text, _ = generate_input_output(incremental)
    tokens_per_haystack = tok.count(sample_input_text) / incremental
    est = int((L / tokens_per_haystack) * 3)
    num_haystack = _search(lambda m: tok.count(generate_input_output(m)[0]) + tg, L, incremental,
                           max(est, incremental * 2), log)
    if num_haystack is None:
        num_haystack = incremental
    rows = []
    for index in range(n):
        used = num_haystack
        while True:
            input_text, answer = generate_input_output(used)
            length = tok.count(input_text) + tg
            if length <= L:
                break
            if used > incremental:
                used -= incremental
            else:                             # the official loop retries forever here
                raise RuntimeError("niah: no haystack fits")
        rows.append(_split(input_text, answer, length, TEMPLATES['niah']['answer_prefix'], index))
    return rows


def _search(total_of, L, lo, hi, log):
    """The generators' binary search: the largest haystack whose prompt + generation fits L."""
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        t = total_of(mid)
        if t <= L:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    log(f"  haystack {best}")
    return best


def _split(input_text, answer, length, answer_prefix_tpl, index):
    i = input_text.rfind(answer_prefix_tpl[:10])
    return {'index': index, 'input': input_text[:i], 'outputs': answer, 'answer_prefix': input_text[i:],
            'length': length}


# ---------------------------------------------------------------- variable_tracking.py
def gen_vt(L, n, seed, tg, tok, hay_dir, type_haystack, num_chains, num_hops, template, log):
    import numpy as np
    rnd = random.Random(seed)
    nrs = np.random.RandomState(seed)
    if type_haystack != 'noise':
        raise NotImplementedError("vt: only the noise haystack is in RULER's task list")
    haystack = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    T = TEMPLATES['variable_tracking']

    def generate_chains(is_icl):
        k = 5 if not is_icl else 3
        hops = num_hops if not is_icl else min(10, num_hops)
        vars_all = [''.join(rnd.choices(string.ascii_uppercase, k=k)).upper() for _ in range((hops + 1) * num_chains)]
        while len(set(vars_all)) < num_chains * (hops + 1):
            vars_all.append(''.join(rnd.choices(string.ascii_uppercase, k=k)).upper())
        vars_ret, chains_ret = [], []
        for i in range(0, len(vars_all), hops + 1):
            this_vars = vars_all[i:i + hops + 1]
            vars_ret.append(this_vars)
            this_chain = [f"VAR {this_vars[0]} = 12345"] if is_icl else [f"VAR {this_vars[0]} = {str(nrs.randint(10000, 99999))}"]
            for j in range(hops):
                this_chain.append(f"VAR {this_vars[j + 1]} = VAR {this_vars[j]} ")
            chains_ret.append(this_chain)
        return vars_ret, chains_ret

    def generate_input_output(num_noises, is_icl):
        vars_, chains = generate_chains(is_icl)
        value = chains[0][0].split("=")[-1].strip()
        sentences = [haystack] * num_noises
        for chain in chains:
            positions = list(sorted(rnd.sample(range(len(sentences)), len(chain))))
            for insert_pi, j in zip(positions, range(len(chain))):
                sentences.insert(insert_pi + j, chain[j])
        context = "\n".join(sentences).replace(". \n", ".\n")
        tpl = template
        if is_icl and tpl != T['template'] + T['answer_prefix']:
            tpl = T['template'] + T['answer_prefix']
        return tpl.format(context=context, query=value, num_v=num_hops + 1), vars_[0]

    def randomize_icl(icl_example):
        icl_tgt = icl_example.strip().split()[-num_hops - 1:]
        for item in icl_tgt:
            new_item = ''.join(rnd.choices(string.ascii_uppercase, k=len(item))).upper()
            icl_example = icl_example.replace(item, new_item)
        return icl_example.replace("12345", str(nrs.randint(10000, 99999)))

    def sys_vartrack(num_samples, max_seq_length, icl_example, final_output):
        add_fewshot = True
        tokens_to_generate = tg if icl_example is not None else 0
        if icl_example:
            incremental = 10
            if L < 4096:
                incremental = 5
        else:
            incremental = 5
        example_tokens = 0
        if add_fewshot and icl_example is not None:
            icl_example = icl_example['input'] + " " + ' '.join(icl_example['outputs']) + '\n'
            example_tokens = tok.count(icl_example)
        is_icl = add_fewshot & (icl_example is None)
        sample_input_text, _ = generate_input_output(incremental, is_icl)
        tokens_per_haystack = tok.count(sample_input_text) / incremental
        est = int((max_seq_length / tokens_per_haystack) * 3)
        num_noises = _search(lambda m: tok.count(generate_input_output(m, is_icl)[0]) + example_tokens + tokens_to_generate,
                             max_seq_length, incremental, max(est, incremental * 2), log)
        if num_noises is None:
            num_noises = incremental
        out = []
        for index in range(num_samples):
            used = num_noises
            while True:
                input_text, answer = generate_input_output(used, is_icl)
                if add_fewshot and icl_example is not None:
                    cutoff = input_text.index(T['template'][:20])
                    input_text = input_text[:cutoff] + randomize_icl(icl_example) + '\n' + input_text[cutoff:]
                length = tok.count(input_text) + tokens_to_generate
                if length <= max_seq_length:
                    break
                if used > incremental:
                    used -= incremental
                else:
                    raise RuntimeError("vt: no haystack fits")
            if final_output:
                out.append(_split(input_text, answer, length, T['answer_prefix'], index))
            else:
                out.append({'index': index, 'input': input_text, 'outputs': answer, 'length': length})
        return out

    icl = sys_vartrack(1, 500, None, False)[0]
    return sys_vartrack(n, L, icl, True)


# ---------------------------------------------------------------- common_words_extraction.py
def gen_cwe(L, n, seed, tg, tok, hay_dir, freq_cw, freq_ucw, num_cw, template, log, num_fewshot=1):
    rnd = random.Random(seed)
    words = sorted(list(set(_words("nounlist.txt") + _words("adjectivelist.txt") + _words("verblist.txt"))))
    random.Random(seed).shuffle(words)
    with open(os.path.join(hay_dir, "english_words.json")) as f:
        randle_words = list(json.load(f).values())

    def get_example(num_words, common_repeats=30, uncommon_repeats=3, common_nums=10):
        if num_words <= len(words):
            word_list_full = rnd.sample(words, num_words)
        else:
            word_list_full = rnd.sample(randle_words, num_words)
        common, uncommon = word_list_full[:common_nums], word_list_full[common_nums:]
        word_list = common * int(common_repeats) + uncommon * int(uncommon_repeats)
        random.Random(seed).shuffle(word_list)
        return ' '.join([f"{i + 1}. {word}" for i, word in enumerate(word_list)]), common

    def generate_input_output(num_words):
        few_shots = []
        if L < 4096:
            for _ in range(num_fewshot):
                few_shots.append(get_example(20, 3, 1, num_cw))
            context, answer = get_example(num_words, 6, 1, num_cw)
        else:
            for _ in range(num_fewshot):
                few_shots.append(get_example(40, 10, 3, num_cw))
            context, answer = get_example(num_words, freq_cw, freq_ucw, num_cw)
        for k in range(len(few_shots)):
            few_shots[k] = template.format(num_cw=num_cw, context=few_shots[k][0], query='') + ' ' + \
                ' '.join([f"{i + 1}. {word}" for i, word in enumerate(few_shots[k][1])])
        input_text = template.format(num_cw=num_cw, context=context, query='')
        return "\n".join(few_shots) + "\n" + input_text, answer

    incremental = 10
    sample_input_text, _ = generate_input_output(4096)
    tokens_per_words = tok.count(sample_input_text) / 4096
    est = int(L // tokens_per_words) * 2
    hi = min(max(est, incremental * 2), len(randle_words))     # the LSA patch's clamp (only binds above ~4M)
    num_words = _search(lambda m: tok.count(generate_input_output(m)[0]) + tg, L, incremental, hi, log)
    if num_words is None:
        num_words = incremental
    rows = []
    for index in range(n):
        used = num_words
        while True:
            input_text, answer = generate_input_output(used)
            length = tok.count(input_text) + tg
            if length <= L:
                break
            if used > incremental:
                used -= incremental
            else:
                raise RuntimeError("cwe: no list fits")
        rows.append(_split(input_text, answer, length, TEMPLATES['common_words_extraction']['answer_prefix'], index))
    return rows


# ---------------------------------------------------------------- freq_words_extraction.py
def gen_fwe(L, n, seed, tg, tok, hay_dir, alpha, template, log, coded_wordlen=6, vocab_size=-1):
    import numpy as np
    from scipy.special import zeta
    rnd = random.Random(seed)
    np.random.seed(seed)                      # seeded by the script; fwe draws nothing from it

    def generate_input_output(max_len, num_words=-1, vocab_size=2000, incremental=10):
        vocab = [''.join(rnd.choices(string.ascii_lowercase, k=coded_wordlen)) for _ in range(vocab_size)]
        while len(set(vocab)) < vocab_size:
            vocab.append(''.join(rnd.choices(string.ascii_lowercase, k=coded_wordlen)))
        vocab = sorted(list(set(vocab)))
        random.Random(seed).shuffle(vocab)
        vocab[0] = '...'

        def gen_text(num_words):
            k = np.arange(1, len(vocab) + 1)
            sampled_cnt = num_words * (k ** -alpha) / zeta(alpha)
            sampled_words = [[w] * zi for w, zi in zip(vocab, sampled_cnt.astype(int))]
            sampled_words = [x for wlst in sampled_words for x in wlst]
            random.Random(seed).shuffle(sampled_words)
            return template.format(context=' '.join(sampled_words), query=''), vocab[1:4]

        if num_words > 0:
            text, answer = gen_text(num_words)
            while tok.count(text) > max_len:
                num_words -= incremental
                text, answer = gen_text(num_words)
        else:
            num_words = max_len // coded_wordlen
            text, answer = gen_text(num_words)
            while tok.count(text) < max_len:
                num_words += incremental
                text, answer = gen_text(num_words)
            num_words -= incremental
        text, answer = gen_text(num_words)
        return text, answer, num_words

    max_seq_length = L - tg
    vs = max_seq_length // 50 if vocab_size == -1 else vocab_size
    _, _, num_example_words = generate_input_output(max_seq_length, vocab_size=vs, incremental=max_seq_length // 32)
    log(f"  words {num_example_words}")
    rows = []
    for index in range(n):
        input_text, answer, _ = generate_input_output(max_seq_length, num_words=num_example_words, vocab_size=vs,
                                                      incremental=max_seq_length // 32)
        length = tok.count(input_text) + tg
        rows.append(_split(input_text, answer, length, TEMPLATES['freq_words_extraction']['answer_prefix'], index))
    return rows


# ---------------------------------------------------------------- qa.py
def gen_qa(L, n, seed, tg, tok, hay_dir, dataset, template, log, pre_samples=0):
    rnd = random.Random(seed)
    if dataset == 'squad':
        data = json.load(open(os.path.join(hay_dir, "squad.json")))
        total_docs = sorted(list(set(p['context'] for d in data['data'] for p in d['paragraphs'])))
        docs_dict = {c: idx for idx, c in enumerate(total_docs)}
        QAS = []
        for d in data['data']:
            more_docs = [docs_dict[p['context']] for p in d['paragraphs']]
            for p in d['paragraphs']:
                for qas in p['qas']:
                    if not qas['is_impossible']:
                        QAS.append({'query': qas['question'], 'outputs': [a['text'] for a in qas['answers']],
                                    'context': [docs_dict[p['context']]],
                                    'more_context': [i for i in more_docs if i != docs_dict[p['context']]]})
    elif dataset == 'hotpotqa':
        data = json.load(open(os.path.join(hay_dir, "hotpotqa.json")))
        total_docs = sorted(list(set(f"{t}\n{''.join(p)}" for d in data for t, p in d['context'])))
        docs_dict = {c: idx for idx, c in enumerate(total_docs)}
        QAS = [{'query': d['question'], 'outputs': [d['answer']],
                'context': [docs_dict[f"{t}\n{''.join(p)}"] for t, p in d['context']]} for d in data]
    DOCS = total_docs

    def generate_input_output(index, num_docs):
        q = QAS[index]
        curr_docs, curr_more = q['context'], q.get('more_context', [])
        if num_docs < len(DOCS):
            if (num_docs - len(curr_docs)) > len(curr_more):
                addition_docs = [i for i, d in enumerate(DOCS) if i not in curr_docs + curr_more]
                all_docs = curr_docs + curr_more + rnd.sample(addition_docs, max(0, num_docs - len(curr_docs) - len(curr_more)))
            else:
                all_docs = curr_docs + rnd.sample(curr_more, num_docs - len(curr_docs))
            all_docs = [DOCS[idx] for idx in all_docs]
        else:
            repeats = (num_docs + len(DOCS) - 1) // len(DOCS)
            all_docs = (DOCS * repeats)[:num_docs]
        random.Random(seed).shuffle(all_docs)
        context = '\n\n'.join([f"Document {i + 1}:\n{d}" for i, d in enumerate(all_docs)])
        return template.format(context=context, query=q['query']), q['outputs']

    incremental = 10
    sample_input_text, _ = generate_input_output(0, incremental)
    tokens_per_doc = tok.count(sample_input_text) / incremental
    est = int((L / tokens_per_doc) * 3)
    num_docs = _search(lambda m: tok.count(generate_input_output(0, m)[0]) + tg, L, incremental,
                       max(est, incremental * 2), log)
    if num_docs is None:
        num_docs = incremental
    rows = []
    for index in range(n):
        used = num_docs
        while True:
            input_text, answer = generate_input_output(index + pre_samples, used)
            length = tok.count(input_text) + tg
            if length <= L:
                break
            if used > incremental:
                used -= incremental
            else:
                raise RuntimeError("qa: no document set fits")
        rows.append(_split(input_text, answer, length, TEMPLATES['qa']['answer_prefix'], index))
    return rows


GEN = {'niah': gen_niah, 'variable_tracking': gen_vt, 'common_words_extraction': gen_cwe,
       'freq_words_extraction': gen_fwe, 'qa': gen_qa}


def generate(task, L, n, seed, tok, hay_dir, log=print):
    kind, args = TASKS[task]
    tg = TEMPLATES[kind]['tokens_to_generate']
    rows = GEN[kind](L, n, seed, tg, tok, hay_dir, template=template_of(kind), log=log, **args)
    for r in rows:
        r['max_length'] = L
        r['tokens_to_generate'] = tg
    return rows


# ---------------------------------------------------------------- files
def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def rows_bytes(rows):
    return b''.join(json.dumps(r, ensure_ascii=True).encode() + b'\n' for r in rows)


def write_atomic(path, data):
    tmp = path + '.part'
    with open(tmp, 'wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def cell_name(task, L):
    return f"{task}_{L}"


def read_cell(setdir, task, L):
    with gzip.open(os.path.join(setdir, cell_name(task, L) + '.jsonl.gz'), 'rb') as f:
        return [json.loads(l) for l in f]


def env_info(tok_path, hay_dir):
    import numpy, tokenizers
    info = {'python': sys.version.split()[0], 'numpy': numpy.__version__, 'tokenizers': tokenizers.__version__,
            'tokenizer_sha256': sha256_file(tok_path),
            'words': {f: sha256_file(os.path.join(WORDS_DIR, f)) for f in ('nounlist.txt', 'adjectivelist.txt', 'verblist.txt')},
            'haystacks': {f: sha256_file(os.path.join(hay_dir, f)) for f in sorted(os.listdir(hay_dir))
                          if f.endswith('.json')},
            'rgen_sha256': sha256_file(os.path.abspath(__file__))}
    try:
        import nltk
        info['nltk'] = nltk.__version__
        from nltk.data import find
        info['punkt_tab'] = os.path.dirname(find('tokenizers/punkt_tab/english/'))
    except Exception:
        pass
    try:
        import scipy
        info['scipy'] = scipy.__version__
    except Exception:
        pass
    return info


def build_cell(setdir, task, L, n, seed, tok_path, hay_dir):
    t0 = time.time()
    tok = Tok(tok_path)
    rows = generate(task, L, n, seed, tok, hay_dir, log=lambda s: print(f"{task} {L}: {s.strip()}", flush=True))
    raw = rows_bytes(rows)
    path = os.path.join(setdir, cell_name(task, L) + '.jsonl.gz')
    gz = gzip.compress(raw, compresslevel=6, mtime=0)
    write_atomic(path, gz)
    prompt = sorted(r['length'] - r['tokens_to_generate'] for r in rows)
    return {'task': task, 'length': L, 'rows': len(rows), 'seed': seed, 'sha256': hashlib.sha256(raw).hexdigest(),
            'gz_sha256': hashlib.sha256(gz).hexdigest(), 'prompt_min': prompt[0], 'prompt_median': prompt[len(prompt) // 2],
            'prompt_max': prompt[-1], 'tokens_to_generate': rows[0]['tokens_to_generate'], 'seconds': round(time.time() - t0, 1)}


def _cell_job(a):
    return build_cell(*a)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('setdir'); b.add_argument('--tasks', required=True); b.add_argument('--lengths', required=True)
    b.add_argument('--samples', type=int, required=True); b.add_argument('--samples-at', default='')
    b.add_argument('--seed', type=int, default=42); b.add_argument('--tokenizer', required=True)
    b.add_argument('--haystacks', required=True); b.add_argument('--deps', default=os.environ.get('RULER_DEPS'))
    b.add_argument('--nltk-data', default=os.environ.get('NLTK_DATA')); b.add_argument('--jobs', type=int, default=1)
    b.add_argument('--name', default=None)
    c = sub.add_parser('check')
    c.add_argument('setdir')
    a = ap.parse_args()
    if a.cmd == 'check':
        m = json.load(open(os.path.join(a.setdir, 'manifest.json')))
        bad = 0
        for k, e in sorted(m['cells'].items()):
            p = os.path.join(a.setdir, k + '.jsonl.gz')
            ok = os.path.exists(p) and hashlib.sha256(gzip.open(p, 'rb').read()).hexdigest() == e['sha256']
            bad += not ok
            print(f"{k:28s} {e['rows']:4d} rows  prompt {e['prompt_min']}-{e['prompt_max']}  {'ok' if ok else 'BAD'}")
        print(f"{len(m['cells'])} cells, {bad} bad")
        sys.exit(1 if bad else 0)
    if a.deps:
        sys.path.insert(0, a.deps)
    if a.nltk_data:
        os.environ['NLTK_DATA'] = a.nltk_data
        import nltk
        nltk.data.path.insert(0, a.nltk_data)
    os.makedirs(a.setdir, exist_ok=True)
    tasks = a.tasks.split(',')
    for t in tasks:
        if t not in TASKS:
            sys.exit(f"unknown task {t}")
    lengths = [int(x) for x in a.lengths.split(',')]
    per = {int(k): int(v) for k, v in (x.split('=') for x in a.samples_at.split(',') if x)}
    mpath = os.path.join(a.setdir, 'manifest.json')
    m = json.load(open(mpath)) if os.path.exists(mpath) else {
        'format': FORMAT, 'name': a.name or os.path.basename(os.path.normpath(a.setdir)), 'upstream': UPSTREAM,
        'seed': a.seed, 'cells': {}}
    if m['seed'] != a.seed or m['format'] != FORMAT:
        sys.exit("manifest seed / format differ: a new set needs a new directory")
    env = env_info(a.tokenizer, a.haystacks)
    if 'env' in m and {k: v for k, v in m['env'].items() if k != 'punkt_tab'} != {k: v for k, v in env.items() if k != 'punkt_tab'}:
        sys.exit(f"environment differs from the manifest's:\n{json.dumps(m['env'])}\n{json.dumps(env)}")
    m['env'] = env
    todo = []
    for L in lengths:
        for t in tasks:
            n = per.get(L, a.samples)
            e = m['cells'].get(cell_name(t, L))
            if e and e['rows'] >= n and os.path.exists(os.path.join(a.setdir, cell_name(t, L) + '.jsonl.gz')):
                continue
            todo.append((a.setdir, t, L, n, a.seed, a.tokenizer, a.haystacks))
    todo.sort(key=lambda x: -x[2])
    print(f"{len(todo)} cells to build", flush=True)

    def done(e):
        m['cells'][cell_name(e['task'], e['length'])] = e
        write_atomic(mpath, json.dumps(m, indent=1, sort_keys=True).encode())
        print(f"{e['task']} {e['length']}: {e['rows']} rows, prompt {e['prompt_min']}-{e['prompt_max']}, {e['seconds']} s",
              flush=True)
    if a.jobs > 1:
        from multiprocessing import get_context
        with get_context('spawn').Pool(a.jobs, initializer=_init_worker, initargs=(a.deps, a.nltk_data)) as pool:
            for e in pool.imap_unordered(_cell_job, todo):
                done(e)
    else:
        for j in todo:
            done(build_cell(*j))


def _init_worker(deps, nltk_data):
    if deps:
        sys.path.insert(0, deps)
    if nltk_data:
        os.environ['NLTK_DATA'] = nltk_data
        import nltk
        nltk.data.path.insert(0, nltk_data)


if __name__ == '__main__':
    main()
