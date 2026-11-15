# KL-IQR-V1 entries (Perire / radiance PR #364) on t183 and t186n (#194, w177)

Scored with their build (`qc/qwen38-kl-guided-iqr` @ 3eec7eb6, ROCm 10.1, gfx1201) on box2, one radiance per card: bf16 KV, `--tp 1`, `--max-model-len 16384`, `--max-num-seqs 16`, `--kld-ref` on w164's golden-t183 / golden-t186n. Means over gen rows (gen.u8). Our incumbents: snapshot 2026-11-15T20:21:30Z (live: lead-srch/run/drv/state.jsonl, multi/v3b/results.jsonl)

## Headline (text-plane bits: the entries' stated bits; IQR-V4 and our containers on the same basis)

| entry | text-plane bpw | t183 KL (95 % doc CI) | vs IQR-V4 at same bpw (t183) | vs our V3 incumbent at same bpw (t183) | t186n KL (95 % doc CI) | vs IQR-V4 at same bpw (t186n) |
|---|---:|---:|---:|---:|---:|---:|
| KL-IQR-V1-3.00 (Nano) | 2.9980 | 0.02957 [0.02341, 0.03747] | -6.8 % (IQR-V4 0.03173) | +15.5 % (ours 0.02560) | 0.04337 [0.03587, 0.05427] | -33.4 % (IQR-V4 0.06509) |
| KL-IQR-V1-4.00 (Tiny) | 3.9982 | 0.00718 [0.00597, 0.00854] | -12.3 % (IQR-V4 0.00819) | +1.0 % (ours 0.00711) | 0.01355 [0.01025, 0.01933] | -21.6 % (IQR-V4 0.01728) |
| KL-IQR-V1-5.50 (Small) | 5.4989 | 0.00116 [0.00098, 0.00133] | -23.6 % (IQR-V4 0.00151) | -11.2 % (ours 0.00130) | 0.00197 [0.00154, 0.00275] | -41.9 % (IQR-V4 0.00339) |

Secondary, literal file-size metric (the entries' full file bytes, vision tower included; our incumbents are text-only, so this charges the entries for the vision tower):

| entry | file bpw | t183 KL | vs our V3 incumbent at the file bpw (t183) |
|---|---:|---:|---:|
| KL-IQR-V1-3.00 (Nano) | 3.2754 | 0.02957 | +59.9 % (ours 0.01849) |
| KL-IQR-V1-4.00 (Tiny) | 4.2755 | 0.00718 | +41.8 % (ours 0.00506) |
| KL-IQR-V1-5.50 (Small) | 5.7763 | 0.00116 | +16.5 % (ours 0.000992) |

## Sanity: their build against our reads of IQR-V4 iq3r

- t183: their build 0.01725 vs our refs-t183 0.01725: +0.043 % (gate ±3 %: PASS)
- t186n: their build 0.03467 vs w164's rows (our build) 0.03467: -0.010 %
- Size convention check: IQR-V4 iq3r's text-plane bpw from its container 3.4723 vs its stated 3.472 (w134); entries: KL-IQR-V1-3.00 2.9980 vs stated 2.9980, KL-IQR-V1-4.00 3.9982 vs stated 3.9980, KL-IQR-V1-5.50 5.4989 vs stated 5.4988

## Entries: sizes and sha256

| entry | file | bytes | sha256 (LFS oid, verified) | file bpw | text-plane bpw | stated bits | t183 KL | t186n KL |
|---|---|---:|---|---:|---:|---:|---:|---:|
| nano | `qwen3.8-27b-kl-nano.rad` | 11011899392 | `8e5f87015e0dac59…` | 3.2754 | 2.9980 | 2.9980 | 0.02957 | 0.04337 |
| tiny | `qwen3.8-27b-kl-tiny.rad` | 14374330368 | `e8a6815113892d7f…` | 4.2755 | 3.9982 | 3.9980 | 0.00718 | 0.01355 |
| small | `qwen3.8-27b-kl-small.rad` | 19420073984 | `cd7db855096c53fc…` | 5.7763 | 5.4989 | 5.4988 | 0.00116 | 0.00197 |

## IQR-V4 reference curve (text-plane bpw from its README and refs-t183; t186n from w164's rows, our build)

| IQR-V4 | bpw (w134) | t183 KL (refs-t183, our build) | t186n KL (w164 rows, our build) |
|---|---:|---:|---:|
| iq2r-l | 2.7970 | 0.04134 | n/a |
| iq2r | 2.4700 | 0.06785 | 0.13131 |
| iq3r-l | 3.8040 | 0.01033 | n/a |
| iq3r-s | 3.1570 | 0.02574 | n/a |
| iq3r | 3.4720 | 0.01725 | 0.03467 |
| iq4r-l | 4.7930 | 0.00312 | n/a |
| iq4r-s | 4.1420 | 0.00690 | n/a |
| iq4r | 4.4730 | 0.00483 | 0.00922 |
| iq5r-s | 5.0710 | 0.00226 | n/a |
| iq5r | 5.4330 | 0.00159 | 0.00363 |
| iq6r-s | 6.0630 | 0.000966 | n/a |
| iq6r | 6.4390 | 0.000797 | 0.00124 |
| iq7r | 7.4200 | 0.000502 | n/a |

## Our V3 incumbents on t183

| size | bpw (text-only container) | t183 KL | tag |
|---:|---:|---:|---|
| 1.25 | 1.2512 | 0.44256 | v3s183-1.25-thompson-00183 |
| 1.5 | 1.5012 | 0.26417 | v3s183-1.5-inc-00640 |
| 1.75 | 1.7512 | 0.16505 | v3s183-1.75-inc-00641 |
| 2.0 | 2.0012 | 0.10852 | v3s183-2-manual-01796 |
| 2.5 | 2.5012 | 0.05610 | v3s183-2.5-manual-01628 |
| 3.0 | 3.0012 | 0.02547 | v3s183-3-manual-01487 |
| 3.5 | 3.5012 | 0.01420 | v3s183-3.5-inc-01349 |
| 4.0 | 4.0012 | 0.00708 | v3s183-4-inc-01351 |
| 4.5 | 4.5012 | 0.00384 | v3s183-4.5-inc-01758 |
| 5.0 | 5.0012 | 0.00212 | v3s183-5-inc-01760 |
| 6.0 | 6.0012 | 0.000796 | v3s183-6-inc-01762 |

## Size bases

- **Text plane (primary).** Container tensors minus the vision tower (`v.*`) and the drafter: trunk, head and token embedding, bytes × 8 / 26,895,998,464. The entries' stated bits are this number (to 0.0003 bpw); so is IQR-V4's w134 bpw (3.472 from its container: 3.4724). This is the basis the board uses per IQR-V4's README ("bpw counts the language model … the drafter and the vision tower come on top").
- **File bpw (secondary).** Whole file bytes × 8 / 26,895,998,464: the task's literal metric. The entries carry a 921 MB bf16 vision tower (0.277 bpw), which adds to their file size and not to their stated bits. Our search containers are text-only, so their file bpw is already the text-plane bpw.
- The default metric in AGENTS (trunk + head over trunk + head parameters, embedding excluded) is not computed here.

## Caveats

- The IQR-V4 and incumbent curves are log-linear interpolations of single reads per size, unpaired with the entries. Ratios carry no interval; the 95 % intervals are the entries' own, per document (178 docs on t183; 21 docs with gen rows on t186n).
- t186n is a 21-document slice (32k gen positions): intervals are ±20–40 %. Read its ratios as a direction only.
- Our incumbents are the live search's snapshot; re-run `extract_incumbents.py` for a new snapshot.
- Header and vocab bytes: a container also holds about 11–14 MB outside its tensors (strings, vocab, directory). The entries' stated bits and IQR-V4's bpw leave them out; our incumbents' bpw (file bytes, from the search) includes them, about +0.004 bpw (our V2 3.00: 13.7 MB). That biases the vs-ours ratios by about +1 % against the entries. Not corrected: the search containers are not on box2.
- t186n nano: IQR-V4 has no t186n read between 2.47 and 3.47 bpw, so the nano comparison interpolates across one bit.
