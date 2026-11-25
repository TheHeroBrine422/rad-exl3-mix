### by origin

| key | docs | tokens |
|---|---:|---:|
| calib | 4789 | 10,259,331 |
| xcorp-v1 | 53587 | 96,107,719 |
| xgap-v1 | 1005 | 4,985,415 |

### origin x cell x category

| key | docs | tokens |
|---|---:|---:|
| ('calib', '-', 'agentic') | 27 | 1,089,833 |
| ('calib', '-', 'books') | 168 | 559,476 |
| ('calib', '-', 'chat') | 374 | 704,539 |
| ('calib', '-', 'code') | 1065 | 1,007,333 |
| ('calib', '-', 'encyclopedic') | 259 | 301,326 |
| ('calib', '-', 'legal') | 40 | 105,423 |
| ('calib', '-', 'long') | 7 | 421,934 |
| ('calib', '-', 'long-agentic') | 8 | 1,980,045 |
| ('calib', '-', 'math') | 319 | 502,916 |
| ('calib', '-', 'multilingual') | 588 | 656,900 |
| ('calib', '-', 'qa') | 380 | 300,166 |
| ('calib', '-', 'reasoning') | 99 | 714,092 |
| ('calib', '-', 'science') | 77 | 604,894 |
| ('calib', '-', 'tools') | 497 | 410,310 |
| ('calib', '-', 'web') | 881 | 900,144 |
| ('xcorp-v1', '-', 'agentic') | 300 | 12,123,514 |
| ('xcorp-v1', '-', 'books') | 1315 | 6,612,569 |
| ('xcorp-v1', '-', 'chat') | 6115 | 8,401,980 |
| ('xcorp-v1', '-', 'code') | 7908 | 12,004,390 |
| ('xcorp-v1', '-', 'encyclopedic') | 3640 | 3,600,370 |
| ('xcorp-v1', '-', 'legal') | 379 | 1,202,415 |
| ('xcorp-v1', '-', 'long') | 164 | 4,844,177 |
| ('xcorp-v1', '-', 'math') | 3692 | 6,005,716 |
| ('xcorp-v1', '-', 'multilingual') | 7419 | 7,475,586 |
| ('xcorp-v1', '-', 'qa') | 4795 | 3,600,170 |
| ('xcorp-v1', '-', 'reasoning') | 1173 | 8,416,798 |
| ('xcorp-v1', '-', 'science') | 811 | 7,212,121 |
| ('xcorp-v1', '-', 'tools') | 5506 | 3,807,879 |
| ('xcorp-v1', '-', 'web') | 10370 | 10,800,034 |
| ('xgap-v1', 'agent-sessions', 'agentic') | 25 | 2,020,060 |
| ('xgap-v1', 'code-langs', 'code') | 56 | 150,467 |
| ('xgap-v1', 'long-chat', 'chat') | 17 | 133,901 |
| ('xgap-v1', 'long-docs', 'books') | 10 | 543,484 |
| ('xgap-v1', 'long-docs', 'code') | 43 | 780,697 |
| ('xgap-v1', 'long-docs', 'encyclopedic') | 4 | 55,077 |
| ('xgap-v1', 'long-docs', 'legal') | 5 | 181,810 |
| ('xgap-v1', 'long-docs', 'science') | 11 | 299,287 |
| ('xgap-v1', 'long-docs', 'textbook') | 4 | 177,439 |
| ('xgap-v1', 'long-docs', 'web') | 5 | 53,338 |
| ('xgap-v1', 'multilingual-chat', 'chat') | 323 | 132,923 |
| ('xgap-v1', 'multilingual-text', 'multilingual') | 375 | 380,289 |
| ('xgap-v1', 'tools-other', 'tools') | 127 | 76,643 |

### tokens by origin x cell x length band

| origin / cell | <1k | 1-2k | 2-4k | 4-8k | 8-16k | 16-32k | 32-64k | 64-128k | 128k+ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| calib / - | 1,562,584 | 1,300,013 | 1,127,069 | 1,464,027 | 1,102,427 | 477,462 | 1,045,704 | 200,000 | 1,980,045 |
| xcorp-v1 / - | 16,776,444 | 16,931,517 | 11,543,109 | 18,453,025 | 14,523,017 | 4,526,736 | 11,329,347 | 2,024,524 | 0 |
| xgap-v1 / agent-sessions | 0 | 0 | 0 | 38,854 | 64,144 | 19,315 | 80,187 | 568,939 | 1,248,621 |
| xgap-v1 / code-langs | 0 | 33,717 | 72,291 | 44,459 | 0 | 0 | 0 | 0 | 0 |
| xgap-v1 / long-chat | 0 | 10,871 | 10,277 | 10,124 | 19,720 | 0 | 82,909 | 0 | 0 |
| xgap-v1 / long-docs | 0 | 0 | 0 | 0 | 414,714 | 736,676 | 286,908 | 652,834 | 0 |
| xgap-v1 / multilingual-chat | 78,568 | 49,855 | 4,500 | 0 | 0 | 0 | 0 | 0 | 0 |
| xgap-v1 / multilingual-text | 126,459 | 77,822 | 62,925 | 25,804 | 45,179 | 42,100 | 0 | 0 | 0 |
| xgap-v1 / tools-other | 60,955 | 15,688 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### xgap holdout by cell

| key | docs | tokens |
|---|---:|---:|
| ('agent-sessions', 'holdout') | 2 | 162,465 |
| ('agent-sessions', 'train') | 23 | 1,857,595 |
| ('code-langs', 'holdout') | 5 | 14,464 |
| ('code-langs', 'train') | 51 | 136,003 |
| ('long-chat', 'holdout') | 1 | 1,266 |
| ('long-chat', 'train') | 16 | 132,635 |
| ('long-docs', 'holdout') | 9 | 278,248 |
| ('long-docs', 'train') | 73 | 1,812,884 |
| ('multilingual-chat', 'holdout') | 33 | 12,012 |
| ('multilingual-chat', 'train') | 290 | 120,911 |
| ('multilingual-text', 'holdout') | 36 | 27,436 |
| ('multilingual-text', 'train') | 339 | 352,853 |
| ('tools-other', 'holdout') | 13 | 6,896 |
| ('tools-other', 'train') | 114 | 69,747 |

### xgap cell x lang

| key | docs | tokens |
|---|---:|---:|
| ('code-langs', 'Swift') | 3 | 6,062 |
| ('code-langs', 'c') | 1 | 7,318 |
| ('code-langs', 'cpp') | 3 | 7,030 |
| ('code-langs', 'csharp') | 2 | 9,574 |
| ('code-langs', 'go') | 2 | 8,163 |
| ('code-langs', 'haskell') | 2 | 7,105 |
| ('code-langs', 'java') | 1 | 7,762 |
| ('code-langs', 'javascript') | 3 | 7,509 |
| ('code-langs', 'julia') | 4 | 6,540 |
| ('code-langs', 'kotlin') | 2 | 5,735 |
| ('code-langs', 'lua') | 3 | 6,165 |
| ('code-langs', 'ocaml') | 1 | 5,624 |
| ('code-langs', 'php') | 3 | 5,932 |
| ('code-langs', 'python') | 3 | 6,062 |
| ('code-langs', 'r') | 3 | 6,189 |
| ('code-langs', 'ruby') | 2 | 6,534 |
| ('code-langs', 'rust') | 3 | 7,321 |
| ('code-langs', 'scala') | 4 | 6,865 |
| ('code-langs', 'shell') | 3 | 6,123 |
| ('code-langs', 'sql') | 2 | 8,277 |
| ('code-langs', 'typescript') | 3 | 6,337 |
| ('code-langs', 'zig') | 3 | 6,240 |
| ('multilingual-chat', 'Chinese') | 40 | 15,550 |
| ('multilingual-chat', 'Russian') | 44 | 53,165 |
| ('multilingual-chat', 'Vietnamese') | 42 | 4,860 |
| ('multilingual-chat', 'amh') | 1 | 108 |
| ('multilingual-chat', 'ar') | 1 | 71 |
| ('multilingual-chat', 'arb') | 1 | 65 |
| ('multilingual-chat', 'ary') | 1 | 46 |
| ('multilingual-chat', 'ca') | 2 | 548 |
| ('multilingual-chat', 'de') | 10 | 4,950 |
| ('multilingual-chat', 'ell') | 1 | 91 |
| ('multilingual-chat', 'es') | 51 | 16,835 |
| ('multilingual-chat', 'eu') | 1 | 255 |
| ('multilingual-chat', 'fr') | 5 | 1,015 |
| ('multilingual-chat', 'fra') | 3 | 172 |
| ('multilingual-chat', 'guj') | 3 | 801 |
| ('multilingual-chat', 'hau') | 1 | 93 |
| ('multilingual-chat', 'ind') | 1 | 756 |
| ('multilingual-chat', 'ita') | 1 | 151 |
| ('multilingual-chat', 'jpn') | 1 | 70 |
| ('multilingual-chat', 'kir') | 2 | 1,185 |
| ('multilingual-chat', 'mal') | 1 | 106 |
| ('multilingual-chat', 'mar') | 1 | 54 |
| ('multilingual-chat', 'nld') | 2 | 198 |
| ('multilingual-chat', 'npi') | 1 | 146 |
| ('multilingual-chat', 'nya') | 1 | 128 |
| ('multilingual-chat', 'pan') | 2 | 3,491 |
| ('multilingual-chat', 'pbt') | 1 | 178 |
| ('multilingual-chat', 'plt') | 5 | 483 |
| ('multilingual-chat', 'por') | 11 | 889 |
| ('multilingual-chat', 'pt-BR') | 4 | 2,568 |
| ('multilingual-chat', 'ru') | 9 | 3,291 |
| ('multilingual-chat', 'sin') | 12 | 2,632 |
| ('multilingual-chat', 'som') | 5 | 524 |
| ('multilingual-chat', 'spa') | 6 | 1,355 |
| ('multilingual-chat', 'sun') | 1 | 76 |
| ('multilingual-chat', 'tam') | 5 | 811 |
| ('multilingual-chat', 'tel') | 4 | 895 |
| ('multilingual-chat', 'th') | 1 | 146 |
| ('multilingual-chat', 'tur') | 5 | 571 |
| ('multilingual-chat', 'uk-UA') | 1 | 354 |
| ('multilingual-chat', 'ukr') | 1 | 173 |
| ('multilingual-chat', 'urd') | 1 | 89 |
| ('multilingual-chat', 'vi') | 1 | 360 |
| ('multilingual-chat', 'vie') | 3 | 2,228 |
| ('multilingual-chat', 'wol') | 1 | 102 |
| ('multilingual-chat', 'yor') | 7 | 6,735 |
| ('multilingual-chat', 'zh') | 7 | 1,815 |
| ('multilingual-chat', 'zho') | 2 | 92 |
| ('multilingual-chat', 'zsm') | 10 | 1,646 |
| ('multilingual-text', 'ar') | 9 | 4,769 |
| ('multilingual-text', 'arb_Arab') | 28 | 16,859 |
| ('multilingual-text', 'ben_Beng') | 12 | 8,273 |
| ('multilingual-text', 'bn') | 8 | 5,303 |
| ('multilingual-text', 'bul_Cyrl') | 13 | 8,337 |
| ('multilingual-text', 'cmn_Hani') | 11 | 19,087 |
| ('multilingual-text', 'el') | 9 | 5,081 |
| ('multilingual-text', 'ell_Grek') | 18 | 13,736 |
| ('multilingual-text', 'fa') | 10 | 4,773 |
| ('multilingual-text', 'fas_Arab') | 21 | 13,186 |
| ('multilingual-text', 'he') | 1 | 14,577 |
| ('multilingual-text', 'heb_Hebr') | 6 | 8,157 |
| ('multilingual-text', 'hi') | 1 | 18,418 |
| ('multilingual-text', 'hin_Deva') | 12 | 7,893 |
| ('multilingual-text', 'hy') | 3 | 5,049 |
| ('multilingual-text', 'hye_Armn') | 3 | 9,512 |
| ('multilingual-text', 'ja') | 8 | 5,222 |
| ('multilingual-text', 'jpn_Jpan') | 17 | 18,549 |
| ('multilingual-text', 'ka') | 14 | 4,962 |
| ('multilingual-text', 'kat_Geor') | 5 | 11,033 |
| ('multilingual-text', 'kaz_Cyrl') | 12 | 8,247 |
| ('multilingual-text', 'kk') | 1 | 23,682 |
| ('multilingual-text', 'ko') | 3 | 4,914 |
| ('multilingual-text', 'kor_Hang') | 12 | 8,188 |
| ('multilingual-text', 'ml') | 1 | 9,945 |
| ('multilingual-text', 'mr') | 5 | 4,766 |
| ('multilingual-text', 'ru') | 10 | 4,702 |
| ('multilingual-text', 'rus_Cyrl') | 23 | 16,157 |
| ('multilingual-text', 'sr') | 5 | 4,882 |
| ('multilingual-text', 'srp_Cyrl') | 5 | 8,077 |
| ('multilingual-text', 'ta') | 12 | 4,873 |
| ('multilingual-text', 'tam_Taml') | 8 | 8,304 |
| ('multilingual-text', 'te') | 7 | 4,666 |
| ('multilingual-text', 'th') | 7 | 4,654 |
| ('multilingual-text', 'tha_Thai') | 15 | 7,981 |
| ('multilingual-text', 'uk') | 5 | 5,204 |
| ('multilingual-text', 'ukr_Cyrl') | 9 | 8,341 |
| ('multilingual-text', 'ur') | 6 | 4,824 |
| ('multilingual-text', 'urd_Arab') | 2 | 7,891 |
| ('multilingual-text', 'vi') | 2 | 8,443 |
| ('multilingual-text', 'vie_Latn') | 7 | 13,807 |
| ('multilingual-text', 'zh') | 9 | 4,965 |

### chat docs: thinking x has_system

| key | docs | tokens |
|---|---:|---:|
| ('calib', 'on', False) | 973 | 1,932,840 |
| ('calib', 'on', True) | 35 | 3,069,878 |
| ('xcorp-v1', 'on', False) | 11915 | 20,875,656 |
| ('xcorp-v1', 'on', True) | 1251 | 13,084,025 |
| ('xgap-v1', 'off', False) | 394 | 1,003,819 |
| ('xgap-v1', 'off', True) | 85 | 163,291 |
| ('xgap-v1', 'on', False) | 13 | 1,196,417 |

### xgap length tag (xl = whole session > 128k)

| key | docs | tokens |
|---|---:|---:|
| - | 999 | 3,736,794 |
| xl | 6 | 1,248,621 |
