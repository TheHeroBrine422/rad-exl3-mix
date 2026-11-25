### by origin

| key | docs | tokens |
|---|---:|---:|
| calib | 4789 | 10,259,331 |
| xcorp-v1 | 53587 | 96,107,719 |
| xgap-v1 | 31234 | 72,705,609 |

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
| ('xgap-v1', 'agent-sessions', 'agentic') | 263 | 16,994,842 |
| ('xgap-v1', 'code-langs', 'code') | 2403 | 6,279,476 |
| ('xgap-v1', 'long-chat', 'chat') | 975 | 4,801,446 |
| ('xgap-v1', 'long-docs', 'books') | 211 | 10,281,538 |
| ('xgap-v1', 'long-docs', 'code') | 142 | 1,974,949 |
| ('xgap-v1', 'long-docs', 'encyclopedic') | 140 | 1,845,943 |
| ('xgap-v1', 'long-docs', 'legal') | 102 | 1,867,288 |
| ('xgap-v1', 'long-docs', 'science') | 289 | 6,233,183 |
| ('xgap-v1', 'long-docs', 'textbook') | 108 | 2,186,127 |
| ('xgap-v1', 'long-docs', 'web') | 84 | 929,568 |
| ('xgap-v1', 'multilingual-chat', 'chat') | 10869 | 4,826,010 |
| ('xgap-v1', 'multilingual-text', 'multilingual') | 11172 | 11,583,306 |
| ('xgap-v1', 'tools-other', 'tools') | 4476 | 2,901,933 |

### tokens by origin x cell x length band

| origin / cell | <1k | 1-2k | 2-4k | 4-8k | 8-16k | 16-32k | 32-64k | 64-128k | 128k+ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| calib / - | 1,562,584 | 1,300,013 | 1,127,069 | 1,464,027 | 1,102,427 | 477,462 | 1,045,704 | 200,000 | 1,980,045 |
| xcorp-v1 / - | 16,776,444 | 16,931,517 | 11,543,109 | 18,453,025 | 14,523,017 | 4,526,736 | 11,329,347 | 2,024,524 | 0 |
| xgap-v1 / agent-sessions | 0 | 0 | 0 | 87,379 | 298,136 | 1,052,816 | 3,980,435 | 8,480,809 | 3,095,267 |
| xgap-v1 / code-langs | 2,048 | 1,767,220 | 2,150,394 | 2,359,814 | 0 | 0 | 0 | 0 | 0 |
| xgap-v1 / long-chat | 0 | 373,678 | 1,237,714 | 1,401,070 | 571,810 | 407,397 | 374,568 | 435,209 | 0 |
| xgap-v1 / long-docs | 0 | 0 | 0 | 0 | 6,465,585 | 6,459,868 | 6,151,864 | 6,241,279 | 0 |
| xgap-v1 / multilingual-chat | 2,955,926 | 1,045,971 | 802,857 | 21,256 | 0 | 0 | 0 | 0 | 0 |
| xgap-v1 / multilingual-text | 3,508,465 | 2,032,255 | 1,927,932 | 1,589,004 | 1,402,823 | 1,122,827 | 0 | 0 | 0 |
| xgap-v1 / tools-other | 2,118,290 | 688,181 | 95,462 | 0 | 0 | 0 | 0 | 0 | 0 |

### xgap holdout by cell

| key | docs | tokens |
|---|---:|---:|
| ('agent-sessions', 'holdout') | 26 | 1,419,060 |
| ('agent-sessions', 'train') | 237 | 15,575,782 |
| ('code-langs', 'holdout') | 240 | 621,735 |
| ('code-langs', 'train') | 2163 | 5,657,741 |
| ('long-chat', 'holdout') | 98 | 453,965 |
| ('long-chat', 'train') | 877 | 4,347,481 |
| ('long-docs', 'holdout') | 107 | 2,457,144 |
| ('long-docs', 'train') | 969 | 22,861,452 |
| ('multilingual-chat', 'holdout') | 1086 | 475,883 |
| ('multilingual-chat', 'train') | 9783 | 4,350,127 |
| ('multilingual-text', 'holdout') | 1116 | 1,178,897 |
| ('multilingual-text', 'train') | 10056 | 10,404,409 |
| ('tools-other', 'holdout') | 447 | 291,397 |
| ('tools-other', 'train') | 4029 | 2,610,536 |

### xgap cell x lang

| key | docs | tokens |
|---|---:|---:|
| ('code-langs', 'c') | 75 | 221,122 |
| ('code-langs', 'cpp') | 79 | 223,546 |
| ('code-langs', 'csharp') | 89 | 212,273 |
| ('code-langs', 'dart') | 97 | 226,638 |
| ('code-langs', 'elixir') | 97 | 221,859 |
| ('code-langs', 'erlang') | 64 | 224,451 |
| ('code-langs', 'fortran') | 72 | 224,936 |
| ('code-langs', 'go') | 84 | 228,083 |
| ('code-langs', 'haskell') | 88 | 227,298 |
| ('code-langs', 'java') | 94 | 222,781 |
| ('code-langs', 'javascript') | 88 | 227,041 |
| ('code-langs', 'julia') | 88 | 226,834 |
| ('code-langs', 'kotlin') | 97 | 227,474 |
| ('code-langs', 'lean') | 64 | 223,066 |
| ('code-langs', 'lua') | 88 | 228,072 |
| ('code-langs', 'ocaml') | 72 | 218,319 |
| ('code-langs', 'perl') | 78 | 221,797 |
| ('code-langs', 'php') | 93 | 228,537 |
| ('code-langs', 'python') | 85 | 227,821 |
| ('code-langs', 'r') | 90 | 220,694 |
| ('code-langs', 'ruby') | 93 | 224,333 |
| ('code-langs', 'rust') | 75 | 226,661 |
| ('code-langs', 'scala') | 90 | 227,125 |
| ('code-langs', 'shell') | 99 | 220,607 |
| ('code-langs', 'sql') | 74 | 219,988 |
| ('code-langs', 'swift') | 111 | 225,798 |
| ('code-langs', 'typescript') | 108 | 228,888 |
| ('code-langs', 'zig') | 71 | 223,434 |
| ('multilingual-chat', 'acq') | 1 | 184 |
| ('multilingual-chat', 'af') | 3 | 929 |
| ('multilingual-chat', 'am') | 25 | 2,566 |
| ('multilingual-chat', 'ar') | 387 | 150,845 |
| ('multilingual-chat', 'ars') | 3 | 186 |
| ('multilingual-chat', 'ary') | 96 | 4,271 |
| ('multilingual-chat', 'bg') | 2 | 2,247 |
| ('multilingual-chat', 'bn') | 19 | 8,962 |
| ('multilingual-chat', 'ca') | 136 | 34,727 |
| ('multilingual-chat', 'ceb') | 11 | 967 |
| ('multilingual-chat', 'cs') | 6 | 5,539 |
| ('multilingual-chat', 'cy') | 6 | 4,415 |
| ('multilingual-chat', 'da') | 9 | 6,800 |
| ('multilingual-chat', 'de') | 291 | 124,954 |
| ('multilingual-chat', 'el') | 7 | 1,669 |
| ('multilingual-chat', 'eo') | 33 | 7,004 |
| ('multilingual-chat', 'es') | 2110 | 755,551 |
| ('multilingual-chat', 'et') | 3 | 335 |
| ('multilingual-chat', 'eu') | 48 | 9,224 |
| ('multilingual-chat', 'fa') | 46 | 20,359 |
| ('multilingual-chat', 'fi') | 26 | 5,757 |
| ('multilingual-chat', 'fil') | 22 | 2,725 |
| ('multilingual-chat', 'fr') | 369 | 186,439 |
| ('multilingual-chat', 'ga') | 22 | 4,642 |
| ('multilingual-chat', 'gu') | 82 | 51,635 |
| ('multilingual-chat', 'ha') | 60 | 11,367 |
| ('multilingual-chat', 'he') | 5 | 2,947 |
| ('multilingual-chat', 'hi') | 16 | 4,342 |
| ('multilingual-chat', 'ht') | 2 | 1,592 |
| ('multilingual-chat', 'hu') | 21 | 14,630 |
| ('multilingual-chat', 'id') | 574 | 139,934 |
| ('multilingual-chat', 'ig') | 30 | 7,036 |
| ('multilingual-chat', 'it') | 120 | 36,234 |
| ('multilingual-chat', 'ja') | 190 | 71,886 |
| ('multilingual-chat', 'jav') | 4 | 557 |
| ('multilingual-chat', 'ka') | 1 | 1,078 |
| ('multilingual-chat', 'kk') | 1 | 64 |
| ('multilingual-chat', 'kn') | 6 | 1,816 |
| ('multilingual-chat', 'ko') | 58 | 48,455 |
| ('multilingual-chat', 'ku') | 1 | 48 |
| ('multilingual-chat', 'ky') | 148 | 25,159 |
| ('multilingual-chat', 'la') | 46 | 18,000 |
| ('multilingual-chat', 'lt') | 17 | 7,629 |
| ('multilingual-chat', 'mg') | 224 | 25,494 |
| ('multilingual-chat', 'ml') | 34 | 9,308 |
| ('multilingual-chat', 'mn') | 1 | 671 |
| ('multilingual-chat', 'mr') | 69 | 11,518 |
| ('multilingual-chat', 'ms') | 195 | 31,136 |
| ('multilingual-chat', 'my') | 6 | 1,034 |
| ('multilingual-chat', 'ne') | 59 | 8,222 |
| ('multilingual-chat', 'nl') | 52 | 16,095 |
| ('multilingual-chat', 'nso') | 3 | 395 |
| ('multilingual-chat', 'ny') | 15 | 9,786 |
| ('multilingual-chat', 'pa') | 118 | 115,414 |
| ('multilingual-chat', 'pl') | 76 | 18,954 |
| ('multilingual-chat', 'ps') | 17 | 4,213 |
| ('multilingual-chat', 'pt') | 316 | 116,417 |
| ('multilingual-chat', 'ro') | 6 | 2,005 |
| ('multilingual-chat', 'ru') | 1465 | 1,044,469 |
| ('multilingual-chat', 'sd') | 5 | 3,910 |
| ('multilingual-chat', 'si') | 247 | 77,121 |
| ('multilingual-chat', 'sk') | 3 | 1,845 |
| ('multilingual-chat', 'sn') | 25 | 4,043 |
| ('multilingual-chat', 'so') | 121 | 31,735 |
| ('multilingual-chat', 'sq') | 5 | 273 |
| ('multilingual-chat', 'sr') | 2 | 155 |
| ('multilingual-chat', 'su') | 3 | 348 |
| ('multilingual-chat', 'sv') | 24 | 3,998 |
| ('multilingual-chat', 'sw') | 5 | 665 |
| ('multilingual-chat', 'ta') | 241 | 82,541 |
| ('multilingual-chat', 'te') | 137 | 69,214 |
| ('multilingual-chat', 'th') | 77 | 14,758 |
| ('multilingual-chat', 'tl') | 58 | 17,453 |
| ('multilingual-chat', 'tr') | 136 | 54,599 |
| ('multilingual-chat', 'uk') | 58 | 28,900 |
| ('multilingual-chat', 'und') | 32 | 5,492 |
| ('multilingual-chat', 'ur') | 16 | 8,790 |
| ('multilingual-chat', 'vi') | 471 | 423,463 |
| ('multilingual-chat', 'wo') | 42 | 6,622 |
| ('multilingual-chat', 'xh') | 11 | 3,165 |
| ('multilingual-chat', 'yo') | 200 | 167,053 |
| ('multilingual-chat', 'zh') | 1201 | 597,914 |
| ('multilingual-chat', 'zu') | 30 | 21,116 |
| ('multilingual-text', 'ar') | 779 | 754,438 |
| ('multilingual-text', 'bg') | 256 | 304,903 |
| ('multilingual-text', 'bn') | 280 | 463,620 |
| ('multilingual-text', 'el') | 538 | 657,342 |
| ('multilingual-text', 'fa') | 1230 | 656,888 |
| ('multilingual-text', 'he') | 314 | 460,924 |
| ('multilingual-text', 'hi') | 509 | 463,757 |
| ('multilingual-text', 'hy') | 346 | 466,682 |
| ('multilingual-text', 'ja') | 985 | 853,788 |
| ('multilingual-text', 'ka') | 371 | 460,987 |
| ('multilingual-text', 'kk') | 19 | 174,802 |
| ('multilingual-text', 'kk-Cyrl') | 170 | 293,729 |
| ('multilingual-text', 'ko') | 528 | 461,827 |
| ('multilingual-text', 'ml') | 25 | 173,525 |
| ('multilingual-text', 'mr') | 32 | 172,592 |
| ('multilingual-text', 'ru') | 797 | 754,069 |
| ('multilingual-text', 'sr') | 367 | 175,649 |
| ('multilingual-text', 'sr-Cyrl') | 244 | 293,699 |
| ('multilingual-text', 'ta') | 471 | 459,686 |
| ('multilingual-text', 'te') | 75 | 172,020 |
| ('multilingual-text', 'th') | 520 | 463,322 |
| ('multilingual-text', 'uk') | 355 | 462,927 |
| ('multilingual-text', 'ur') | 328 | 469,195 |
| ('multilingual-text', 'vi') | 645 | 658,234 |
| ('multilingual-text', 'zh') | 988 | 854,701 |

### chat docs: thinking x has_system

| key | docs | tokens |
|---|---:|---:|
| ('calib', 'on', False) | 973 | 1,932,840 |
| ('calib', 'on', True) | 35 | 3,069,878 |
| ('xcorp-v1', 'on', False) | 11915 | 20,875,656 |
| ('xcorp-v1', 'on', True) | 1251 | 13,084,025 |
| ('xgap-v1', 'off', False) | 13456 | 16,090,238 |
| ('xgap-v1', 'off', True) | 3027 | 4,847,720 |
| ('xgap-v1', 'on', False) | 100 | 8,586,273 |

### xgap length tag (xl = whole session > 128k)

| key | docs | tokens |
|---|---:|---:|
| - | 31220 | 69,610,342 |
| xl | 14 | 3,095,267 |
