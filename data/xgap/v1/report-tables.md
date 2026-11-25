### cells: budget vs final (xgap-v1)

| cell (by source registry) | budget | xl budget | docs final | tokens final |
|---|---|---|---|---|
| long-docs | 23,437,500 | 0 | 934 | 23,343,647 |
| code-langs | 8,352,232 | 0 | 2,545 | 8,254,425 |
| multilingual-text | 11,894,000 | 0 | 11,172 | 11,583,306 |
| multilingual-chat | 5,000,000 | 0 | 10,869 | 4,826,010 |
| long-chat | 5,000,000 | 0 | 975 | 4,801,446 |
| tools-other | 3,300,000 | 0 | 4,476 | 2,901,933 |
| agent-sessions | 13,800,000 | 3,000,000 | 263 | 16,994,842 |

### cells as tagged (a code file of 8k+ tokens is a long-docs document)

| cell | docs | tokens |
|---|---|---|
| agent-sessions | 263 | 16,994,842 |
| code-langs | 2,403 | 6,279,476 |
| long-chat | 975 | 4,801,446 |
| long-docs | 1,076 | 25,318,596 |
| multilingual-chat | 10,869 | 4,826,010 |
| multilingual-text | 11,172 | 11,583,306 |
| tools-other | 4,476 | 2,901,933 |

### underfilled sources (< 85 % of budget, final tokens after dedup)

| source | final | budget | % | stopped |
|---|---|---|---|---|
| tools--hermes-func | 501,902 | 600,000 | 84 % | exhausted |
| tools--hermes-multi | 201,885 | 400,000 | 50 % | budgets-full |

### long-docs by band and source category (tokens)

| category | 8-16k | 16-32k | 32-64k | 64-128k |
|---|---|---|---|---|
| books | 315,492 | 1,585,314 | 3,393,615 | 4,987,117 |
| code | 1,189,429 | 785,520 | 0 | 0 |
| encyclopedic | 1,216,726 | 629,217 | 0 | 0 |
| legal | 624,815 | 633,322 | 327,393 | 281,758 |
| science | 1,560,897 | 2,198,451 | 1,857,006 | 616,829 |
| textbook | 628,658 | 628,044 | 573,850 | 355,575 |
| web | 929,568 | 0 | 0 | 0 |

### code languages (code-langs = files 1k-8k tokens; long = 8k+ files, inside long-docs)

| lang | docs 1-8k | tokens | docs 8k+ | tokens |
|---|---|---|---|---|
| c | 75 | 221,122 | 5 | 74,248 |
| cpp | 79 | 223,546 | 5 | 66,271 |
| csharp | 89 | 212,273 | 5 | 75,537 |
| dart | 97 | 226,638 | 6 | 85,353 |
| elixir | 97 | 221,859 | 6 | 77,490 |
| erlang | 64 | 224,451 | 6 | 82,776 |
| fortran | 72 | 224,936 | 6 | 87,073 |
| go | 84 | 228,083 | 5 | 64,508 |
| haskell | 88 | 227,298 | 4 | 60,012 |
| java | 94 | 222,781 | 6 | 82,589 |
| javascript | 88 | 227,041 | 5 | 68,746 |
| julia | 88 | 226,834 | 5 | 65,801 |
| kotlin | 97 | 227,474 | 4 | 63,630 |
| lean | 64 | 223,066 | 6 | 83,456 |
| lua | 88 | 228,072 | 4 | 56,709 |
| ocaml | 72 | 218,319 | 3 | 46,589 |
| perl | 78 | 221,797 | 5 | 74,383 |
| php | 93 | 228,537 | 6 | 69,800 |
| python | 85 | 227,821 | 5 | 70,268 |
| r | 90 | 220,694 | 7 | 83,177 |
| ruby | 93 | 224,333 | 3 | 30,141 |
| rust | 75 | 226,661 | 5 | 77,479 |
| scala | 90 | 227,125 | 5 | 74,748 |
| shell | 99 | 220,607 | 5 | 67,290 |
| sql | 74 | 219,988 | 5 | 72,656 |
| swift | 111 | 225,798 | 6 | 78,871 |
| typescript | 108 | 228,888 | 5 | 70,761 |
| zig | 71 | 223,434 | 4 | 64,587 |

### human languages: tokens by (cell, lang), top 45 of 133

| cell / lang | docs | tokens |
|---|---|---|
| long-chat / en | 540 | 2,499,413 |
| multilingual-chat / ru | 1465 | 1,044,469 |
| multilingual-text / zh | 988 | 854,701 |
| multilingual-text / ja | 985 | 853,788 |
| multilingual-chat / es | 2110 | 755,551 |
| multilingual-text / ar | 779 | 754,438 |
| multilingual-text / ru | 797 | 754,069 |
| long-chat / ru | 124 | 707,647 |
| multilingual-text / vi | 645 | 658,234 |
| multilingual-text / el | 538 | 657,342 |
| multilingual-text / fa | 1230 | 656,888 |
| multilingual-chat / zh | 1201 | 597,914 |
| multilingual-text / ur | 328 | 469,195 |
| multilingual-text / hy | 346 | 466,682 |
| multilingual-text / hi | 509 | 463,757 |
| multilingual-text / bn | 280 | 463,620 |
| multilingual-text / th | 520 | 463,322 |
| multilingual-text / uk | 355 | 462,927 |
| multilingual-text / ko | 528 | 461,827 |
| multilingual-text / ka | 371 | 460,987 |
| multilingual-text / he | 314 | 460,924 |
| multilingual-text / ta | 471 | 459,686 |
| long-chat / zh | 87 | 435,803 |
| multilingual-chat / vi | 471 | 423,463 |
| long-chat / es | 83 | 338,700 |
| multilingual-text / bg | 256 | 304,903 |
| multilingual-text / kk-Cyrl | 170 | 293,729 |
| multilingual-text / sr-Cyrl | 244 | 293,699 |
| multilingual-chat / fr | 369 | 186,439 |
| multilingual-text / sr | 367 | 175,649 |
| multilingual-text / kk | 19 | 174,802 |
| long-chat / pt | 21 | 174,756 |
| multilingual-text / ml | 25 | 173,525 |
| multilingual-text / mr | 32 | 172,592 |
| multilingual-text / te | 75 | 172,020 |
| multilingual-chat / yo | 200 | 167,053 |
| multilingual-chat / ar | 387 | 150,845 |
| multilingual-chat / id | 574 | 139,934 |
| long-chat / pl | 5 | 137,733 |
| multilingual-chat / de | 291 | 124,954 |
| multilingual-chat / pt | 316 | 116,417 |
| multilingual-chat / pa | 118 | 115,414 |
| long-chat / fr | 28 | 89,644 |
| multilingual-chat / ta | 241 | 82,541 |
| multilingual-chat / si | 247 | 77,121 |

### chat docs: thinking x has_system by origin

| origin | thinking | has_system | docs | tokens |
|---|---|---|---|---|
| calib | on | False | 973 | 1,932,840 |
| calib | on | True | 35 | 3,069,878 |
| xcorp-v1 | on | False | 11915 | 20,875,656 |
| xcorp-v1 | on | True | 1251 | 13,084,025 |
| xgap-v1 | off | False | 13456 | 16,090,238 |
| xgap-v1 | off | True | 3027 | 4,847,720 |
| xgap-v1 | on | False | 100 | 8,586,273 |

### agent sessions by source: docs, tokens, docs with any inferred tool definition

| source | docs | tokens | docs w/ inferred tools |
|---|---|---|---|
| AlinCiocan__fable-5-claude-code-traces | 12 | 510,460 | 10 |
| Infatoshi__kernelbench-hard-traces | 23 | 2,787,274 | 14 |
| Infatoshi__kernelbench-mega-traces | 14 | 1,563,234 | 13 |
| armand0e__claude-fable-5-claude-code | 33 | 1,993,398 | 24 |
| championswimmer__pi-coding-sessions | 11 | 455,275 | 9 |
| nemotron-swe | 23 | 1,196,474 | 0 |
| swe-rebench-openhands | 19 | 1,027,683 | 0 |
| swe-smith | 41 | 984,301 | 0 |
| thomasmustier__pi-extensions-sessions | 13 | 799,080 | 11 |
| thomasmustier__pi-for-excel-sessions | 19 | 1,352,759 | 15 |
| thomasmustier__pi-mono-sessions | 19 | 973,884 | 4 |
| thomasmustier__pi-nes-sessions | 4 | 101,922 | 1 |
| thomasmustier__pine-of-glass-sessions | 12 | 1,104,298 | 7 |
| thomasmustier__pinet-sessions | 5 | 522,899 | 3 |
| trace-commons__agent-traces | 15 | 1,621,901 | 14 |

### harness tools with REAL definitions (sessions using them)

| format | tool | sessions |
|---|---|---|
| claude-code | Read | 88 |
| claude-code | Bash | 86 |
| claude-code | Edit | 86 |
| claude-code | Write | 80 |
| pi | read | 71 |
| pi | bash | 70 |
| claude-code | TaskStop | 30 |
| pi | write | 27 |
| claude-code | TaskCreate | 22 |
| claude-code | TaskUpdate | 22 |
| pi | edit | 20 |
| claude-code | WebFetch | 15 |
| claude-code | Grep | 14 |
| claude-code | AskUserQuestion | 11 |
| claude-code | Glob | 9 |
| claude-code | TodoWrite | 9 |
| claude-code | WebSearch | 6 |
| claude-code | Agent | 6 |
| claude-code | ScheduleWakeup | 3 |
| pi | grep | 2 |

### most common INFERRED tools (sessions)

| format | tool | sessions |
|---|---|---|
| claude-code | ToolSearch | 31 |
| claude-code | TaskOutput | 22 |
| pi | edit | 20 |
| claude-code | PowerShell | 19 |
| claude-code | Skill | 13 |
| claude-code | Monitor | 12 |
| pi | subagent | 8 |
| pi | pinet | 6 |
| pi | oracle | 6 |
| pi | bash | 5 |
| pi | fetch_content | 4 |
| pi | web_search | 4 |
| pi | get_search_content | 3 |
| pi | shell_command | 3 |
| pi | Task | 3 |
| claude-code | ExitPlanMode | 2 |
| claude-code | SendUserMessage | 2 |
| claude-code | edit | 2 |
| claude-code | glob | 2 |
| claude-code | grep | 2 |
| claude-code | read | 2 |
| claude-code | shell | 2 |
| claude-code | mcp__Claude_Preview__preview_console_logs | 2 |
| claude-code | mcp__Claude_Preview__preview_eval | 2 |
| claude-code | mcp__Claude_Preview__preview_screenshot | 2 |

### agent sessions: xl / pieces / whole

| kind | docs |
|---|---|
| xl (whole, > 128k) | 14 |
| pieces | 39 |
| whole <= 128k | 210 |

### dedup

| step | count |
|---|---|
| rejected at build vs calib / public / xcorp | {"lines:xcorp": 2620, "lines:calib": 286, "lines:public": 39, "exact:xcorp": 86, "exact:calib": 10, "exact:public": 1, "xgap-in-source": 24} |
| dropped in assembly (xgap vs xgap) | {"lines": 272} |
| candidate docs entering assembly | 31506 |

### holdout

| split | docs |
|---|---|
| train | 28114 |
| holdout | 3120 |

