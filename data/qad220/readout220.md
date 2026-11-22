# w220 GPU gates

| gate | verdict |
|---|---|
| d | PASS |
| b (eager) | PASS (bitwise identical) |
| b (compiled) | PASS (bitwise identical) |
| a 0.25 M | PASS (ratio 1.0050 [0.9951, 1.0194]) |
| a 0.5 M | PASS (ratio 0.9846 [0.9654, 1.0039]) |


## d. defaults = qad200 (2 steps, eager): **IDENTICAL**
```
step 1: train_kl 0.04573589688264076 / 0.04573589688264076  gnorm 0.023639 / 0.023639  IDENTICAL
step 2: train_kl 0.05168729126774262 / 0.05168729126774262  gnorm 0.022371 / 0.022371  IDENTICAL
ALL IDENTICAL
720 tensors, 0 differ, max |diff| 0.000e+00
```

## b. frozen prefix (eager; layers 32-63 trainable; frozen-prefix run vs the full-backward run, same trainable set)
```
360 vectors: max |diff| 0.000e+00, max rel diff (to the vector's max |g|) 0.000e+00, median over vectors 0.000e+00
batch 0: KL/row 0.04573589688264076 / 0.04573589688264076  gnorm 1.14388216e-02 / 1.14388216e-02  rel 0.00e+00
batch 1: KL/row 0.051920380752894284 / 0.051920380752894284  gnorm 1.39129761e-02 / 1.39129761e-02  rel 0.00e+00
```
run-to-run floor (gb-pre vs its repeat gb-pre-r):
```
360 vectors: max |diff| 0.000e+00, max rel diff (to the vector's max |g|) 0.000e+00, median over vectors 0.000e+00
batch 0: KL/row 0.04573589688264076 / 0.04573589688264076  gnorm 1.14388216e-02 / 1.14388216e-02  rel 0.00e+00
batch 1: KL/row 0.051920380752894284 / 0.051920380752894284  gnorm 1.39129761e-02 / 1.39129761e-02  rel 0.00e+00
```

## b. frozen prefix (compiled; layers 32-63 trainable; frozen-prefix run vs the full-backward run, same trainable set)
```
360 vectors: max |diff| 0.000e+00, max rel diff (to the vector's max |g|) 0.000e+00, median over vectors 0.000e+00
batch 0: KL/row 0.04573589688264076 / 0.04573589688264076  gnorm 1.14390312e-02 / 1.14390312e-02  rel 0.00e+00
batch 1: KL/row 0.051920380752894284 / 0.051920380752894284  gnorm 1.39106634e-02 / 1.39106634e-02  rel 0.00e+00
```
run-to-run floor (gbc-pre vs its repeat gbc-pre-r):
```
360 vectors: max |diff| 0.000e+00, max rel diff (to the vector's max |g|) 0.000e+00, median over vectors 0.000e+00
batch 0: KL/row 0.04573589688264076 / 0.04573589688264076  gnorm 1.14390312e-02 / 1.14390312e-02  rel 0.00e+00
batch 1: KL/row 0.051920380752894284 / 0.051920380752894284  gnorm 1.39106634e-02 / 1.39106634e-02  rel 0.00e+00
```

## c. speed (2.50 student, V 24 x Q 1024 plan = ~850 gen rows a step; other cards busy at the same time as in production)
```
/data/q27/w220/runs/sp-eager: 98.0 s / step (median of 3: 95.2, 98.0, 98.8), 223 gen rows/s, peak 28.4 GiB
/data/q27/w220/runs/sp-comp: 88.9 s / step (median of 3: 86.0, 88.9, 90.5), 246 gen rows/s, peak 28.4 GiB
/data/q27/w220/runs/sp-comp-32: 61.3 s / step (median of 3: 59.3, 61.3, 62.8), 356 gen rows/s, peak 28.0 GiB
/data/q27/w220/runs/sp-comp-48: 45.5 s / step (median of 3: 43.7, 45.5, 46.9), 479 gen rows/s, peak 27.8 GiB
/data/q27/w220/runs/sp-eager-32: 65.2 s / step (median of 3: 63.1, 65.2, 66.3), 335 gen rows/s, peak 28.0 GiB
```

| config | s / step | speed-up vs eager |
|---|---|---|
| sp-eager | 98.0 | 1.00x |
| sp-comp | 88.9 | 1.10x |
| sp-comp-32 | 61.3 | 1.60x |
| sp-comp-48 | 45.5 | 2.15x |
| sp-eager-32 | 65.2 | 1.50x |

## a. compiled 25C vs eager 25A at 0.25 M (j217, paired per document; ratio compiled / eager): **1.0050** [0.9951, 1.0194] (docs better 82 / worse 84); KL 0.045644 vs 0.045418; **PASS** (|ratio - 1| <= 2 %); vs the unhealed release: compiled 0.9685, eager 0.9637

## a. compiled 25C vs eager 25A at 0.5 M (j217, paired per document; ratio compiled / eager): **0.9846** [0.9654, 1.0039] (docs better 87 / worse 79); KL 0.046219 vs 0.046944; **PASS** (|ratio - 1| <= 2 %); vs the unhealed release: compiled 0.9807, eager 0.9961
