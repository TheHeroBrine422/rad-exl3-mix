#!/usr/bin/env python3
"""prebuild-mad.py [CONFIG] -- build rxq's run-time kernels for the mad codebook (exl3frac + v3/e3/exl3hash: exl3cb i8 and
exl3frac2 mul1 with EXL3_CB_HASH, A / B from the settings' [quant.codebook.mad]) without a card, and print the code
stamp the ladder's jobs will carry. Env as a job (tools/env-<host>.sh): PYTHONPATH, RXQ_RXM, RXQ_EXL3_SRC,
TORCH_EXTENSIONS_DIR, PYTORCH_ROCM_ARCH, ROCM_HOME. (#177-03, w151; tools/prebuild.py's mad twin)"""
import sys, time
t = time.time()
from rxq import config as C, engine
cfg = C.load(sys.argv[1] if len(sys.argv) > 1 else None)
Q = engine.setup("cpu", codebook="mad", qtf2=True, mad=C.mad_params(cfg))
print(time.strftime("%FT%TZ", time.gmtime()), "mad built in", round(time.time() - t), "s; settings", C.settings_stamp(cfg),
      "; code stamp", engine.code_stamp(Q), "; Q", Q.__file__, flush=True)
