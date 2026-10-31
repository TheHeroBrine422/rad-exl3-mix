 What I'd do, in order:
 1. fp8 with the existing op (0.76 → 0.38 bpw). Measure ΔKLD on the 100k corpus for 2.00 and 4.00.
 2. int4 and u2 per-group from rtn (group=128, rule=search), with and without fwht128, about 0.2 and 0.1 bpw. That needs one small gather row in our plugin.
 3. Mixed precision by token frequency. A lookup has no Hessian; the closest equivalent is how often each token appears. We can count frequencies on the challenge's calib split (not the test corpus, to avoid overfitting), keep the
    frequent rows at higher precision, and push the long tail to 2 bits.
 4. Trellis (EXL3) on the embedding, only if steps 1–3 leave a real gap.