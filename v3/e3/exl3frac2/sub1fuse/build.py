# build.py TREE VARIANT -- #142: build exl3frac2 from TREE (its own build dir) without importing the exllamav3 package
# (W108_STUB, SRC from env-dad1.sh)
import os, sys, time
sys.path[:0] = [os.environ["W108_STUB"], sys.argv[1]]
import exl3frac2
t = time.time(); exl3frac2.load(os.environ["SRC"], sys.argv[2], verbose=False); print("built", sys.argv[1:], round(time.time() - t), "s", flush=True)
