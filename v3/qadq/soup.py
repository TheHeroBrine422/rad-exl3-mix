import sys, os, glob, torch
from safetensors.torch import load_file, save_file
out = sys.argv[1]; srcs = sys.argv[2:]
os.makedirs(out + "/scales", exist_ok=True)
for f in sorted(glob.glob(srcs[0] + "/scales/*.safetensors")):
    nm = os.path.basename(f); T = [load_file(s + "/scales/" + nm) for s in srcs]
    save_file({k: torch.exp(sum(torch.log(t[k].double()) for t in T) / len(T)).to(T[0][k].dtype) for k in T[0]}, out + "/scales/" + nm)
open(out + "/mon.json", "w").write('{"soup": %r}' % srcs); open(out + "/.done", "w").close()
print(out, len(os.listdir(out + "/scales")))
