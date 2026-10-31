#!/bin/bash
# isa.sh [KERNEL-REGEX] -- disassemble the dev build's gemm_nt kernel and count the hottest loop's instructions
P=${1:-ILi4ELi2ELi4ELb1ELi1E}
ssh-gpu.sh "cd /tmp/co && objcopy -O binary --only-section=.hip_fatbin /root/radiance-hero/plugin/b-card31-exl3r4d/CMakeFiles/exl3r4d.dir/exl3r4d.hip.o fb.bin && B=/opt/rocm/core-10.1/lib/llvm/bin; t=\$(\$B/clang-offload-bundler --list --type=o --input=fb.bin | grep gfx1201 | head -1); \$B/clang-offload-bundler --unbundle --type=o --targets=\$t --input=fb.bin --output=k.co && \$B/llvm-objdump -d --no-show-raw-insn k.co > k.s && python3 - '$P' <<'PY'
import re,sys,collections
txt=open('/tmp/co/k.s').read()
pat=sys.argv[1]
# split functions
funcs=re.split(r'\n(?=[0-9a-f]+ <)', txt)
f=[x for x in funcs if 'gemm_nt_kernel' in x.split('\n')[0] and re.search(pat, x.split('\n')[0])]
if not f: print('no kernel'); sys.exit()
body=f[0].split('\n')
print(body[0][:100], len(body),'lines')
ins=[]
base=int(body[0].split()[0],16)
for l in body[1:]:
    m=re.match(r'\s*(\S+)\s*(.*?)\s*//\s*([0-9A-Fa-f]+):',l)
    if not m: continue
    tg=re.search(r'<[^>]*\+0x([0-9a-f]+)>',l)
    ins.append((int(m.group(3),16),m.group(1),base+int(tg.group(1),16) if tg else None))
addr={a:i for i,(a,_,_) in enumerate(ins)}
best=None
for i,(a,op,t) in enumerate(ins):
    if (op.startswith('s_cbranch') or op=='s_branch') and t is not None and t<=a and t in addr:
        seg=ins[addr[t]:i+1]
        w=sum(1 for x in seg if 'wmma' in x[1])
        if best is None or w>best[0]: best=(w,seg)
if not best: print('no loop'); sys.exit()
seg=best[1]; c=collections.Counter()
for _,op,_ in seg:
    k=('v_wmma' if 'wmma' in op else 'ds_' if op.startswith('ds_') else 'buffer/global' if op.startswith(('buffer_','global_')) else
       's_waitcnt' if op.startswith('s_wait') else 'salu' if op.startswith('s_') else 'valu' if op.startswith('v_') else op)
    c[k]+=1
print('hot loop: %d instructions'%len(seg), dict(c))
ops=collections.Counter(op for _,op,_ in seg if op.startswith('v_') and 'wmma' not in op)
print('valu ops:', ops.most_common(14))
PY"
