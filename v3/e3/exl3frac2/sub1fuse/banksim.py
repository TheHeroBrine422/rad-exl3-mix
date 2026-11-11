# banksim.py -- #142: simulated LDS bank cost (max distinct dwords per bank, 64 banks) of the byte-sum table reads of one
# wave instruction, lanes = (a, b): state base + sa * a + sb * b, a < 2^A, b < 2^(5 - A), over bases and salts
C=0x83DCD12D
def bs(x): return sum((x>>(8*i))&255 for i in range(4))
def salt(i): return (0x9E37*((i&15)+1))&0xFFFF
def cost(addrs, nb=64):
    banks={}
    for a in addrs:
        d=a//4; banks.setdefault(d%nb,set()).add(d)
    return max(len(v) for v in banks.values())
def run(A, sa, sb):
    tot=n=0
    for pos in range(16):
        sl=salt(pos)
        for base in range(0, 65536, 4096+8):
            for h in (0,1):
                ad=[2*bs(((((base + sa*(l & ((1<<A)-1)) + sb*(l >> A)) % 65536 + h) ^ sl)*C) & 0xffffffff) for l in range(32)]
                tot+=cost(ad); n+=1
    return round(tot/n,2)
for A, sa, sb in [(0,1,1024),(5,1024,1),(2,8,1024),(3,8,1024),(1,8,1024),(2,8,2048),(2,2,1024),(3,2,1024),(2,4,1024),(4,8,8192),(3,8,4096),(2,8,4096),(2,8,256),(1,2,1024),(3,4,1024)]:
    print(A, sa, sb, run(A, sa, sb))
