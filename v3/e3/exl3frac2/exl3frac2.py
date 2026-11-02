"""exl3frac2 -- the V3 kernel pass's fractional trellis quantizer (qtf2.cuh) as a side extension for the prebuilt
exllamav3 1.6.3 (with pack / unpack_trellis_frac that accept KA = 0): same call as quantize_tiles_frac, output bit-identical to the fork's kernel for every rate it has
(check: qtf2check.py), plus 1/8 steps K0.125-2.875 (incl. sub-1, KA = 0: salted 0-bit read) and the menu's
quarter / half steps. Built in two variants: "mul1" (plain codebook) and "i8" (EXL3_MUL1_I8, the int8-rounded one).
Use after exl3frac / exl3cb's install:  exl3frac2.install(SRC, "i8")  (SRC = <fork>/exllamav3/exllamav3_ext, for
codebook.cuh). Only quantize_tiles_frac and frac_k are swapped. Env TORCH_EXTENSIONS_DIR, PYTORCH_ROCM_ARCH, MAX_JOBS."""
import glob, os

HERE = os.path.dirname(os.path.abspath(__file__))


def frac_mask(n: int) -> int:
    return sum(1 << i for i in range(16) if ((i + 1) * n) // 16 - (i * n) // 16)


def frac_k(K):
    """(KA, MASK) for K = KA + n / 16, n even, KA >= 0 (K > 0); None for integer K."""
    if isinstance(K, int) or float(K).is_integer():
        return None
    ka = int(K)
    n = (K - ka) * 16
    assert K > 0 and abs(n - round(n)) < 1e-9 and round(n) % 2 == 0, f"fractional bitrate {K}: need K = KA + n / 16, n even"
    return ka, frac_mask(round(n))


def load(src: str, variant: str = "mul1", verbose: bool = False):
    assert variant in ("mul1", "i8"), variant
    import torch
    from torch.utils import cpp_extension as C
    from exllamav3.util import cuda_flags as F
    src = os.path.abspath(src)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", os.path.join(HERE, "build"))
    hip = bool(torch.version.hip)
    d = ["-DEXL3_MUL1_I8"] if variant == "i8" else []
    defs = os.environ.get("QTF2_DEFS", "").split()           # timing-only builds (qtf2bench.py): QTF2_NOBACK, QTF2_NOHIST
    d += [f"-D{x}" for x in defs]
    cflags, cuda = ["-O3"] + d, F.cuda_cflags(cuda_home=C.CUDA_HOME, hip=hip) + d
    if hip:
        assert os.environ.get("PYTORCH_ROCM_ARCH"), "set PYTORCH_ROCM_ARCH (e.g. gfx1201)"
        F.use_rocm_sdk_devel(C)
        inc = F.hip_include_flags(src)
        cflags += inc; cuda += inc
        F.patch_hip_ninja_file_writer(C)
    units = sorted(glob.glob(f"{HERE}/qtf2_inst_a*.cu"))
    assert len(units) == 8, f"run mkqtf2.py first ({len(units)} instance units)"
    # pack / unpack: the fork's frac.cu with KA >= 0 allowed (its kernels are generic in D(i); only the range check
    # refused KA = 0); generated here so the copy always matches the fork
    fs = open(os.path.join(src, "quant", "frac.cu")).read()
    a, b = "KA >= 1 && KA <= 7", "KA >= 0 && KA <= 7"
    assert fs.count(a) == 1, "frac.cu changed: re-check the KA = 0 patch"
    fs = fs.replace(a, b).replace('#include "../util.h"', '#include "util.h"').replace('#include "../util.cuh"', '#include "util.cuh"')
    gen = os.path.join(HERE, "qtf2_frac_gen.cu")
    if not os.path.exists(gen) or open(gen).read() != fs: open(gen, "w").write(fs)
    return C.load(name=f"exl3frac2_{variant}" + "".join("_" + x.lower() for x in defs), sources=[f"{HERE}/qtf2_bind.cu", gen] + units,
                  extra_include_paths=[HERE, os.path.join(src, "quant"), src], extra_cflags=cflags,
                  extra_cuda_cflags=cuda, verbose=verbose)


class _Ext:
    """exllamav3's extension (or exl3frac / exl3cb's wrapper) with quantize_tiles_frac from exl3frac2"""
    def __init__(self, base, q): self._base, self._q = base, q
    NAMES = ("quantize_tiles_frac", "pack_trellis_frac", "unpack_trellis_frac")
    def __getattr__(self, name): return getattr(self._q if name in self.NAMES else self._base, name)


def install(src: str, variant: str = "mul1", verbose: bool = False):
    import exllamav3.modules.quant.exl3_lib.quantize as Q
    if isinstance(Q.ext, _Ext): Q.ext = Q.ext._base
    Q.ext = _Ext(Q.ext, load(src, variant, verbose))
    Q.frac_k = frac_k
    return Q
