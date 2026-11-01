"""exl3cb -- exllamav3's trellis search with an int8- or int4-rounded mul1 codebook, on the prebuilt exllamav3 1.6.3.

V3's int compute multiplies rounded codebook values: int8 (E4's nfq_wdec_i8: rint(v / 4 k_inv), +-127; free to K6
when the trellis is searched with it, E3b) or the int4 grid (E3: 16 half-integer levels, for W4A4 at K <= 3). The
search must see the rounded values, or LDLQ feeds back the wrong error. Branch rxm-frac-quarter makes the rounding a
compile option of codebook.cuh; this builds quantize.cu + the integer- and fractional-K instance units with it as a
side extension (exl3cb_i8 / exl3cb_i4) and install() swaps quantize_tiles / quantize_tiles_scratch /
quantize_tiles_frac into exllamav3's quantizer (with exl3frac's frac_k: quarter steps work too). Packing and storage
are unchanged: the indices are ordinary EXL3 indices; only a decoder that rounds the same way reads them as intended.
Use: import exl3cb; exl3cb.install(SRC, "i8")  with SRC = <fork>/exllamav3/exllamav3_ext.
Env: TORCH_EXTENSIONS_DIR (default <here>/build), PYTORCH_ROCM_ARCH, MAX_JOBS, ROCM_HOME (GPU box: /opt/rocm/core-10.1)."""
import glob, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exl3frac"))
import exl3frac


def load(src: str, variant: str, verbose: bool = False):
    assert variant in ("i8", "i4"), variant
    import torch
    from torch.utils import cpp_extension as C
    from exllamav3.util import cuda_flags as F
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.abspath(src)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", os.path.join(here, "build"))
    hip = bool(torch.version.hip)
    d = [f"-DEXL3_MUL1_{variant.upper()}"]
    cflags, cuda = ["-O3"] + d, F.cuda_cflags(cuda_home=C.CUDA_HOME, hip=hip) + d
    if hip:
        assert os.environ.get("PYTORCH_ROCM_ARCH"), "set PYTORCH_ROCM_ARCH (e.g. gfx1201)"
        F.use_rocm_sdk_devel(C)
        inc = F.hip_include_flags(src)
        cflags += inc; cuda += inc
        F.patch_hip_ninja_file_writer(C)
    q = f"{src}/quant"
    units = [f"{q}/quantize.cu", f"{q}/exl3_devctx.cu"] + sorted(glob.glob(f"{q}/comp_units/quantize_tiles_inst_k*.cu")) \
        + sorted(glob.glob(f"{q}/comp_units/quantize_tiles_frac_inst_a*.cu"))
    assert len(units) == 2 + 8 + 7, f"unexpected source set under {q}: {len(units)}"
    return C.load(name=f"exl3cb_{variant}", sources=[os.path.join(here, "cb_bind.cpp")] + units,
                  extra_include_paths=[q, src], extra_cflags=cflags, extra_cuda_cflags=cuda,
                  extra_ldflags=["-lhipblas"] if hip else [], verbose=verbose)


class _Ext:
    NAMES = ("quantize_tiles", "quantize_tiles_scratch", "quantize_tiles_frac")
    def __init__(self, base, cb): self._base, self._cb = base, cb
    def __getattr__(self, name): return getattr(self._cb if name in self.NAMES else self._base, name)


def install(src: str, variant: str, verbose: bool = False):
    import exllamav3.modules.quant.exl3_lib.quantize as Q
    base = Q.ext._base if isinstance(Q.ext, (_Ext, exl3frac._Ext)) else Q.ext
    Q.ext = _Ext(base, load(src, variant, verbose))
    Q.frac_k = exl3frac.frac_k
    Q.get_temp_buffers.cache_clear()            # scratch sizes come from the swapped quantize_tiles_scratch
    return Q
