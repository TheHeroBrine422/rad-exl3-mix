// exl3cb: exllamav3's trellis search (quantize.cu: integer K, fractional K, scratch sizing) built with an int8- or
// int4-rounded mul1 codebook (EXL3_MUL1_I8 / EXL3_MUL1_I4, codebook.cuh on branch rxm-frac-quarter). See exl3cb.py.
#include <torch/extension.h>
#include "quantize.cuh"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("quantize_tiles", &quantize_tiles, "quantize_tiles (rounded codebook)");
    m.def("quantize_tiles_scratch", &quantize_tiles_scratch, "quantize_tiles_scratch");
    m.def("quantize_tiles_frac", &quantize_tiles_frac, "quantize_tiles_frac (rounded codebook)");
}
