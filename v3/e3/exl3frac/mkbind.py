#!/usr/bin/env python3
"""mkbind.py FORK_QUANTIZE_CU -- regenerate frac_bind.cu from the fork's exllamav3_ext/quant/quantize.cu (the
quantize_tiles_frac function with its instance table) plus a pybind module."""
import os, sys
src = open(sys.argv[1]).read()
i = src.index('// Fractional-rate quantizer, mul1 codebook only')
j = src.index('\n}\n', src.index('void quantize_tiles_frac', i)) + 3
head = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "frac_bind.cu")).read().split("// Fractional-rate", 1)[0]
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "frac_bind.cu"), "w").write(
    head + src[i:j] + '\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m)\n{\n    m.def("quantize_tiles_frac", &quantize_tiles_frac, '
    '"quantize_tiles_frac (V3 menu instances)");\n}\n')
