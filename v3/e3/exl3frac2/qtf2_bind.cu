// qtf2: the V3 kernel pass's fractional quantizer (qtf2.cuh) as a side extension, with the same call as the fork's
// quantize_tiles_frac (so exllamav3's quantize.py and ourq.py are unchanged) plus pack / unpack for KA = 0.
// Built twice by exl3frac2.py: plain mul1 and -DEXL3_MUL1_I8 (the int8-rounded codebook, codebook.cuh).
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "qtf2_types.cuh"
#include "qtf2.cuh"
#include "qtf2_table.inc"

#define QTF2_CHECK(x) do { cudaError_t e_ = (x); TORCH_CHECK(e_ == cudaSuccess, "qtf2: ", cudaGetErrorString(e_)); } while (0)

// temp_costs (batch, 2, 65536 >> KA) half: used when KA < 2 (costs in global); temp_edges (batch, 256, 65536 >> KA)
// short: holds the byte history (half of it). The batch is min of the two first dims, as before.
void quantize_tiles_frac(at::Tensor input_tiles, at::Tensor output_tiles, at::Tensor output_indices,
                         at::Tensor temp_costs, at::Tensor temp_edges, int KA, int64_t MASK)
{
    const at::cuda::OptionalCUDAGuard device_guard(input_tiles.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(input_tiles.dim() == 2 && input_tiles.size(1) == 256, "qtf2: tiles must be (n, 256)");
    TORCH_CHECK(input_tiles.scalar_type() == at::kFloat && output_tiles.scalar_type() == at::kFloat, "qtf2: float tiles");
    TORCH_CHECK(output_indices.scalar_type() == at::kShort && temp_costs.scalar_type() == at::kHalf
                && temp_edges.scalar_type() == at::kShort, "qtf2: dtypes");
    TORCH_CHECK(output_tiles.sizes() == input_tiles.sizes() && output_indices.sizes() == input_tiles.sizes(), "qtf2: shapes");
    const int64_t edges_max = 65536 >> KA;
    TORCH_CHECK(temp_costs.dim() == 3 && temp_costs.size(1) == 2 && temp_costs.size(2) == edges_max, "qtf2: temp_costs shape");
    TORCH_CHECK(temp_edges.dim() == 3 && temp_edges.size(1) == 256 && temp_edges.size(2) == edges_max, "qtf2: temp_edges shape");
    for (const auto& t : {input_tiles, output_tiles, output_indices, temp_costs, temp_edges})
        TORCH_CHECK(t.is_contiguous() && t.device() == input_tiles.device(), "qtf2: layout");
    const Qtf2Entry* e = nullptr;
    for (const auto& r : qtf2_table) if (r.ka == KA && r.mask == (uint32_t) MASK) e = &r;
    TORCH_CHECK(e, "qtf2: no instance for (KA, MASK) = (", KA, ", ", MASK, ")");
    const int num_tiles = input_tiles.size(0);
    if (!num_tiles) return;
    const fp_qtf2 kernel = e->fn();
    QTF2_CHECK(cudaFuncSetAttribute((const void*) kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, e->shmem));
    const int max_batch = (int) std::min(temp_costs.size(0), temp_edges.size(0));
    for (int b = 0; b < num_tiles; b += max_batch)
    {
        const int bsz = std::min(max_batch, num_tiles - b);
        kernel<<<bsz, e->nt, e->shmem, stream>>>(
            (const float*) input_tiles.data_ptr() + (int64_t) 256 * b, (float*) output_tiles.data_ptr() + (int64_t) 256 * b,
            (uint16_t*) output_indices.data_ptr() + (int64_t) 256 * b, (half*) temp_costs.data_ptr(),
            (uint16_t*) temp_edges.data_ptr(), nullptr);
        QTF2_CHECK(cudaPeekAtLastError());
    }
}

// The instances this build has: [(KA, MASK)]
std::vector<std::tuple<int, int64_t>> qtf2_instances()
{
    std::vector<std::tuple<int, int64_t>> v;
    for (const auto& r : qtf2_table) v.emplace_back(r.ka, (int64_t) r.mask);
    return v;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("quantize_tiles_frac", &quantize_tiles_frac, "qtf2 fractional trellis quantizer (V3 kernel pass)");
    m.def("qtf2_instances", &qtf2_instances, "(KA, MASK) instances in this build");
}
