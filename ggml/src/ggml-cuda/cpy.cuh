#include "common.cuh"

#define CUDA_CPY_BLOCK_SIZE 64

void ggml_cuda_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * src1);

void ggml_cuda_dup(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// several copies that share shape, strides and types (e.g. recurrent-state snapshots written to
// consecutive cache slots) in one launch; same element mapping as ggml_cuda_cpy
#define GGML_CUDA_CPY_MULTI_MAX 16

void ggml_cuda_cpy_f32_multi(ggml_backend_cuda_context & ctx, const ggml_tensor * const * src0, ggml_tensor * const * src1, int n);
