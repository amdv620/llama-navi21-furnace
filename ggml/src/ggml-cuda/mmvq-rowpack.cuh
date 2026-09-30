#pragma once

// EXPERIMENT (GGML_CUDA_ROWPACK=1, RDNA2, Q4_K, 2D weights, up to 8 columns): the weight matrix
// is repacked once into a side buffer where 32 consecutive rows are interleaved in 16-byte
// pieces, so a wave with one row per lane streams contiguous 512-byte runs, and the q8_1
// activations, identical for every lane, are read through the scalar path. The generic MMVQ
// kernel saturates the vector memory unit on the activations it re-reads for every 4 rows; here
// they are read once per 32 rows and never touch it. Small row counts split K across waves and
// reduce the partial sums in a fixed order (deterministic).
//
// Numerics: fp32 summation in a different order from mul_mat_vec_q, and the sub-block min term
// uses the q8_1 block sums as the MMQ prefill kernel does.
//
// The side buffers duplicate the weights (6.1 GB for this model's Q4_K tensors) and live until
// the process exits; this is an experiment, not a shipping path.

#include "common.cuh"

#include <unordered_map>
#include <mutex>
#include <string>

#define ROWPACK_ROWS   32
#define ROWPACK_Q4K_PIECES 9   // 144 bytes = 9 x 16

static inline bool ggml_cuda_rowpack_enabled() {
    static const bool v = getenv("GGML_CUDA_ROWPACK") != nullptr && atoi(getenv("GGML_CUDA_ROWPACK")) != 0;
    return v;
}

// smallest column count that takes the rowpack path (below it the generic kernel is as fast or faster)
static inline int ggml_cuda_rowpack_min_ncols() {
    static const int v = getenv("GGML_CUDA_ROWPACK_MIN") ? atoi(getenv("GGML_CUDA_ROWPACK_MIN")) : 7;
    return v;
}

#define ROWPACK_Q5K_PIECES 11  // 176 bytes
#define ROWPACK_Q6K_PIECES 14  // 210 bytes padded to 224: ql 0..127, qh 128..191, scales 192..207, d 208..209

// [tile][super-block][piece][lane][16 B]; rows past the end and padding are zero. Source blocks
// are `src_bytes` apart and only 2-byte aligned (Q6_K), so pieces are copied as 16-bit words.
static __global__ void rowpack_repack(const uint16_t * __restrict__ src, uint4 * __restrict__ dst,
                                      const int nrows, const int nsb, const int pieces, const int src_bytes) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;   // destination piece index
    const int64_t ntiles = (nrows + ROWPACK_ROWS - 1) / ROWPACK_ROWS;
    if (i >= ntiles * nsb * pieces * ROWPACK_ROWS) {
        return;
    }
    const int lane = i % ROWPACK_ROWS;
    const int p    = (i / ROWPACK_ROWS) % pieces;
    const int b    = (i / (ROWPACK_ROWS*pieces)) % nsb;
    const int tile =  i / ((int64_t) ROWPACK_ROWS*pieces*nsb);
    const int row  = tile*ROWPACK_ROWS + lane;
    uint16_t w[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    if (row < nrows) {
        const uint16_t * sp = src + (((int64_t) row*nsb + b)*src_bytes + 16*p) / 2;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            if (16*p + 2*k < src_bytes) {
                w[k] = sp[k];
            }
        }
    }
    dst[i] = make_uint4((uint32_t) w[0] | ((uint32_t) w[1] << 16), (uint32_t) w[2] | ((uint32_t) w[3] << 16),
                        (uint32_t) w[4] | ((uint32_t) w[5] << 16), (uint32_t) w[6] | ((uint32_t) w[7] << 16));
}

static __device__ __forceinline__ int rowpack_byte(const uint32_t w, const int i) { return (w >> (8*i)) & 0xFF; }

// get_scale_min_k4 from the 16-byte block header {dm, scales[12]} held as four dwords
static __device__ __forceinline__ void rowpack_scale_min(const uint4 & h, const int j, int & sc, int & m) {
    if (j < 4) {
        sc = rowpack_byte(h.y, j) & 63;
        m  = rowpack_byte(h.z, j) & 63;
    } else {
        const int q4 = rowpack_byte(h.w, j - 4);
        const int q0 = rowpack_byte(h.y, j - 4);
        const int q1 = rowpack_byte(h.z, j - 4);
        sc = (q4 & 0xF) | ((q0 >> 6) << 4);
        m  = (q4 >>  4) | ((q1 >> 6) << 4);
    }
}

static __device__ __forceinline__ float2 rowpack_half2(const uint32_t w) {
    return __half22float2(*reinterpret_cast<const half2 *>(&w));
}

// wp: repacked weights; y: q8_1 activations, column j at y + j*stride_col_y blocks (36 B each);
// grid (tiles, nsplit); with nsplit > 1 the partial sums go to part[split][ncols][nrows]
// The activations are wave-uniform and read through scalar loads (SGPR operands of v_dot4);
// staging them through shared memory was measured 5x slower (both dot operands then need
// vector registers and the 8-column instance spills).
template <ggml_type type, int ncols>
__launch_bounds__(ROWPACK_ROWS, 1)
static __global__ void mul_mat_vec_q45k_rowpack(
        const uint4 * __restrict__ wp, const block_q8_1 * __restrict__ y, float * __restrict__ dst, float * __restrict__ part,
        const int nsb, const int nrows, const int stride_col_y, const int stride_col_dst) {
    const int r      = threadIdx.x;
    const int tile   = blockIdx.x;
    const int nsplit = gridDim.y;
    // wave-uniform loop bounds, stated as such so the address arithmetic stays on the scalar unit
    const int b0 = __builtin_amdgcn_readfirstlane((int) ((int64_t) nsb *  blockIdx.y      / nsplit));
    const int b1 = __builtin_amdgcn_readfirstlane((int) ((int64_t) nsb * (blockIdx.y + 1) / nsplit));
    constexpr int PIECES = type == GGML_TYPE_Q5_K ? ROWPACK_Q5K_PIECES : ROWPACK_Q4K_PIECES;
    constexpr int QS0    = type == GGML_TYPE_Q5_K ? 3 : 1;   // first qs piece (Q5_K: pieces 1..2 are qh)
    const uint4 * base = wp + (int64_t) tile * nsb * PIECES * ROWPACK_ROWS;
    // block b of column j starts at dword 9*(j*stride_col_y + b): ds, then 8 dwords of quants.
    // 32-bit, per-column bases hoisted: keeps the address arithmetic on the scalar unit.
    const uint32_t * ycol[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        ycol[j] = (const uint32_t *) y + 9u * (uint32_t) j * (uint32_t) stride_col_y;
    }

    float acc[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        acc[j] = 0.0f;
    }

    for (int b = b0; b < b1; ++b) {
        const uint4 * blk = base + (int64_t) b * PIECES * ROWPACK_ROWS;
        const uint4 hdr = blk[r];
        const float2 dm = rowpack_half2(hdr.x);
        [[maybe_unused]] uint32_t qh[8];
        if constexpr (type == GGML_TYPE_Q5_K) {
            const uint4 h0 = blk[1 * ROWPACK_ROWS + r];
            const uint4 h1 = blk[2 * ROWPACK_ROWS + r];
            qh[0] = h0.x; qh[1] = h0.y; qh[2] = h0.z; qh[3] = h0.w; qh[4] = h1.x; qh[5] = h1.y; qh[6] = h1.z; qh[7] = h1.w;
        }

#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const uint4 q0 = blk[(QS0 + 2*c) * ROWPACK_ROWS + r];
            const uint4 q1 = blk[(QS0 + 1 + 2*c) * ROWPACK_ROWS + r];
            const uint32_t q[8] = { q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w };
            int vlo[8], vhi[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                vlo[i] =  q[i]       & 0x0F0F0F0F;
                vhi[i] = (q[i] >> 4) & 0x0F0F0F0F;
                if constexpr (type == GGML_TYPE_Q5_K) {
                    // 5th bit: bit 2c of qh for the low nibbles, bit 2c+1 for the high ones
                    vlo[i] |= ((qh[i] >> (2*c))     << 4) & 0x10101010;
                    vhi[i] |= ((qh[i] >> (2*c + 1)) << 4) & 0x10101010;
                }
            }

            int sc, m;
            rowpack_scale_min(hdr, 2*c,     sc, m);
            const float rs0 = dm.x * (float) sc, rm0 = dm.y * (float) m;
            rowpack_scale_min(hdr, 2*c + 1, sc, m);
            const float rs1 = dm.x * (float) sc, rm1 = dm.y * (float) m;

#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                // uniform addresses: scalar loads
                const uint32_t * u0 = ycol[j] + 9u * (uint32_t) (8*b + 2*c);
                const uint32_t * u1 = u0 + 9;
                int s0 = 0, s1 = 0;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    s0 = ggml_cuda_dp4a(vlo[i], (int) u0[1 + i], s0);
                    s1 = ggml_cuda_dp4a(vhi[i], (int) u1[1 + i], s1);
                }
                const float2 ds0 = rowpack_half2(u0[0]);
                const float2 ds1 = rowpack_half2(u1[0]);
                acc[j] = fmaf((float) s0, rs0 * ds0.x, acc[j]);
                acc[j] = fmaf(-rm0, ds0.y, acc[j]);
                acc[j] = fmaf((float) s1, rs1 * ds1.x, acc[j]);
                acc[j] = fmaf(-rm1, ds1.y, acc[j]);
            }
        }
    }

    const int row = tile*ROWPACK_ROWS + r;
    if (row >= nrows) {
        return;
    }
    if (nsplit == 1) {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            dst[(int64_t) j*stride_col_dst + row] = acc[j];
        }
    } else {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            part[((int64_t) blockIdx.y*ncols + j)*nrows + row] = acc[j];
        }
    }
}

// Q6_K: 14 pieces per block (ql 0..7, qh 8..11, scales 12, d 13). A lane decodes one half
// (128 weights) at a time: four groups of 32 weights, each one q8_1 block of activations, with
// two 16-weight scales per group. No min term.
template <int ncols>
__launch_bounds__(ROWPACK_ROWS, 1)
static __global__ void mul_mat_vec_q6k_rowpack(
        const uint4 * __restrict__ wp, const block_q8_1 * __restrict__ y, float * __restrict__ dst, float * __restrict__ part,
        const int nsb, const int nrows, const int stride_col_y, const int stride_col_dst) {
    const int r      = threadIdx.x;
    const int tile   = blockIdx.x;
    const int nsplit = gridDim.y;
    const int b0 = __builtin_amdgcn_readfirstlane((int) ((int64_t) nsb *  blockIdx.y      / nsplit));
    const int b1 = __builtin_amdgcn_readfirstlane((int) ((int64_t) nsb * (blockIdx.y + 1) / nsplit));
    const uint4 * base = wp + (int64_t) tile * nsb * ROWPACK_Q6K_PIECES * ROWPACK_ROWS;
    const uint32_t * ycol[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        ycol[j] = (const uint32_t *) y + 9u * (uint32_t) j * (uint32_t) stride_col_y;
    }

    float acc[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        acc[j] = 0.0f;
    }

    for (int b = b0; b < b1; ++b) {
        const uint4 * blk = base + (int64_t) b * ROWPACK_Q6K_PIECES * ROWPACK_ROWS;
        const uint4 scp = blk[12 * ROWPACK_ROWS + r];   // 16 int8 scales
        const uint4 dp  = blk[13 * ROWPACK_ROWS + r];
        const float d   = __half2float(*reinterpret_cast<const half *>(&dp.x));
        const uint32_t scw[4] = { scp.x, scp.y, scp.z, scp.w };

#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint4 a0 = blk[(4*h + 0) * ROWPACK_ROWS + r];
            const uint4 a1 = blk[(4*h + 1) * ROWPACK_ROWS + r];
            const uint4 b0v = blk[(4*h + 2) * ROWPACK_ROWS + r];
            const uint4 b1v = blk[(4*h + 3) * ROWPACK_ROWS + r];
            const uint4 h0 = blk[(8 + 2*h) * ROWPACK_ROWS + r];
            const uint4 h1 = blk[(9 + 2*h) * ROWPACK_ROWS + r];
            const uint32_t qa[8] = { a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w };
            const uint32_t qb[8] = { b0v.x, b0v.y, b0v.z, b0v.w, b1v.x, b1v.y, b1v.z, b1v.w };
            const uint32_t qh[8] = { h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w };

#pragma unroll
            for (int k = 0; k < 4; ++k) {
                // group k of half h: weights 128h + 32k + i, q8_1 block 4h + k, scales 8h + 2k and 8h + 2k + 1
                int v[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const uint32_t ql = (k & 1) ? qb[i] : qa[i];
                    const uint32_t lo = (k & 2) ? ((ql >> 4) & 0x0F0F0F0F) : (ql & 0x0F0F0F0F);
                    v[i] = __vsubss4((int) (lo | (((qh[i] >> (2*k)) << 4) & 0x30303030)), 0x20202020);
                }
                const float sa = d * (float) (int8_t) rowpack_byte(scw[(8*h + 2*k) / 4], (8*h + 2*k) % 4);
                const float sb = d * (float) (int8_t) rowpack_byte(scw[(8*h + 2*k + 1) / 4], (8*h + 2*k + 1) % 4);

#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    const uint32_t * w = ycol[j] + 9u * (uint32_t) (8*b + 4*h + k);
                    int s0 = 0, s1 = 0;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        s0 = ggml_cuda_dp4a(v[i],     (int) w[1 + i],     s0);
                        s1 = ggml_cuda_dp4a(v[4 + i], (int) w[5 + i],     s1);
                    }
                    const float d8 = rowpack_half2(w[0]).x;
                    acc[j] = fmaf((float) s0, sa * d8, acc[j]);
                    acc[j] = fmaf((float) s1, sb * d8, acc[j]);
                }
            }
        }
    }

    const int row = tile*ROWPACK_ROWS + r;
    if (row >= nrows) {
        return;
    }
    if (nsplit == 1) {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            dst[(int64_t) j*stride_col_dst + row] = acc[j];
        }
    } else {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            part[((int64_t) blockIdx.y*ncols + j)*nrows + row] = acc[j];
        }
    }
}

// sums the K-split partials in split order (deterministic) and writes the result
static __global__ void rowpack_reduce(const float * __restrict__ part, float * __restrict__ dst,
                                      const int ncols, const int nrows, const int stride_col_dst, const int nsplit) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (int64_t) ncols*nrows) {
        return;
    }
    const int j   = i / nrows;
    const int row = i % nrows;
    float s = 0.0f;
    for (int k = 0; k < nsplit; ++k) {
        s += part[((int64_t) k*ncols + j)*nrows + row];
    }
    dst[(int64_t) j*stride_col_dst + row] = s;
}

// side-buffer cache of repacked weights, keyed by the weight tensor's device pointer
struct ggml_cuda_rowpack_entry {
    uint4 * data;
    int     nrows;
    int     nsb;
    int     pieces;
};

static inline int ggml_cuda_rowpack_pieces(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_K: return ROWPACK_Q4K_PIECES;
        case GGML_TYPE_Q5_K: return ROWPACK_Q5K_PIECES;
        case GGML_TYPE_Q6_K: return ROWPACK_Q6K_PIECES;
        default:             return 0;
    }
}

// GGML_CUDA_ROWPACK_TYPES=q4k,q5k,q6k selects the types that take the path (default: all three)
static inline bool ggml_cuda_rowpack_type_enabled(const ggml_type type) {
    static const std::string sel = getenv("GGML_CUDA_ROWPACK_TYPES") ? getenv("GGML_CUDA_ROWPACK_TYPES") : "q4k,q5k";   // Q6_K measured slower than the generic kernel
    switch (type) {
        case GGML_TYPE_Q4_K: return sel.find("q4k") != std::string::npos;
        case GGML_TYPE_Q5_K: return sel.find("q5k") != std::string::npos;
        case GGML_TYPE_Q6_K: return sel.find("q6k") != std::string::npos;
        default:             return false;
    }
}

static ggml_cuda_rowpack_entry ggml_cuda_rowpack_get(const ggml_tensor * src0, cudaStream_t stream) {
    static std::unordered_map<const void *, ggml_cuda_rowpack_entry> cache;
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    auto it = cache.find(src0->data);
    if (it != cache.end()) {
        return it->second;
    }
    ggml_cuda_rowpack_entry e;
    e.nrows  = (int) src0->ne[1];
    e.nsb    = (int) (src0->ne[0] / QK_K);
    e.pieces = ggml_cuda_rowpack_pieces(src0->type);
    const int64_t ntiles = (e.nrows + ROWPACK_ROWS - 1) / ROWPACK_ROWS;
    const int64_t npieces = ntiles * e.nsb * e.pieces * ROWPACK_ROWS;
    CUDA_CHECK(cudaMalloc(&e.data, npieces * sizeof(uint4)));
    const int64_t nblocks = (npieces + 255) / 256;
    rowpack_repack<<<(unsigned) nblocks, 256, 0, stream>>>((const uint16_t *) src0->data, e.data, e.nrows, e.nsb, e.pieces, (int) ggml_type_size(src0->type));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cache[src0->data] = e;
    return e;
}

// true when this call can and should take the rowpack path
static inline bool ggml_cuda_rowpack_applicable(const ggml_tensor * src0, const ggml_tensor * ids, const bool has_fusion, const int64_t ncols_dst, const int cc) {
    return ggml_cuda_rowpack_enabled() && GGML_CUDA_CC_IS_RDNA2(cc) && ggml_cuda_rowpack_pieces(src0->type) != 0 && ggml_cuda_rowpack_type_enabled(src0->type) && !ids && !has_fusion &&
           src0->ne[2] == 1 && src0->ne[3] == 1 && ggml_is_contiguous(src0) && ncols_dst >= ggml_cuda_rowpack_min_ncols() && ncols_dst <= 8 &&
           src0->ne[0] % QK_K == 0;
}

static void ggml_cuda_mul_mat_vec_rowpack(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const char * src1_q8_1, float * dst_d,
        const int64_t ncols_dst, const int64_t stride_col_y, const int64_t stride_col_dst, cudaStream_t stream) {
    const ggml_cuda_rowpack_entry e = ggml_cuda_rowpack_get(src0, stream);
    const int ntiles = (e.nrows + ROWPACK_ROWS - 1) / ROWPACK_ROWS;

    // enough waves to keep the memory pipeline full: about 7 per SIMD on a 36-WGP part
    static const int target = getenv("GGML_CUDA_ROWPACK_WAVES") ? atoi(getenv("GGML_CUDA_ROWPACK_WAVES")) : 1024;
    int nsplit = 1;
    while (ntiles*nsplit < target && nsplit < 8 && e.nsb / (nsplit*2) >= 4) {
        nsplit *= 2;
    }

    ggml_cuda_pool_alloc<float> part(ctx.pool());
    float * part_d = nullptr;
    if (nsplit > 1) {
        part.alloc((size_t) nsplit * ncols_dst * e.nrows);
        part_d = part.get();
    }

    const dim3 grid(ntiles, nsplit);
    const dim3 block(ROWPACK_ROWS);
    const block_q8_1 * y = (const block_q8_1 *) src1_q8_1;
#define ROWPACK_ARGS e.data, y, dst_d, part_d, e.nsb, e.nrows, (int) stride_col_y, (int) stride_col_dst
#define ROWPACK_SWITCH45(T) \
    switch (ncols_dst) { \
        case 1: mul_mat_vec_q45k_rowpack<T, 1><<<grid, block, 0, stream>>>(ROWPACK_ARGS); break; \
        case 2: mul_mat_vec_q45k_rowpack<T, 2><<<grid, block, 0, stream>>>(ROWPACK_ARGS); break; \
        case 3: mul_mat_vec_q45k_rowpack<T, 3><<<grid, block, 0, stream>>>(ROWPACK_ARGS); break; \
        case 4: mul_mat_vec_q45k_rowpack<T, 4><<<grid, block, 0, stream>>>(ROWPACK_ARGS); break; \
        case 5: mul_mat_vec_q45k_rowpack<T, 5><<<grid, block, 0, stream>>>(ROWPACK_ARGS); break; \
        case 6: mul_mat_vec_q45k_rowpack<T, 6><<<grid, block, 0, stream>>>(ROWPACK_ARGS); break; \
        case 7: mul_mat_vec_q45k_rowpack<T, 7><<<grid, block, 0, stream>>>(ROWPACK_ARGS); break; \
        case 8: mul_mat_vec_q45k_rowpack<T, 8><<<grid, block, 0, stream>>>(ROWPACK_ARGS); break; \
        default: GGML_ABORT("fatal error"); \
    }
    switch (src0->type) {
        case GGML_TYPE_Q4_K: ROWPACK_SWITCH45(GGML_TYPE_Q4_K); break;
        case GGML_TYPE_Q5_K: ROWPACK_SWITCH45(GGML_TYPE_Q5_K); break;
        case GGML_TYPE_Q6_K:
            switch (ncols_dst) {
                case 1: mul_mat_vec_q6k_rowpack<1><<<grid, block, 0, stream>>>(ROWPACK_ARGS); break;
                case 2: mul_mat_vec_q6k_rowpack<2><<<grid, block, 0, stream>>>(ROWPACK_ARGS); break;
                case 3: mul_mat_vec_q6k_rowpack<3><<<grid, block, 0, stream>>>(ROWPACK_ARGS); break;
                case 4: mul_mat_vec_q6k_rowpack<4><<<grid, block, 0, stream>>>(ROWPACK_ARGS); break;
                case 5: mul_mat_vec_q6k_rowpack<5><<<grid, block, 0, stream>>>(ROWPACK_ARGS); break;
                case 6: mul_mat_vec_q6k_rowpack<6><<<grid, block, 0, stream>>>(ROWPACK_ARGS); break;
                case 7: mul_mat_vec_q6k_rowpack<7><<<grid, block, 0, stream>>>(ROWPACK_ARGS); break;
                case 8: mul_mat_vec_q6k_rowpack<8><<<grid, block, 0, stream>>>(ROWPACK_ARGS); break;
                default: GGML_ABORT("fatal error");
            }
            break;
        default: GGML_ABORT("fatal error");
    }
#undef ROWPACK_SWITCH45
#undef ROWPACK_ARGS
    CUDA_CHECK(cudaGetLastError());
    if (nsplit > 1) {
        const int64_t n = ncols_dst * e.nrows;
        rowpack_reduce<<<(unsigned) ((n + 255) / 256), 256, 0, stream>>>(part_d, dst_d, (int) ncols_dst, e.nrows, (int) stride_col_dst, nsplit);
        CUDA_CHECK(cudaGetLastError());
    }
}
