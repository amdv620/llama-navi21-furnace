#pragma once

// Quantized matrix times 2..8 activation columns for RDNA2 (the verify batch of speculative
// decoding), K-quant weights only.
//
// mul_mat_vec_q gives each lane 16 weights of a 256-weight super-block and loads, per column,
// four separate ints of q8_1 activation plus two block scales: on a V620 the vector memory unit
// saturates on load instructions and cache lines long before the weight stream reaches the
// memory bandwidth, and the 8-column kernel runs at 55% of it. Here a lane owns whole
// sub-blocks (two of 32 weights for Q4_K/Q5_K, four of 16 for Q6_K), so its weights are
// contiguous 16-byte loads, each column's activations for a sub-block are whole q8_1 blocks
// (two 16-byte loads), the scales come from one header load, and the min term uses the
// activation block sums the MMQ prefill kernel already uses. Per (row, column) and 32 weights
// this is 8 v_dot4 and 4 float instructions against the 16 dot4 (half of them activation sums)
// and ~26 float ops of the generic kernel, with a fraction of the load instructions.
//
// The summation order differs from mul_mat_vec_q (fp32 rounding only; the min term is the one
// the prefill path already uses), so outputs are not bit-identical to the generic kernel.

#include "common.cuh"
#include "vecdotq.cuh"

// four consecutive dwords, 4-byte aligned; the backend merges these into one global_load_dwordx4
static __device__ __forceinline__ int4 mmvq_wide_load4(const void * p) {
    const int * q = (const int *) p;
    int4 r;
    r.x = q[0];
    r.y = q[1];
    r.z = q[2];
    r.w = q[3];
    return r;
}

static __device__ __forceinline__ int mmvq_wide_dot16(const int4 & v, const int4 & u, int s) {
    s = ggml_cuda_dp4a(v.x, u.x, s);
    s = ggml_cuda_dp4a(v.y, u.y, s);
    s = ggml_cuda_dp4a(v.z, u.z, s);
    s = ggml_cuda_dp4a(v.w, u.w, s);
    return s;
}

static __device__ __forceinline__ int mmvq_wide_byte(const int w, const int i) {
    return (w >> (8*i)) & 0xFF;
}

// 6-bit scale and min of sub-block j of a Q4_K/Q5_K super-block (get_scale_min_k4), from the
// 16-byte block header {dm, scales[12]} held as four dwords
static __device__ __forceinline__ void mmvq_wide_scale_min_k4(const int4 & hdr, const int j, int & sc, int & m) {
    // scales[0..3] = hdr.y, scales[4..7] = hdr.z, scales[8..11] = hdr.w
    if (j < 4) {
        sc = mmvq_wide_byte(hdr.y, j) & 63;
        m  = mmvq_wide_byte(hdr.z, j) & 63;
    } else {
        const int q4 = mmvq_wide_byte(hdr.w, j - 4);   // scales[j+4]
        const int q0 = mmvq_wide_byte(hdr.y, j - 4);   // scales[j-4]
        const int q1 = mmvq_wide_byte(hdr.z, j - 4);   // scales[j]
        sc = (q4 & 0xF) | ((q0 >> 6) << 4);
        m  = (q4 >>  4) | ((q1 >> 6) << 4);
    }
}

// ---------------------------------------------------------------------------------------------
// per-type chunk description: how many lanes share a super-block, the loads of a chunk, and
// how the activations of a group (one q8_1 block, or part of one) are found

template <ggml_type type> struct mmvq_wide_traits;

// Q4_K: chunk (c, t) = low nibbles (sub-block 2c) and high nibbles (sub-block 2c+1) of qs bytes
// 32c+16t .. +15; the 8 lanes of a super-block read its 128 qs bytes contiguously
template <> struct mmvq_wide_traits<GGML_TYPE_Q4_K> {
    static constexpr int lanes_per_block = 8;
    static constexpr int ngroups         = 2;   // sub-blocks per chunk
    static constexpr int nvec            = 1;   // int4 of activations (16 values) per group
    static constexpr bool has_min        = true;

    struct weights {
        int4  v[2][1];
        float rs[2];
        float rm[2];
    };

    static __device__ __forceinline__ void load(const void * vx, const int64_t blk, const int chunk, weights & w) {
        const block_q4_K * b = (const block_q4_K *) vx + blk;
        const int c = chunk >> 1;
        const int4 hdr = mmvq_wide_load4(b);
        const int4 q   = mmvq_wide_load4(b->qs + 16*chunk);
        w.v[0][0] = make_int4( q.x       & 0x0F0F0F0F,  q.y       & 0x0F0F0F0F,  q.z       & 0x0F0F0F0F,  q.w       & 0x0F0F0F0F);
        w.v[1][0] = make_int4((q.x >> 4) & 0x0F0F0F0F, (q.y >> 4) & 0x0F0F0F0F, (q.z >> 4) & 0x0F0F0F0F, (q.w >> 4) & 0x0F0F0F0F);

        const float2 dm = __half22float2(*(const half2 *) &hdr.x);
        int sc, m;
        mmvq_wide_scale_min_k4(hdr, 2*c,     sc, m);
        w.rs[0] = dm.x * (float) sc;
        w.rm[0] = dm.y * (float) m;
        mmvq_wide_scale_min_k4(hdr, 2*c + 1, sc, m);
        w.rs[1] = dm.x * (float) sc;
        w.rm[1] = dm.y * (float) m;
    }

    static __device__ __forceinline__ int y_block(const int chunk, const int g) { return 2*(chunk >> 1) + g; }
    static __device__ __forceinline__ int y_off  (const int chunk)              { return 16*(chunk & 1); }

    // sq = d8 * (sum of the 16 activation quants), shared by the rows
    static __device__ __forceinline__ float dot(const weights & w, const int g, const int4 * u, const float d8, const float sq, float acc) {
        acc = fmaf((float) mmvq_wide_dot16(w.v[g][0], u[0], 0), w.rs[g] * d8, acc);
        acc = fmaf(-w.rm[g], sq, acc);
        return acc;
    }
};

// Q5_K: as Q4_K plus the 5th bit from qh (bit 2c for the low nibbles, 2c+1 for the high ones)
template <> struct mmvq_wide_traits<GGML_TYPE_Q5_K> {
    static constexpr int lanes_per_block = 8;
    static constexpr int ngroups         = 2;
    static constexpr int nvec            = 1;
    static constexpr bool has_min        = true;

    struct weights {
        int4  v[2][1];
        float rs[2];
        float rm[2];
    };

    static __device__ __forceinline__ void load(const void * vx, const int64_t blk, const int chunk, weights & w) {
        const block_q5_K * b = (const block_q5_K *) vx + blk;
        const int c = chunk >> 1;
        const int t = chunk & 1;
        const int4 hdr = mmvq_wide_load4(b);
        const int4 qh  = mmvq_wide_load4(b->qh + 16*t);
        const int4 q   = mmvq_wide_load4(b->qs + 16*chunk);
        const int sl = 2*c;
        const int sh = 2*c + 1;
        w.v[0][0] = make_int4(( q.x       & 0x0F0F0F0F) | (((qh.x >> sl) << 4) & 0x10101010),
                              ( q.y       & 0x0F0F0F0F) | (((qh.y >> sl) << 4) & 0x10101010),
                              ( q.z       & 0x0F0F0F0F) | (((qh.z >> sl) << 4) & 0x10101010),
                              ( q.w       & 0x0F0F0F0F) | (((qh.w >> sl) << 4) & 0x10101010));
        w.v[1][0] = make_int4(((q.x >> 4) & 0x0F0F0F0F) | (((qh.x >> sh) << 4) & 0x10101010),
                              ((q.y >> 4) & 0x0F0F0F0F) | (((qh.y >> sh) << 4) & 0x10101010),
                              ((q.z >> 4) & 0x0F0F0F0F) | (((qh.z >> sh) << 4) & 0x10101010),
                              ((q.w >> 4) & 0x0F0F0F0F) | (((qh.w >> sh) << 4) & 0x10101010));

        const float2 dm = __half22float2(*(const half2 *) &hdr.x);
        int sc, m;
        mmvq_wide_scale_min_k4(hdr, 2*c,     sc, m);
        w.rs[0] = dm.x * (float) sc;
        w.rm[0] = dm.y * (float) m;
        mmvq_wide_scale_min_k4(hdr, 2*c + 1, sc, m);
        w.rs[1] = dm.x * (float) sc;
        w.rm[1] = dm.y * (float) m;
    }

    static __device__ __forceinline__ int y_block(const int chunk, const int g) { return 2*(chunk >> 1) + g; }
    static __device__ __forceinline__ int y_off  (const int chunk)              { return 16*(chunk & 1); }

    static __device__ __forceinline__ float dot(const weights & w, const int g, const int4 * u, const float d8, const float sq, float acc) {
        acc = fmaf((float) mmvq_wide_dot16(w.v[g][0], u[0], 0), w.rs[g] * d8, acc);
        acc = fmaf(-w.rm[g], sq, acc);
        return acc;
    }
};

// Q6_K: chunk (h, p) = half h, byte position p: four groups of 16 weights (scales 8h+2k+p)
template <> struct mmvq_wide_traits<GGML_TYPE_Q6_K> {
    static constexpr int lanes_per_block = 4;
    static constexpr int ngroups         = 4;
    static constexpr int nvec            = 1;
    static constexpr bool has_min        = false;

    struct weights {
        int4  v[4][1];
        float rs[4];
    };

    static __device__ __forceinline__ void load(const void * vx, const int64_t blk, const int chunk, weights & w) {
        const block_q6_K * b = (const block_q6_K *) vx + blk;
        const int h = chunk >> 1;
        const int p = chunk & 1;

        const int4 qa = mmvq_wide_load4(b->ql + 64*h      + 16*p);
        const int4 qb = mmvq_wide_load4(b->ql + 64*h + 32 + 16*p);
        const int4 qh = mmvq_wide_load4(b->qh + 32*h      + 16*p);

        // group k: ql nibble (a low, b low, a high, b high) plus bits 2k..2k+1 of qh, minus 32
        w.v[0][0] = make_int4(__vsubss4(( qa.x       & 0x0F0F0F0F) | (((qh.x >> 0) << 4) & 0x30303030), 0x20202020),
                              __vsubss4(( qa.y       & 0x0F0F0F0F) | (((qh.y >> 0) << 4) & 0x30303030), 0x20202020),
                              __vsubss4(( qa.z       & 0x0F0F0F0F) | (((qh.z >> 0) << 4) & 0x30303030), 0x20202020),
                              __vsubss4(( qa.w       & 0x0F0F0F0F) | (((qh.w >> 0) << 4) & 0x30303030), 0x20202020));
        w.v[1][0] = make_int4(__vsubss4(( qb.x       & 0x0F0F0F0F) | (((qh.x >> 2) << 4) & 0x30303030), 0x20202020),
                              __vsubss4(( qb.y       & 0x0F0F0F0F) | (((qh.y >> 2) << 4) & 0x30303030), 0x20202020),
                              __vsubss4(( qb.z       & 0x0F0F0F0F) | (((qh.z >> 2) << 4) & 0x30303030), 0x20202020),
                              __vsubss4(( qb.w       & 0x0F0F0F0F) | (((qh.w >> 2) << 4) & 0x30303030), 0x20202020));
        w.v[2][0] = make_int4(__vsubss4(((qa.x >> 4) & 0x0F0F0F0F) | (((qh.x >> 4) << 4) & 0x30303030), 0x20202020),
                              __vsubss4(((qa.y >> 4) & 0x0F0F0F0F) | (((qh.y >> 4) << 4) & 0x30303030), 0x20202020),
                              __vsubss4(((qa.z >> 4) & 0x0F0F0F0F) | (((qh.z >> 4) << 4) & 0x30303030), 0x20202020),
                              __vsubss4(((qa.w >> 4) & 0x0F0F0F0F) | (((qh.w >> 4) << 4) & 0x30303030), 0x20202020));
        w.v[3][0] = make_int4(__vsubss4(((qb.x >> 4) & 0x0F0F0F0F) | (((qh.x >> 6) << 4) & 0x30303030), 0x20202020),
                              __vsubss4(((qb.y >> 4) & 0x0F0F0F0F) | (((qh.y >> 6) << 4) & 0x30303030), 0x20202020),
                              __vsubss4(((qb.z >> 4) & 0x0F0F0F0F) | (((qh.z >> 6) << 4) & 0x30303030), 0x20202020),
                              __vsubss4(((qb.w >> 4) & 0x0F0F0F0F) | (((qh.w >> 6) << 4) & 0x30303030), 0x20202020));

        // scales[8h .. 8h+7] as two dwords; group k uses index 8h + 2k + p
        const int * sc32 = (const int *) (b->scales + 8*h);
        const int s0 = sc32[0];
        const int s1 = sc32[1];
        const float d = __half2float(b->d);
        w.rs[0] = d * (float) (int8_t) mmvq_wide_byte(s0, 0 + p);
        w.rs[1] = d * (float) (int8_t) mmvq_wide_byte(s0, 2 + p);
        w.rs[2] = d * (float) (int8_t) mmvq_wide_byte(s1, 0 + p);
        w.rs[3] = d * (float) (int8_t) mmvq_wide_byte(s1, 2 + p);
    }

    static __device__ __forceinline__ int y_block(const int chunk, const int g) { return 4*(chunk >> 1) + g; }
    static __device__ __forceinline__ int y_off  (const int chunk)              { return 16*(chunk & 1); }

    static __device__ __forceinline__ float dot(const weights & w, const int g, const int4 * u, const float d8, const float /*s8*/, float acc) {
        return fmaf((float) mmvq_wide_dot16(w.v[g][0], u[0], 0), w.rs[g] * d8, acc);
    }
};

// ---------------------------------------------------------------------------------------------

template <ggml_type type, int ncols_dst, int RPB>
__launch_bounds__(32, 1)
static __global__ void mul_mat_vec_q_wide(
        const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
        const uint32_t ncols_x, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint3 channel_ratio, const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint3 sample_ratio, const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst) {
    using traits = mmvq_wide_traits<type>;
    constexpr int LPB = traits::lanes_per_block;
    constexpr int NG  = traits::ngroups;
    constexpr int NV  = traits::nvec;
    constexpr int BPI = 32 / LPB;             // super-blocks per wave iteration

    const int lane  = threadIdx.x;
    const int chunk = lane % LPB;
    const int grp   = lane / LPB;

    const uint32_t channel_dst = blockIdx.y;
    const uint32_t channel_x   = fastdiv(channel_dst, channel_ratio);
    const uint32_t channel_y   = channel_dst;
    const uint32_t sample_dst  = blockIdx.z;
    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    const uint32_t row0 = RPB*blockIdx.x;
    const int blocks_per_row = ncols_x / QK_K;

    const block_q8_1 * y = (const block_q8_1 *) vy + sample_y*stride_sample_y + channel_y*stride_channel_y;
    const int64_t x_off = (int64_t) sample_x*stride_sample_x + (int64_t) channel_x*stride_channel_x;

    // rows past the end read the last valid row (never stored)
    int64_t row_off[RPB];
#pragma unroll
    for (int r = 0; r < RPB; ++r) {
        const uint32_t row = min(row0 + r, nrows_x - 1);
        row_off[r] = x_off + (int64_t) row*stride_row_x;
    }

    float acc[ncols_dst][RPB];
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int r = 0; r < RPB; ++r) {
            acc[j][r] = 0.0f;
        }
    }

    for (int blk = grp; blk < blocks_per_row; blk += BPI) {
        typename traits::weights w[RPB];
#pragma unroll
        for (int r = 0; r < RPB; ++r) {
            traits::load(vx, row_off[r] + blk, chunk, w[r]);
        }

        const block_q8_1 * yb = y + (int64_t) blk*(QK_K/QK8_1);
        const int          yo = traits::y_off(chunk);

#pragma unroll
        for (int g = 0; g < NG; ++g) {
            const int ybi = traits::y_block(chunk, g);
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const block_q8_1 * yjb = yb + (int64_t) j*stride_col_y + ybi;
                int4 u[NV];
#pragma unroll
                for (int v = 0; v < NV; ++v) {
                    u[v] = mmvq_wide_load4(yjb->qs + yo + 16*v);
                }
                const float2 ds = __half22float2(yjb->ds);
                float sq = 0.0f;
                if constexpr (traits::has_min) {
                    // d8 times the sum of these activation quants, for the sub-block min term
                    int sumq = 0;
#pragma unroll
                    for (int v = 0; v < NV; ++v) {
                        sumq = mmvq_wide_dot16(make_int4(0x01010101, 0x01010101, 0x01010101, 0x01010101), u[v], sumq);
                    }
                    sq = ds.x * (float) sumq;
                }
#pragma unroll
                for (int r = 0; r < RPB; ++r) {
                    acc[j][r] = traits::dot(w[r], g, u, ds.x, sq, acc[j][r]);
                }
            }
        }
    }

    dst += (int64_t) sample_dst*stride_sample_dst + (int64_t) channel_dst*stride_channel_dst + row0;

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int r = 0; r < RPB; ++r) {
            const float v = warp_reduce_sum<32>(acc[j][r]);
            if (lane == r && row0 + r < nrows_x) {
                dst[(int64_t) j*stride_col_dst + r] = v;
            }
        }
    }
}

template <ggml_type type>
static constexpr bool mmvq_wide_type_supported() {
    return type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K;
}

// output rows per workgroup (GGML_CUDA_MMVQ_WIDE_RPB=2|4|8 for experiments; default 4)
static inline int mmvq_wide_rows() {
    static const int v = [] { const char * e = getenv("GGML_CUDA_MMVQ_WIDE_RPB"); const int r = e ? atoi(e) : 4; return r == 2 || r == 8 ? r : 4; }();
    return v;
}

// GGML_CUDA_NO_MMVQ_WIDE=1 disables the kernel; GGML_CUDA_MMVQ_WIDE_MIN=n sets the smallest column count (default 5)
static inline int mmvq_wide_min_ncols() {
    static const int v = [] {
        if (getenv("GGML_CUDA_NO_MMVQ_WIDE")) {
            return 1 << 30;
        }
        const char * e = getenv("GGML_CUDA_MMVQ_WIDE_MIN");
        return e ? atoi(e) : 5;
    }();
    return v;
}

template <ggml_type type>
static void mul_mat_vec_q_wide_launch(
        const void * vx, const void * vy, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const uint3 channel_ratio, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const uint3 sample_ratio, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int nchannels_dst, const int nsamples_dst, cudaStream_t stream) {
    const dim3 block_dims(32, 1, 1);

    switch (ncols_dst) {
#define MMVQ_WIDE_LAUNCH(n, rpb) \
        { \
            const dim3 block_nums((nrows_x + rpb - 1) / rpb, nchannels_dst, nsamples_dst); \
            const ggml_cuda_kernel_launch_params lp(block_nums, block_dims, 0, stream); \
            ggml_cuda_kernel_launch(mul_mat_vec_q_wide<type, n, rpb>, lp, vx, vy, dst, (uint32_t) ncols_x, (uint32_t) nrows_x, \
                (uint32_t) stride_row_x, (uint32_t) stride_col_y, (uint32_t) stride_col_dst, \
                channel_ratio, (uint32_t) stride_channel_x, (uint32_t) stride_channel_y, (uint32_t) stride_channel_dst, \
                sample_ratio, (uint32_t) stride_sample_x, (uint32_t) stride_sample_y, (uint32_t) stride_sample_dst); \
        }
#define MMVQ_WIDE_CASE(n) \
        case n: \
            switch (mmvq_wide_rows()) { \
                case 2:  MMVQ_WIDE_LAUNCH(n, 2) break; \
                case 8:  MMVQ_WIDE_LAUNCH(n, 8) break; \
                default: MMVQ_WIDE_LAUNCH(n, 4) break; \
            } \
            break;
        MMVQ_WIDE_CASE(2)
        MMVQ_WIDE_CASE(3)
        MMVQ_WIDE_CASE(4)
        MMVQ_WIDE_CASE(5)
        MMVQ_WIDE_CASE(6)
        MMVQ_WIDE_CASE(7)
        MMVQ_WIDE_CASE(8)
#undef MMVQ_WIDE_CASE
#undef MMVQ_WIDE_LAUNCH
        default:
            GGML_ABORT("fatal error");
    }
}
