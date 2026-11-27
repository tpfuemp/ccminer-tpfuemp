// nebhash v2 attempts on CUDA from a table of per-sample gradients. The contract is in nebhash2_table_cuda.h.
//
// The gradient of an attempt is the sum of the gradients of its 16 samples, clipped afterwards, and the gradient of a
// sample depends on the model and on that sample only. For one model state the host evaluates every sample once with
// neb2_sample_grad of the C reference and uploads one row per sample. A row holds, per layer: every gradient entry of
// a convolution (8 or 16 bits, or 32 when a value does not fit), and for a linear layer its cout errors then its cin
// inputs, whose outer product is the gradient of the layer. One thread block evaluates one attempt: it gathers the
// factors of its 16 rows, transposed so that one entry of a linear layer is four dp4a, sums, clips, and hashes.
#include "nebhash2_table_cuda.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <thread>
#include <vector>
#include "nebhash2.h"
#include "cuda/blake3_device.cuh"

#ifndef THREADS
#define THREADS 128
#endif
#define CHUNK 65536u  // attempts per launch

__device__ __forceinline__ uint32_t rotl32(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }

#define QR(a, b, c, d) \
  x[a] += x[b]; x[d] = rotl32(x[d] ^ x[a], 16); x[c] += x[d]; x[b] = rotl32(x[b] ^ x[c], 12); \
  x[a] += x[b]; x[d] = rotl32(x[d] ^ x[a], 8);  x[c] += x[d]; x[b] = rotl32(x[b] ^ x[c], 7);

static __device__ void chacha8_block0(const uint32_t key[8], uint32_t out[16]) {
  uint32_t s[16], x[16];
  s[0] = 0x61707865u; s[1] = 0x3320646eu; s[2] = 0x79622d32u; s[3] = 0x6b206574u;
  for (int i = 0; i < 8; i++) s[4 + i] = key[i];
  s[12] = 0; s[13] = 0; s[14] = 0; s[15] = 0;
  for (int i = 0; i < 16; i++) x[i] = s[i];
  for (int r = 0; r < 4; r++) {
    QR(0, 4, 8, 12) QR(1, 5, 9, 13) QR(2, 6, 10, 14) QR(3, 7, 11, 15)
    QR(0, 5, 10, 15) QR(1, 6, 11, 12) QR(2, 7, 8, 13) QR(3, 4, 9, 14)
  }
  for (int i = 0; i < 16; i++) out[i] = x[i] + s[i];
}

enum { NARROW = 0, WIDE = 1, LINEAR = 2, NARROW8 = 3 };

// The layout of a row and of the shared memory of a block.
struct TPlan {
  int nlayers;
  int kind[NEB2_MAX_LAYERS], nw[NEB2_MAX_LAYERS], cin[NEB2_MAX_LAYERS], cout[NEB2_MAX_LAYERS];
  int at[NEB2_MAX_LAYERS];                             // offset of the layer in a row
  int sh_d[NEB2_MAX_LAYERS], sh_in[NEB2_MAX_LAYERS];   // linear: error words in shared; sh_in is unused (rows are reloaded)
  int row, shared_words;
  uint32_t linear_zero[8];
};
static __device__ __constant__ TPlan P;

#ifdef NEB_NEGATIVE_CONTROL_TABLE_CLIP
#define CLIPV (NEB_CLIP - 1)  // wrong on purpose: the check must fail
#else
#define CLIPV NEB_CLIP
#endif

__device__ __forceinline__ uint32_t warp_sum(uint32_t v)
{
  for (int off = 16; off; off >>= 1)
    v += __shfl_down_sync(0xffffffffu, v, off);
  return v;
}

__global__ void header_prefix(const uint32_t *header, uint32_t *cv) {
  uint32_t first[16];
  for (int i = 0; i < 16; i++) first[i] = header[i];
  blake3_compress(blake3_iv(), first, 0, 64, BLAKE3_CHUNK_START, cv);
}

// One linear layer of an attempt. With FIXED (cin % 8 == 0; THREADS % 8 == 0 below), every entry f = o * cin + q of
// this thread has f & 7 == q & 7 == tid & 7, so it adds into ONE lane: a plain accumulator replaces the runtime-indexed
// lane[f & 7], which ptxas turned into an 8-way compare + predicated read and write per entry. Sums are mod 2^32, so
// the order of the additions does not change the ticket. !FIXED keeps the per-entry lane index.
static_assert(THREADS % 8 == 0, "the fixed lane index needs THREADS % 8 == 0");
template <bool FIXED>
__device__ __forceinline__ void linear_layer(const uint8_t *__restrict__ rows, const uint32_t *rb, const int32_t *S,
                                             int i, int tid, const int *act, int nact, uint32_t base, uint32_t *lane) {
  const int cin = P.cin[i];
  const int xat = P.at[i] + P.cout[i];
  uint32_t acc = 0;
  for (int q = tid; q < cin; q += THREADS) {
    int32_t x[4];
    for (int w = 0; w < 4; w++) {
      uint32_t v = 0;
      for (int b = 0; b < 4; b++)
        v |= (uint32_t)__ldg(&rows[rb[4 * w + b] + xat + q]) << (8 * b);
      x[w] = (int32_t)v;
    }
    for (int j = 0; j < nact; j++) {
      const int o = act[j];
      const int32_t *d = S + P.sh_d[i] + 4 * o;
      int G = 0;
      G = __dp4a(d[0], x[0], G);
      G = __dp4a(d[1], x[1], G);
      G = __dp4a(d[2], x[2], G);
      G = __dp4a(d[3], x[3], G);
      const uint32_t f = (uint32_t)(o * cin + q);
      const int cg = G < -CLIPV ? -CLIPV : (G > CLIPV ? CLIPV : G);
      const uint32_t v = ((uint32_t)cg ^ (f + base)) - (f + base);
      if (FIXED) acc += v; else lane[f & 7] += v;
    }
  }
  if (FIXED) lane[tid & 7] += acc;
}

// Linear layer, two inputs per thread (needs cin % 8 == 0 and an even xat): one 16-bit load per sample brings
// x[q] and x[q+1] of that sample, so 16 loads serve two input columns instead of 32 byte loads. Bytes of four
// samples are transposed with PRMT into the sample-packed words the dp4a takes (byte k = sample 4w + k), exactly the
// words the byte-load path builds. Lane of every entry: q & 7 = (2 * tid + j) & 7, two fixed accumulators.
__device__ __forceinline__ void linear_layer_pairs(const uint8_t *__restrict__ rows, const uint32_t *rb, const int32_t *S,
                                                   int i, int tid, const int *act, int nact, uint32_t base, uint32_t *lane) {
  const int cin = P.cin[i];
  const int xat = P.at[i] + P.cout[i];
  uint32_t acc0 = 0, acc1 = 0;
  for (int q = 2 * tid; q < cin; q += 2 * THREADS) {
    int32_t x0[4], x1[4];
#pragma unroll
    for (int w = 0; w < 4; w++) {
      const uint32_t a = __ldg((const unsigned short *)(rows + rb[4 * w + 0] + xat + q));
      const uint32_t b = __ldg((const unsigned short *)(rows + rb[4 * w + 1] + xat + q));
      const uint32_t c = __ldg((const unsigned short *)(rows + rb[4 * w + 2] + xat + q));
      const uint32_t d = __ldg((const unsigned short *)(rows + rb[4 * w + 3] + xat + q));
      const uint32_t t01 = a | (b << 16), t23 = c | (d << 16);  // bytes: s0q s0q1 s1q s1q1 | s2q s2q1 s3q s3q1
      x0[w] = (int32_t)__byte_perm(t01, t23, 0x6420);
      x1[w] = (int32_t)__byte_perm(t01, t23, 0x7531);
    }
    for (int j = 0; j < nact; j++) {
      const int o = act[j];
      const int32_t *d = S + P.sh_d[i] + 4 * o;
      const int32_t d0 = d[0], d1 = d[1], d2 = d[2], d3 = d[3];
      int G0 = __dp4a(d0, x0[0], 0), G1 = __dp4a(d0, x1[0], 0);
      G0 = __dp4a(d1, x0[1], G0); G1 = __dp4a(d1, x1[1], G1);
      G0 = __dp4a(d2, x0[2], G0); G1 = __dp4a(d2, x1[2], G1);
      G0 = __dp4a(d3, x0[3], G0); G1 = __dp4a(d3, x1[3], G1);
      const uint32_t f0 = (uint32_t)(o * cin + q), f1 = f0 + 1;
      const int c0 = G0 < -CLIPV ? -CLIPV : (G0 > CLIPV ? CLIPV : G0);
      const int c1 = G1 < -CLIPV ? -CLIPV : (G1 > CLIPV ? CLIPV : G1);
      acc0 += ((uint32_t)c0 ^ (f0 + base)) - (f0 + base);
      acc1 += ((uint32_t)c1 ^ (f1 + base)) - (f1 + base);
    }
  }
  lane[(2 * tid) & 7] += acc0;
  lane[(2 * tid + 1) & 7] += acc1;
}

// Pascal (sm_6x) fits 7 blocks/SM at 72 registers; minBlocks 8 caps it at 64 (4 B spill) for 8 blocks: +10-12 % on the
// 1080 Ti over 5 live models, but only when this file is built WITHOUT --maxrregcount (with it: +4.5-8 %), hence the
// per-file override in both build systems. Every sm_75/86 arm lost (minBlocks 10/12 spill: -24..-28 %; THREADS 64/256:
// -7..-12 %), so those keep the plain bound (sm_75 SASS unchanged, sm_86 neutral without the cap).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
#define NEB_TABLE_BOUNDS __launch_bounds__(THREADS, 8)
#else
#define NEB_TABLE_BOUNDS __launch_bounds__(THREADS)
#endif
__global__ void NEB_TABLE_BOUNDS table_kernel(const uint8_t *__restrict__ rows, uint32_t n,
                                                        const uint32_t *__restrict__ h0, uint32_t start,
                                                        const uint32_t *__restrict__ bound, uint32_t *hits,
                                                        uint32_t max_hits, const uint32_t *__restrict__ cv) {
  extern __shared__ int32_t S[];  // per linear layer: cout x 4 words of errors (inputs reloaded from rows)
  __shared__ uint32_t key[8], idx[NEB_NB], lanes[8];
  __shared__ uint32_t wsum[8][THREADS / 32];
  __shared__ int active[NEB2_MAX_LAYERS][32], active_count[NEB2_MAX_LAYERS];
  const int tid = threadIdx.x;
  const uint32_t nonce = start + blockIdx.x;
  if (tid < 8) lanes[tid] = 0;
  if (tid == 0) {
    uint32_t b1[16] = { h0[16], h0[17], h0[18], nonce };
    uint32_t k[8], u[16];
    blake3_compress(cv, b1, 0, 16, BLAKE3_CHUNK_END | BLAKE3_ROOT, k);
    for (int i = 0; i < 8; i++) key[i] = k[i];
    chacha8_block0(k, u);
    for (int j = 0; j < NEB_NB; j++) idx[j] = (uint32_t)(((unsigned long long)u[j] * n) >> 32);
  }
  __syncthreads();
  // Row offsets fit in 32 bits for the official dataset (build() rejects a
  // larger table), which avoids a 64-bit multiply on every row byte load.
  uint32_t rb[NEB_NB];
#pragma unroll
  for (int b = 0; b < NEB_NB; b++) rb[b] = idx[b] * (uint32_t)P.row;
  // Only the cout error words need staging. The cin input words are loaded once
  // per input element and reused across all outputs, so shared memory stays tiny
  // and many more blocks fit on an SM. The row bytes and their packing are the
  // same as before; only the loop order changes, and the lane sums are mod 2^32.
  for (int i = 0; i < P.nlayers; i++) {
    if (P.kind[i] != LINEAR) continue;
    for (int o = tid; o < P.cout[i]; o += THREADS) {
      for (int w = 0; w < 4; w++) {
        uint32_t v = 0;
        for (int b = 0; b < 4; b++)
          v |= (uint32_t)__ldg(&rows[rb[4 * w + b] + P.at[i] + o]) << (8 * b);
        S[P.sh_d[i] + 4 * o + w] = (int32_t)v;
      }
    }
  }
  __syncthreads();
  if (tid < P.nlayers && P.kind[tid] == LINEAR) {
    int count = 0;
    for (int o = 0; o < P.cout[tid]; o++) {
      const int32_t *d = S + P.sh_d[tid] + 4 * o;
      if ((d[0] | d[1] | d[2] | d[3]) != 0) active[tid][count++] = o;
    }
    active_count[tid] = count;
  }
  __syncthreads();
  uint32_t lane[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int i = 0; i < P.nlayers; i++) {
    const uint32_t base = 7919u * (uint32_t)(i + 1);
    if (P.kind[i] == LINEAR) {
      if ((P.cin[i] & 7) == 0 && ((P.at[i] + P.cout[i]) & 1) == 0)
        linear_layer_pairs(rows, rb, S, i, tid, active[i], active_count[i], base, lane);
      else if ((P.cin[i] & 7) == 0)
        linear_layer<true>(rows, rb, S, i, tid, active[i], active_count[i], base, lane);
      else
        linear_layer<false>(rows, rb, S, i, tid, active[i], active_count[i], base, lane);
    } else {
      if (P.kind[i] == NARROW8) {
        // Four consecutive int8 factors occupy one aligned word. Transpose the
        // bytes of four rows so each resulting word holds one factor from four
        // samples, then dp4a sums those four samples with no 16-bit truncation.
        uint32_t acc4[4] = {0, 0, 0, 0};  // lane (4 * tid + j) & 7, fixed per thread and j
        for (int f0 = 4 * tid; f0 < P.nw[i]; f0 += 4 * THREADS) {
          int G0 = 0, G1 = 0, G2 = 0, G3 = 0;
#pragma unroll
          for (int r = 0; r < NEB_NB; r += 4) {
            const uint32_t p0 = __ldg((const uint32_t *)(rows + rb[r + 0] + P.at[i] + f0));
            const uint32_t p1 = __ldg((const uint32_t *)(rows + rb[r + 1] + P.at[i] + f0));
            const uint32_t p2 = __ldg((const uint32_t *)(rows + rb[r + 2] + P.at[i] + f0));
            const uint32_t p3 = __ldg((const uint32_t *)(rows + rb[r + 3] + P.at[i] + f0));
            const uint32_t t0 = (p0 & 0xffu) | ((p1 & 0xffu) << 8) | ((p2 & 0xffu) << 16) | ((p3 & 0xffu) << 24);
            const uint32_t t1 = ((p0 >> 8) & 0xffu) | (((p1 >> 8) & 0xffu) << 8) | (((p2 >> 8) & 0xffu) << 16) | (((p3 >> 8) & 0xffu) << 24);
            const uint32_t t2 = ((p0 >> 16) & 0xffu) | (((p1 >> 16) & 0xffu) << 8) | (((p2 >> 16) & 0xffu) << 16) | (((p3 >> 16) & 0xffu) << 24);
            const uint32_t t3 = ((p0 >> 24) & 0xffu) | (((p1 >> 24) & 0xffu) << 8) | (((p2 >> 24) & 0xffu) << 16) | (((p3 >> 24) & 0xffu) << 24);
            G0 = __dp4a((int)t0, (int)0x01010101, G0);
            G1 = __dp4a((int)t1, (int)0x01010101, G1);
            G2 = __dp4a((int)t2, (int)0x01010101, G2);
            G3 = __dp4a((int)t3, (int)0x01010101, G3);
          }
          const int G[4] = {G0, G1, G2, G3};
#pragma unroll
          for (int j = 0; j < 4; j++) {
            const int f = f0 + j;
            if (f >= P.nw[i]) break;
            const int cg = G[j] < -CLIPV ? -CLIPV : (G[j] > CLIPV ? CLIPV : G[j]);
            acc4[j] += (uint32_t)cg ^ ((uint32_t)f + base);
          }
        }
#pragma unroll
        for (int j = 0; j < 4; j++) lane[(4 * tid + j) & 7] += acc4[j];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
        // quads only on Pascal: measured +4..10 % on sm_61 but -1..3 % on sm_86 against the pairs below
      } else if (P.kind[i] == NARROW && (P.nw[i] & 3) == 0) {
        // int16 factors in quads: one 64-bit load per sample brings f .. f+3 (layers and rows are 8-byte aligned,
        // 2 * f is a multiple of 8), a quarter of the loads of the per-factor gather. Sums stay 32-bit (exact).
        // f = 4 * tid + 4 * THREADS * k, so the lanes are (4 * tid + j) & 7: four fixed accumulators.
        uint32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
        for (int f = 4 * tid; f < P.nw[i]; f += 4 * THREADS) {
          int G0 = 0, G1 = 0, G2 = 0, G3 = 0;
#pragma unroll
          for (int b = 0; b < NEB_NB; b++) {
            const uint2 v = __ldg((const uint2 *)(rows + rb[b] + P.at[i] + 2 * f));
            G0 += (int)(int16_t)(v.x & 0xffffu);
            G1 += (int)v.x >> 16;
            G2 += (int)(int16_t)(v.y & 0xffffu);
            G3 += (int)v.y >> 16;
          }
          const int c0 = G0 < -CLIPV ? -CLIPV : (G0 > CLIPV ? CLIPV : G0);
          const int c1 = G1 < -CLIPV ? -CLIPV : (G1 > CLIPV ? CLIPV : G1);
          const int c2 = G2 < -CLIPV ? -CLIPV : (G2 > CLIPV ? CLIPV : G2);
          const int c3 = G3 < -CLIPV ? -CLIPV : (G3 > CLIPV ? CLIPV : G3);
          acc0 += (uint32_t)c0 ^ ((uint32_t)f + base);
          acc1 += (uint32_t)c1 ^ ((uint32_t)(f + 1) + base);
          acc2 += (uint32_t)c2 ^ ((uint32_t)(f + 2) + base);
          acc3 += (uint32_t)c3 ^ ((uint32_t)(f + 3) + base);
        }
        lane[(4 * tid) & 7] += acc0;
        lane[(4 * tid + 1) & 7] += acc1;
        lane[(4 * tid + 2) & 7] += acc2;
        lane[(4 * tid + 3) & 7] += acc3;
#endif
      } else if (P.kind[i] == NARROW && (P.nw[i] & 1) == 0) {
        // int16 factors in pairs: one 32-bit load per sample brings entries f and f+1 (low / high half; at, the
        // row size and 2 * f are multiples of 4), halving the loads of the gather. Sums stay 32-bit, so they are
        // exact. f = 2 * tid + 2 * THREADS * k, so the lanes are (2 * tid + j) & 7: two fixed accumulators.
        uint32_t acc0 = 0, acc1 = 0;
        for (int f = 2 * tid; f < P.nw[i]; f += 2 * THREADS) {
          int G0 = 0, G1 = 0;
#pragma unroll
          for (int b = 0; b < NEB_NB; b++) {
            const uint32_t v = __ldg((const uint32_t *)(rows + rb[b] + P.at[i] + 2 * f));
            G0 += (int)(int16_t)(v & 0xffffu);
            G1 += (int)v >> 16;
          }
          const int c0 = G0 < -CLIPV ? -CLIPV : (G0 > CLIPV ? CLIPV : G0);
          const int c1 = G1 < -CLIPV ? -CLIPV : (G1 > CLIPV ? CLIPV : G1);
          acc0 += (uint32_t)c0 ^ ((uint32_t)f + base);
          acc1 += (uint32_t)c1 ^ ((uint32_t)(f + 1) + base);
        }
        lane[(2 * tid) & 7] += acc0;
        lane[(2 * tid + 1) & 7] += acc1;
      } else {
        uint32_t acc = 0;  // lane tid & 7 for every f of this thread
        for (int f = tid; f < P.nw[i]; f += THREADS) {
          int G = 0;
          if (P.kind[i] == NARROW) {
            for (int b = 0; b < NEB_NB; b++) G += __ldg((const int16_t *)(rows + rb[b] + P.at[i] + 2 * f));
          } else {
            for (int b = 0; b < NEB_NB; b++) G += __ldg((const int32_t *)(rows + rb[b] + P.at[i] + 4 * f));
          }
          const int cg = G < -CLIPV ? -CLIPV : (G > CLIPV ? CLIPV : G);
          acc += (uint32_t)cg ^ ((uint32_t)f + base);
        }
        lane[tid & 7] += acc;
      }
    }
  }
  // One reduction per warp instead of 256 shared-memory atomics per lane.
  // Addition is modulo 2^32, so changing the accumulation order is safe.
  {
    const int warp = tid >> 5;
    for (int k = 0; k < 8; k++) {
      const uint32_t v = warp_sum(lane[k]);
      if (!(tid & 31)) wsum[k][warp] = v;
    }
    __syncthreads();
    if (!tid) {
      for (int k = 0; k < 8; k++) {
        uint32_t v = 0;
        for (int w = 0; w < THREADS / 32; w++) v += wsum[k][w];
        lanes[k] = v + P.linear_zero[k];
      }
    }
  }
  if (tid == 0) {
    uint32_t blk[16], t[8];
    for (int i = 0; i < 8; i++) { blk[i] = lanes[i]; blk[8 + i] = key[i]; }
    blake3_compress(blake3_iv(), blk, 0, 64, BLAKE3_CHUNK_START | BLAKE3_CHUNK_END | BLAKE3_ROOT, t);
    bool ok = true;
    for (int k = 7; k >= 0; k--) {
      if (t[k] != bound[k]) { ok = t[k] < bound[k]; break; }
    }
    if (ok) {
      const uint32_t at = atomicAdd(hits, 1u);
      if (at < max_hits) hits[1 + at] = nonce;
    }
  }
}

struct neb2t_scan {
  int device, arch;
  TPlan plan;
  uint32_t n, capacity, max_hits;
  std::vector<uint8_t> values, labels, rows;
  uint8_t *d_rows;
  size_t d_rows_bytes;
  uint32_t *d_header, *d_bound, *d_hits, *d_cv;
  uint8_t cached_prefix[64];
  bool prefix_ready;
};

#define TRY(x) do { const cudaError_t e_ = (x); if (e_ != cudaSuccess) return (int)e_; } while (0)

static void make_plan(const neb2_arch *A, const int *wide, const int *narrow8, TPlan *p) {
  memset(p, 0, sizeof *p);
  p->nlayers = A->nlayers;
  int at = 0, sh = 0;
  for (int i = 0; i < A->nlayers; i++) {
    const neb2_layer &L = A->l[i];
    p->nw[i] = L.nw;
    p->cin[i] = L.cin;
    p->cout[i] = L.cout;
    p->at[i] = at;
    if (L.kind == NEB2_LINEAR) {
      p->kind[i] = LINEAR;
      for (int f = 0; f < L.nw; f++)
        p->linear_zero[f & 7] += (uint32_t)f + 7919u * (uint32_t)(i + 1);
      at += L.cout + L.cin;
      p->sh_d[i] = sh; sh += 4 * L.cout;
      p->sh_in[i] = sh; // unused: input words are reloaded per q, not staged
    } else {
      p->kind[i] = wide[i] ? WIDE : (narrow8[i] ? NARROW8 : NARROW);
      at += wide[i] ? 4 * L.nw : (narrow8[i] ? ((L.nw + 3) & ~3) : 2 * L.nw);
    }
    at = (at + 7) & ~7;  // 8-byte aligned layers and rows: the int16 quad path loads 64 bits per sample
  }
  p->row = at;
  p->shared_words = sh;
}

// Per-sample gradients of one model: the convolution entries as int32 (the table width of each layer is chosen
// afterwards from the range actually seen) and the linear factors (cout errors then cin inputs) as int8.
struct SampleGrads {
  std::vector<int32_t> conv;
  std::vector<int8_t> lin;
  int conv_w = 0, lin_w = 0;
};

// One evaluation of every sample with neb2_sample_table_row of the C reference (neb2_sample_grad without the linear
// gradients, which the row never reads: only their factors), on every host thread, recording the range of each
// convolution layer. The former build loop chose the widths by refilling the whole table once per layer that
// overflowed int8 or int16, i.e. up to four full gradient passes on mainnet models (~0.65 s each on an i7-7700K),
// with the GPU idle meanwhile.
static void sample_grads(neb2t_scan *s, const neb2_arch *A, const int8_t *weights, SampleGrads &sg,
                         int32_t lo[NEB2_MAX_LAYERS], int32_t hi[NEB2_MAX_LAYERS]) {
  neb2_model *m = (neb2_model *)calloc(1, sizeof *m);
  m->arch = s->arch;
  memcpy(m->w, weights, (size_t)A->nweights);
  for (int l = 0; l < A->nlayers; l++) {
    if (A->l[l].kind == NEB2_LINEAR) sg.lin_w += A->l[l].cout + A->l[l].cin;
    else sg.conv_w += A->l[l].nw;
  }
  sg.conv.assign((size_t)sg.conv_w * s->n, 0);
  sg.lin.assign((size_t)sg.lin_w * s->n, 0);
  unsigned nt = std::thread::hardware_concurrency();
  if (nt == 0) nt = 4;
  std::vector<int32_t> tlo((size_t)nt * NEB2_MAX_LAYERS, 0), thi((size_t)nt * NEB2_MAX_LAYERS, 0);
  std::vector<std::thread> th;
  for (unsigned t = 0; t < nt; t++)
    th.emplace_back([&, t] {
      std::vector<int32_t> g(NEB2_MAX_WEIGHTS);
      std::vector<int8_t> fac(2 * NEB2_MAX_WEIGHTS);
      int32_t *mylo = tlo.data() + (size_t)t * NEB2_MAX_LAYERS, *myhi = thi.data() + (size_t)t * NEB2_MAX_LAYERS;
      for (uint32_t i = t; i < s->n; i += nt) {
        neb2_sample_table_row(m, s->values.data() + (size_t)i * A->nin, s->labels[i], g.data(), fac.data());
        int32_t *c = sg.conv.data() + (size_t)i * sg.conv_w;
        int8_t *f = sg.lin.data() + (size_t)i * sg.lin_w;
        const int8_t *fp = fac.data();
        for (int l = 0; l < A->nlayers; l++) {
          const neb2_layer &L = A->l[l];
          if (L.kind == NEB2_LINEAR) {
            memcpy(f, fp, (size_t)(L.cout + L.cin));
            f += L.cout + L.cin;
            fp += L.cout + L.cin;
          } else {
            for (int k = 0; k < L.nw; k++) {
              const int32_t v = g[L.w_off + k];
              c[k] = v;
              if (v < mylo[l]) mylo[l] = v;
              if (v > myhi[l]) myhi[l] = v;
            }
            c += L.nw;
          }
        }
      }
    });
  for (auto &x : th) x.join();
  free(m);
  for (int l = 0; l < NEB2_MAX_LAYERS; l++) {
    lo[l] = hi[l] = 0;
    for (unsigned t = 0; t < nt; t++) {
      lo[l] = std::min(lo[l], tlo[(size_t)t * NEB2_MAX_LAYERS + l]);
      hi[l] = std::max(hi[l], thi[(size_t)t * NEB2_MAX_LAYERS + l]);
    }
  }
}

// Writes the row of every sample in the layout of s->plan from the stored gradients, on every host thread.
static void pack_rows(neb2t_scan *s, const neb2_arch *A, const SampleGrads &sg) {
  const TPlan &p = s->plan;
  s->rows.assign((size_t)p.row * s->n, 0);
  unsigned nt = std::thread::hardware_concurrency();
  if (nt == 0) nt = 4;
  std::vector<std::thread> th;
  for (unsigned t = 0; t < nt; t++)
    th.emplace_back([&, t] {
      for (uint32_t i = t; i < s->n; i += nt) {
        uint8_t *row = s->rows.data() + (size_t)i * p.row;
        const int32_t *c = sg.conv.data() + (size_t)i * sg.conv_w;
        const int8_t *f = sg.lin.data() + (size_t)i * sg.lin_w;
        for (int l = 0; l < A->nlayers; l++) {
          const neb2_layer &L = A->l[l];
          if (p.kind[l] == LINEAR) {
            memcpy(row + p.at[l], f, (size_t)(L.cout + L.cin));
            f += L.cout + L.cin;
            continue;
          }
          if (p.kind[l] == WIDE) {
            memcpy(row + p.at[l], c, (size_t)4 * L.nw);
          } else if (p.kind[l] == NARROW8) {
            uint8_t *o = row + p.at[l];
            for (int k = 0; k < L.nw; k++) o[k] = (uint8_t)(int8_t)c[k];
          } else {
            int16_t *o = (int16_t *)(row + p.at[l]);
            for (int k = 0; k < L.nw; k++) o[k] = (int16_t)c[k];
          }
          c += L.nw;
        }
      }
    });
  for (auto &x : th) x.join();
}

static int build(neb2t_scan *s, const int8_t *weights) {
  const neb2_arch *A = neb2_arch_get(s->arch);
  // the kernel lists the active outputs of a linear layer in a fixed shared array of 32
  for (int i = 0; i < A->nlayers; i++)
    if (A->l[i].kind == NEB2_LINEAR && A->l[i].cout > 32) return (int)cudaErrorInvalidValue;
  SampleGrads sg;
  int32_t lo[NEB2_MAX_LAYERS], hi[NEB2_MAX_LAYERS];
  sample_grads(s, A, weights, sg, lo, hi);
  // narrowest width that holds every entry of the layer: int8 (dp4a path), else int16, else int32
  int wide[NEB2_MAX_LAYERS] = {0}, n8[NEB2_MAX_LAYERS] = {0};
  for (int i = 0; i < A->nlayers; i++) {
    if (A->l[i].kind == NEB2_LINEAR) continue;
    if (lo[i] >= -128 && hi[i] <= 127) n8[i] = 1;
    else if (lo[i] < -32768 || hi[i] > 32767) wide[i] = 1;
  }
  make_plan(A, wide, n8, &s->plan);
  pack_rows(s, A, sg);
  if (s->rows.size() > UINT32_MAX) return (int)cudaErrorInvalidValue;
  TRY(cudaSetDevice(s->device));
  if (s->rows.size() != s->d_rows_bytes) {
    cudaFree(s->d_rows);
    s->d_rows = NULL;
    TRY(cudaMalloc(&s->d_rows, s->rows.size()));
    s->d_rows_bytes = s->rows.size();
  }
  return (int)cudaMemcpy(s->d_rows, s->rows.data(), s->rows.size(), cudaMemcpyHostToDevice);
}

extern "C" {

int neb2t_scan_open(int device, int arch, const uint8_t *values, const uint8_t *labels, uint32_t n, const int8_t *weights,
                   uint32_t capacity, uint32_t max_hits, neb2t_scan **out) {
  const neb2_arch *A = neb2_arch_get(arch);
  if (!A || n == 0 || capacity == 0 || max_hits == 0) return (int)cudaErrorInvalidValue;
  neb2t_scan *s = new neb2t_scan();
  s->device = device;
  s->arch = arch;
  s->n = n;
  s->capacity = capacity;
  s->max_hits = max_hits;
  s->values.assign(values, values + (size_t)n * A->nin);
  s->labels.assign(labels, labels + n);
  s->d_rows = NULL;
  s->d_rows_bytes = 0;
  s->d_header = s->d_bound = s->d_hits = NULL;
  memset(&s->plan, 0, sizeof s->plan);
  cudaError_t e = cudaSetDevice(device);
  if (e == cudaSuccess) e = cudaMalloc(&s->d_header, 80);
  if (e == cudaSuccess) e = cudaMalloc(&s->d_cv, 32);
  if (e == cudaSuccess) e = cudaMalloc(&s->d_bound, 32);
  if (e == cudaSuccess) e = cudaMalloc(&s->d_hits, ((size_t)max_hits + 1) * 4);
  int st = (int)e;
  if (st == 0) st = build(s, weights);
  if (st != 0) {
    neb2t_scan_close(s);
    return st;
  }
  *out = s;
  return 0;
}

int neb2t_scan_set_model(neb2t_scan *s, const int8_t *weights) { return build(s, weights); }

int neb2t_scan_run(neb2t_scan *s, const uint8_t *header, uint32_t start, uint32_t count, const uint8_t *bound, uint32_t *hits,
                  uint32_t *nhits) {
  if (count == 0 || count > s->capacity) return (int)cudaErrorInvalidValue;
  TRY(cudaSetDevice(s->device));
  TRY(cudaMemcpyToSymbol(P, &s->plan, sizeof s->plan));
  TRY(cudaMemcpy(s->d_header, header, 80, cudaMemcpyHostToDevice));
  if (!s->prefix_ready || memcmp(s->cached_prefix, header, 64) != 0) {
    header_prefix<<<1, 1>>>(s->d_header, s->d_cv);
    TRY(cudaGetLastError());
    memcpy(s->cached_prefix, header, 64);
    s->prefix_ready = true;
  }
  TRY(cudaMemcpy(s->d_bound, bound, 32, cudaMemcpyHostToDevice));
  TRY(cudaMemset(s->d_hits, 0, 4));
  for (uint32_t done = 0; done < count; done += CHUNK) {
    const uint32_t part = count - done < CHUNK ? count - done : CHUNK;
    table_kernel<<<part, THREADS, (size_t)s->plan.shared_words * 4>>>(s->d_rows, s->n, s->d_header, start + done, s->d_bound,
                                                                      s->d_hits, s->max_hits, s->d_cv);
    TRY(cudaGetLastError());
  }
  uint32_t found = 0;
  TRY(cudaMemcpy(&found, s->d_hits, 4, cudaMemcpyDeviceToHost));
  *nhits = found;
  if (found > 0)
    TRY(cudaMemcpy(hits, s->d_hits + 1, (size_t)(found < s->max_hits ? found : s->max_hits) * 4, cudaMemcpyDeviceToHost));
  return 0;
}

void neb2t_scan_close(neb2t_scan *s) {
  if (!s) return;
  cudaSetDevice(s->device);
  cudaFree(s->d_rows);
  cudaFree(s->d_header);
  cudaFree(s->d_cv);
  cudaFree(s->d_bound);
  cudaFree(s->d_hits);
  delete s;
}

}  // extern "C"
