/* yespower 1.0 SMix -- the memory-hard core, 4 threads per hash instance.
 *
 * Transcribed from sph/yespower_ref.c (smix1 / smix2 / smix) and verified against
 * those functions by an out-of-tree KAT.
 *
 * SCOPE: `X` and `V` are plain memory here.  A fully tuned kernel wants X
 * register-resident and S in shared memory, but that is an optimisation on top of
 * a correct algorithm; this file is the correctness baseline it must keep matching.
 *
 * LANE LAYOUT: thread j owns uint4 index == j (mod 4) of every array, i.e. the
 * contiguous words 16k+4j .. 16k+4j+3 of each 64-byte block.  That is exactly
 * blockmix_pwxform's layout, so the copies and XORs below need no cross-lane
 * traffic at all; the only sync points are the ones inside blockmix.
 */

#ifndef YESPOWER_SMIX_CUH
#define YESPOWER_SMIX_CUH

#include "cuda/pwxform_device.cuh"   /* also pulls in salsa_device.cuh */

/* V cache policy.
 *
 * V is ~22.4 MB per hash and genuinely streaming: smix1 writes each slot
 * once, smix2 reads them at cryptographic random, and with hundreds of
 * instances resident nothing stays cached.  `.cs` (evict-first) is therefore
 * offered as a compile-time ablation, not a default, and must be validated on
 * each arch: the cache hierarchies differ. */
#ifdef YP_V_STREAM
__device__ __forceinline__ uint4 yp_vld(const uint4 *p) { return __ldcs(p); }
__device__ __forceinline__ void  yp_vst(uint4 *p, const uint4 v) { __stcs(p, v); }
#else
__device__ __forceinline__ uint4 yp_vld(const uint4 *p) { return *p; }
__device__ __forceinline__ void  yp_vst(uint4 *p, const uint4 v) { *p = v; }
#endif

/* Test-only fault #3 (live only above one quad): remove the full-block
 * barriers.  At WIDTH == 4 these are redundant (one warp, 4 lanes, private state) and ptxas
 * elides them, which is why YP_PWX_FAULT_NO_SYNCWARP is inert there. Above 4 the
 * spare threads write V/X that lanes 0-3 then read, so these barriers ARE
 * load-bearing and deleting them must corrupt the digest. Neither build system
 * defines this. */
#ifdef YP_FAULT_NO_WIDTH_SYNC
#define YP_WSYNC(m) ((void)0)
#else
#define YP_WSYNC(m) __syncwarp(m)
#endif

/* V-load batching.
 *
 * The V loop issues WIDTH independent uint4 loads; staging a batch into
 * registers before use lets them issue together rather than load-use
 * adjacent.  Costs YP_VBATCH * 4 registers, and on a kernel carrying
 * `__launch_bounds__` any register change moves blocks/SM -- so check REG
 * per arch before reading a result.  sm_61 is the tight one: 16 blocks/SM
 * leaves only 128 registers. */
#ifndef YP_VBATCH
#define YP_VBATCH 1
#endif

__device__ __forceinline__ uint32_t yp_p2floor(uint32_t x)
{
	uint32_t y;
	while ((y = x & (x - 1))) x = y;
	return x;
}

/* wrap(x, i): fold x into [0, i). */
__device__ __forceinline__ uint32_t yp_wrap(uint32_t x, uint32_t i)
{
	const uint32_t n = yp_p2floor(i);
	return (x & (n - 1)) + (i - n);
}

/* integerify(X, r) = X[(2r-1) * 16], the first word of the LAST 64-byte block.
 * Read from memory rather than shuffled out of registers: blockmix_pwxform ends
 * with a __syncwarp, so the value is visible to every lane by here. */
template<uint32_t R>
__device__ __forceinline__ uint32_t yp_integerify(const uint32_t *X)
{
	return X[(2u * R - 1u) * 16u];
}

/* B <-> X use the SIMD shuffle:  X[k*16 + i] = B[k*16 + (i * 5 % 16)].
 * Applied per 64-byte block; the inverse writes B[k*16 + (i*5%16)] = X[k*16+i].
 * Lane j takes i = j, j+4, j+8, j+12 -- a permutation of a permutation, so each
 * lane still reads and writes a disjoint quarter. */
__device__ __forceinline__ void yp_shuffle_in(uint32_t *X, const uint32_t *B,
                                              uint32_t blocks, int j)
{
	for (uint32_t k = 0; k < blocks; k++)
#pragma unroll
		for (int m = 0; m < 4; m++) {
			const int i = j + 4 * m;
			X[k * 16 + i] = B[k * 16 + (i * 5 % 16)];
		}
}

__device__ __forceinline__ void yp_shuffle_out(uint32_t *B, const uint32_t *X,
                                               uint32_t blocks, int j)
{
	for (uint32_t k = 0; k < blocks; k++)
#pragma unroll
		for (int m = 0; m < 4; m++) {
			const int i = j + 4 * m;
			B[k * 16 + (i * 5 % 16)] = X[k * 16 + i];
		}
}

/* smix1.
 *
 * SFILL selects the S-box fill variant: the reference switches on `V == ctx->S`
 * and then uses blockmix_salsa instead of blockmix_pwxform, with r = 1.  Making
 * that a template parameter keeps the branch out of the inner loop and lets R=1
 * fold.
 */
template<uint32_t R, bool SFILL, int PLACE = PWX_S_SHARED, int WIDTH = 4>
__device__ __forceinline__ void yp_smix1(uint32_t *B, uint32_t N, uint32_t *V,
                                         uint32_t *X, uint4 *S,
                                         uint32_t &b0, uint32_t &b1,
                                         uint32_t &b2, uint32_t &w4,
                                         const int j, const unsigned mask,
                                         const int tid, const unsigned wmask)
{
	const uint32_t s4 = 8u * R;            /* words-per-V-slot, in uint4 units */
	uint4 *X4 = (uint4 *)X;
	uint4 *V4 = (uint4 *)V;

	yp_shuffle_in(X, B, 2u * R, j);
	/* `wmask`, not `mask`: this barrier is block scope and all WIDTH threads
	 * reach it, while `mask` covers only the four pwxform lanes.  A thread must
	 * have its own bit set in the mask it passes. */
	__syncwarp(wmask);

	/* The r>1 pre-loop: each 128-byte chunk seeded from the previous one.
	 * Absent in 0.5; present in 1.0, and it is what makes X's chunks distinct
	 * before the main loop starts. */
	if (!SFILL) {
#pragma unroll 1
		for (uint32_t k = 1; k < R; k++) {
			for (uint32_t c = (uint32_t)tid; c < 8u; c += (uint32_t)WIDTH)
				X4[k * 8u + c] = X4[(k - 1u) * 8u + c];
			YP_WSYNC(wmask);
			if (tid < PWX_GATHER)
				blockmix_pwxform<1, PLACE>(X + k * 32u, S, b0, b1, b2, w4, j, mask);
			YP_WSYNC(wmask);
		}
	}

#pragma unroll 1
	for (uint32_t i = 0; i < N; i++) {
		/* V_i <- X. Strided by WIDTH, not by 4: at WIDTH == 4 this is the original
		* loop verbatim (tid == j there), and above it the spare threads -- which
		* sit out pwxform entirely -- carry their share of the transfer. The SoA
		* layout already made lanes touch consecutive uint4s; only the
		* width was wrong, so this moves 16*WIDTH bytes per instruction. */
		for (uint32_t c = (uint32_t)tid; c < s4; c += (uint32_t)WIDTH)
			yp_vst(&V4[i * s4 + c], X4[c]);

		if (i > 1) {
			const uint32_t jj = yp_wrap(yp_integerify<R>(X), i);
			const uint32_t step = (uint32_t)WIDTH * (uint32_t)YP_VBATCH;
			for (uint32_t c0 = (uint32_t)tid; c0 < s4; c0 += step) {
				uint4 vb[YP_VBATCH];
#pragma unroll
				for (int k = 0; k < YP_VBATCH; k++) {
					const uint32_t c = c0 + (uint32_t)k * (uint32_t)WIDTH;
					if (c < s4) vb[k] = yp_vld(&V4[jj * s4 + c]);
				}
#pragma unroll
				for (int k = 0; k < YP_VBATCH; k++) {
					const uint32_t c = c0 + (uint32_t)k * (uint32_t)WIDTH;
					if (c < s4) {
						X4[c].x ^= vb[k].x; X4[c].y ^= vb[k].y;
						X4[c].z ^= vb[k].z; X4[c].w ^= vb[k].w;
			}
		}
			}
		}
		/* Full-block barrier: the XOR above is spread over WIDTH threads and
		* blockmix's 4 lanes read all of X. */
		YP_WSYNC(wmask);

		if (tid < PWX_GATHER) {
		if (SFILL) blockmix_salsa<2>(X, mask);
			else       blockmix_pwxform<R, PLACE>(X, S, b0, b1, b2, w4, j, mask);
		}
		/* Required once WIDTH > 4 and absent from the 4-thread original for a
		* real reason: at WIDTH == 4 thread j owns uint4 index j (mod 4) in BOTH
		* the V loop and blockmix, so each thread only ever re-read its own words.
		* Widening breaks that alignment -- thread tid now reads words blockmix's
		* lane (c mod 4) wrote -- so the next iteration needs this barrier. */
		YP_WSYNC(wmask);
	}

	yp_shuffle_out(B, X, 2u * R, j);
	__syncwarp(wmask);	/* block scope: all WIDTH threads reach this */
}

/* smix2.  Nloop == 2 is the read-only pass (the reference skips the V write). */
template<uint32_t R, int PLACE = PWX_S_SHARED, int WIDTH = 4>
__device__ __forceinline__ void yp_smix2(uint32_t *B, uint32_t N, uint32_t Nloop,
                                         uint32_t *V, uint32_t *X, uint4 *S,
                                         uint32_t &b0, uint32_t &b1,
                                         uint32_t &b2, uint32_t &w4,
                                         const int j, const unsigned mask,
                                         const int tid, const unsigned wmask)
{
	const uint32_t s4 = 8u * R;
	uint4 *X4 = (uint4 *)X;
	uint4 *V4 = (uint4 *)V;

	yp_shuffle_in(X, B, 2u * R, j);
	__syncwarp(wmask);	/* block scope: all WIDTH threads reach this */

#pragma unroll 1
	for (uint32_t i = 0; i < Nloop; i++) {
		const uint32_t jj = yp_integerify<R>(X) & (N - 1u);

		const uint32_t step2 = (uint32_t)WIDTH * (uint32_t)YP_VBATCH;
		for (uint32_t c0 = (uint32_t)tid; c0 < s4; c0 += step2) {
			uint4 vb[YP_VBATCH];
#pragma unroll
			for (int k = 0; k < YP_VBATCH; k++) {
				const uint32_t c = c0 + (uint32_t)k * (uint32_t)WIDTH;
				if (c < s4) vb[k] = yp_vld(&V4[jj * s4 + c]);
		}
#pragma unroll
			for (int k = 0; k < YP_VBATCH; k++) {
				const uint32_t c = c0 + (uint32_t)k * (uint32_t)WIDTH;
				if (c < s4) {
					X4[c].x ^= vb[k].x; X4[c].y ^= vb[k].y;
					X4[c].z ^= vb[k].z; X4[c].w ^= vb[k].w;
					if (Nloop != 2u) yp_vst(&V4[jj * s4 + c], X4[c]);
				}
			}
		}
		YP_WSYNC(wmask);

		if (tid < PWX_GATHER)
			blockmix_pwxform<R, PLACE>(X, S, b0, b1, b2, w4, j, mask);
		YP_WSYNC(wmask);
	}

	yp_shuffle_out(B, X, 2u * R, j);
	__syncwarp(wmask);	/* block scope: all WIDTH threads reach this */
}

#endif /* YESPOWER_SMIX_CUH */
