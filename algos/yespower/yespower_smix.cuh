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
template<uint32_t R, bool SFILL>
__device__ __forceinline__ void yp_smix1(uint32_t *B, uint32_t N, uint32_t *V,
                                         uint32_t *X, uint4 *S,
                                         uint32_t &b0, uint32_t &b1,
                                         uint32_t &b2, uint32_t &w4,
                                         const int j, const unsigned mask)
{
	const uint32_t s4 = 8u * R;            /* words-per-V-slot, in uint4 units */
	uint4 *X4 = (uint4 *)X;
	uint4 *V4 = (uint4 *)V;

	yp_shuffle_in(X, B, 2u * R, j);
	__syncwarp(mask);

	/* The r>1 pre-loop: each 128-byte chunk seeded from the previous one.
	 * Absent in 0.5; present in 1.0, and it is what makes X's chunks distinct
	 * before the main loop starts. */
	if (!SFILL) {
#pragma unroll 1
		for (uint32_t k = 1; k < R; k++) {
			for (uint32_t c = j; c < 8u; c += 4u)
				X4[k * 8u + c] = X4[(k - 1u) * 8u + c];
			__syncwarp(mask);
			blockmix_pwxform<1>(X + k * 32u, S, b0, b1, b2, w4, j, mask);
		}
	}

#pragma unroll 1
	for (uint32_t i = 0; i < N; i++) {
		/* V_i <- X */
		for (uint32_t c = j; c < s4; c += 4u) V4[i * s4 + c] = X4[c];

		if (i > 1) {
			const uint32_t jj = yp_wrap(yp_integerify<R>(X), i);
			for (uint32_t c = j; c < s4; c += 4u) {
				const uint4 v = V4[jj * s4 + c];
				X4[c].x ^= v.x; X4[c].y ^= v.y; X4[c].z ^= v.z; X4[c].w ^= v.w;
			}
		}
		__syncwarp(mask);

		if (SFILL) blockmix_salsa<2>(X, mask);
		else       blockmix_pwxform<R>(X, S, b0, b1, b2, w4, j, mask);
	}

	yp_shuffle_out(B, X, 2u * R, j);
	__syncwarp(mask);
}

/* smix2.  Nloop == 2 is the read-only pass (the reference skips the V write). */
template<uint32_t R>
__device__ __forceinline__ void yp_smix2(uint32_t *B, uint32_t N, uint32_t Nloop,
                                         uint32_t *V, uint32_t *X, uint4 *S,
                                         uint32_t &b0, uint32_t &b1,
                                         uint32_t &b2, uint32_t &w4,
                                         const int j, const unsigned mask)
{
	const uint32_t s4 = 8u * R;
	uint4 *X4 = (uint4 *)X;
	uint4 *V4 = (uint4 *)V;

	yp_shuffle_in(X, B, 2u * R, j);
	__syncwarp(mask);

#pragma unroll 1
	for (uint32_t i = 0; i < Nloop; i++) {
		const uint32_t jj = yp_integerify<R>(X) & (N - 1u);

		for (uint32_t c = j; c < s4; c += 4u) {
			uint4 v = V4[jj * s4 + c];
			X4[c].x ^= v.x; X4[c].y ^= v.y; X4[c].z ^= v.z; X4[c].w ^= v.w;
			if (Nloop != 2u) V4[jj * s4 + c] = X4[c];
		}
		__syncwarp(mask);

		blockmix_pwxform<R>(X, S, b0, b1, b2, w4, j, mask);
	}

	yp_shuffle_out(B, X, 2u * R, j);
	__syncwarp(mask);
}

#endif /* YESPOWER_SMIX_CUH */
