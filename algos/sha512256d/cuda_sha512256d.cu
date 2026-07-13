// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * sha512256d CUDA implementation (double SHA-512/256 over an 80-byte header).
 *
 * Correctness-first port from the cpuminer-opt scalar reference
 * (algo/sha/sha512256d-4way.c). An 80-byte message fits one 128-byte
 * SHA-512 block, so each nonce costs exactly two transforms. Rounds 0..8 of
 * hash1 plus the constant halves of round 9 are hoisted to the host per job
 * (c_pre); the kernel resumes at round 9's `+ w9`. The GPU only screens
 * candidates against the target's high qword; the host recomputes the full
 * double hash and runs fulltest before submit.
 *
 * Donor-kernel note (2026-07-13): radifier's fully hand-unrolled Radiant
 * kernel (ccminer-radiator cuda_rad.cu, d40c089) was transcribed verbatim,
 * proven bit-correct on GPU, and A/B-measured at +0.2% vs this kernel on
 * sm_86 / CUDA 11.8 — a wash. Kept this 8x-smaller library-based kernel.
 * Useful facts from that experiment: vhash[7] = swab32(lo32(q3)); with the
 * donor's truncated tail, lo32(q3) = r + 0x247f2d73 (fold of the omitted
 * final-round K, d-slot and IV feed-forward terms); and --benchmark rescans
 * the same 2^30-nonce window, so diff-1 screens are expected to find
 * nothing there 78% of the time.
 */

#include <stdio.h>
#include <stdint.h>
#include <memory.h>

#include <cuda_helper.h>
#include <miner.h>

#include "cuda/sha512_device.cuh"

#define TPB 256

/* c_header[0..8] = 64-bit big-endian message words w0..w8 of the header
 * (built host-side from the 32-bit work->data words); c_header[9] = w9 with
 * the nonce half zeroed (nbits << 32) — the kernel ORs in its nonce.
 * c_pre = per-job prehash (registers after round 8 + round-9 t1c/t2c):
 * w0..w8 stay uploaded because the rounds 16+ schedule still reads them. */
__constant__ uint64_t c_header[10];
__constant__ uint64_t c_pre[10];

static uint32_t* d_resNonces[MAX_GPUS] = { 0 };

extern bool sha512256d_device_selftest(int thr_id);

// ------------------------------------------------------------------------------------------------

__host__
void sha512256d_init(int thr_id)
{
	cuda_get_arch(thr_id);
	sha512256d_device_selftest(thr_id);
	CUDA_SAFE_CALL(cudaMalloc(&d_resNonces[thr_id], 2*sizeof(uint32_t)));
}

__host__
void sha512256d_free(int thr_id)
{
	if (d_resNonces[thr_id]) cudaFree(d_resNonces[thr_id]);
	d_resNonces[thr_id] = NULL;
}

__host__
void sha512256d_setBlock_80(const uint32_t *pdata)
{
	uint64_t hdr[10], pre[10];
	for (int i = 0; i < 9; i++)
		hdr[i] = ((uint64_t) pdata[2*i] << 32) | pdata[2*i + 1];
	hdr[9] = (uint64_t) pdata[18] << 32;
	sha512_prehash_split_host(hdr, pre);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_header, hdr, sizeof(hdr), 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_pre, pre, sizeof(pre), 0, cudaMemcpyHostToDevice));
}

__global__ __launch_bounds__(TPB)
void sha512256d_gpu_hash(const uint32_t threads, const uint32_t startNonce, uint32_t *result, const uint64_t targ_q3)
{
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;
	if (thread < threads)
	{
		const uint32_t nonce = startNonce + thread;
		uint64_t w[16], st[8];

		// hash1 = SHA512/256(header80): single block, nonce in the low half
		// of w9; rounds 0..8 + the constant halves of round 9 are per-job
		// host work (c_pre), the kernel resumes at round 9's `+ w9`.
		#pragma unroll
		for (int i = 0; i < 9; i++) w[i] = c_header[i];
		w[9] = c_header[9] | nonce;
		w[10] = 0x8000000000000000ULL;
		#pragma unroll
		for (int i = 11; i < 15; i++) w[i] = 0;
		w[15] = 640;
		sha512_256_init_state(st, c_sha512_256_H);
		sha512_transform_80_from_pre9(w, c_pre, st, c_sha512_K);

		// hash2 = SHA512/256(hash1 truncated to 32 bytes = words 0..3).
		// Full transform on purpose: a truncated q3-only final (rounds 77..79
		// elided) was A/B-measured NEGATIVE (-1..-4%) on sm_86 — the irregular
		// tail hurts codegen more than 3 rounds save. Don't re-add it.
		#pragma unroll
		for (int i = 0; i < 4; i++) w[i] = st[i];
		w[4] = 0x8000000000000000ULL;
		#pragma unroll
		for (int i = 5; i < 15; i++) w[i] = 0;
		w[15] = 256;
		sha512_256_init_state(st, c_sha512_256_H);
		sha512_transform_full(w, st, c_sha512_K);

		// share value's high qword is the byte-reversed q3 (fulltest word order)
		if (cuda_swab64(st[3]) <= targ_q3)
		{
			uint32_t tmp = atomicCAS(result, UINT32_MAX, nonce);
			if (tmp != UINT32_MAX)
				result[1] = nonce;
		}
	}
}

__host__
void sha512256d_hash_80(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t targ_q3, uint32_t *resNonces)
{
	dim3 grid((threads + TPB - 1) / TPB);
	dim3 block(TPB);

	CUDA_SAFE_CALL(cudaMemset(d_resNonces[thr_id], 0xFF, 2 * sizeof(uint32_t)));
	cudaDeviceSynchronize();
	sha512256d_gpu_hash <<<grid, block>>> (threads, startNonce, d_resNonces[thr_id], targ_q3);
	cudaDeviceSynchronize();

	CUDA_SAFE_CALL(cudaMemcpy(resNonces, d_resNonces[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
	if (resNonces[0] == resNonces[1]) {
		resNonces[1] = UINT32_MAX;
	}
}
