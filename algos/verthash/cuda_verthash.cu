// SPDX-License-Identifier: GPL-3.0-or-later
//
// Verthash (Vertcoin) CUDA device kernels + host launchers.
//
// Three-kernel pipeline, structurally ported from the VerthashMiner CUDA kernel
// (src/vhCuda/verthash.cu, CryptoGraphics GPLv2), reworked to call the shared
// FIPS-202 permutation cuda/sha3_device.cuh (sha3_keccakf_1600) and to take mdiv
// as a runtime __constant__ instead of a baked #define:
//
//   1. verthash_gpu_precompute  (8 threads/job): the 8x SHA3-512 prehash-72
//      first permutation, one per header-byte-0 increment (i+1). Job-scoped.
//   2. verthash_gpu_sha3_256    (1 thread/nonce): SHA3-256(header||nonce) -> the
//      running 32-byte hash, stored to d_iohashes (also the IO output buffer).
//   3. verthash_gpu_io          (4 threads/nonce): finishes the 8x SHA3-512
//      (final-8 block) into a shared 512-byte subset, then the 4096 random
//      32-byte datafile reads with the cross-lane fnv1a accumulator sync (done
//      with a width-4 warp shuffle -- no block barrier).
//
// The IO kernel appends the nonce offset of any hash whose most-significant word
// (word 7) is <= target[7] to a results list (a safe superset filter -- '<='
// includes the boundary, so no real share is ever missed); the host re-verifies
// each candidate with the CPU oracle + fulltest before submit.
//
// Optimization note (Phase 6, RTX 3060): the algorithm is 4096 *dependent*
// random 32-byte reads into a 1.28 GiB buffer with no locality, so it is
// DRAM-random-access bound (~94 GB/s, ~735 kH/s). Two levers were measured and
// found NOT to help on this card and were dropped: (a) replacing the shared
// accumulator sync with a warp shuffle is perf-neutral (kept -- it is simply
// cleaner, removing the per-iteration block barriers); (b) splitting the SHA3
// subset into its own kernel to raise IO-kernel occupancy 50%->67% was slightly
// *slower* (extra launch + global subset round-trip outweigh the occupancy).

#include <cuda_runtime.h>
#include <stdint.h>
#include "cuda/sha3_device.cuh"

typedef unsigned int uint;

// header words 0..18 (bytes 0..75); word 18 (bytes 72..75) precedes the nonce.
__constant__ uint32_t c_vh_header[19];
// index modulus = ((datafile_size - 32) / 16) + 1.
__constant__ uint32_t c_vh_mdiv;

static __device__ __forceinline__ uint vh_fnv1a(const uint a, const uint b)
{
	return (a ^ b) * 0x1000193U;
}

static __device__ __forceinline__ uint vh_rotl32(const uint x, const uint n)
{
	return (x << n) | (x >> (32 - n));
}

// ---------------------------------------------------------------------------
// 1) 8x SHA3-512 prehash: absorb the 72-byte first block (header[0] += lane+1)
//    and run the first permutation. Store 8 states x 25 uint2 linearly.
__global__ void verthash_gpu_precompute(uint2 *kstates)
{
	const uint t = blockDim.x * blockIdx.x + threadIdx.x;   // 0..7
	if (t >= 8) return;

	uint2 st[25];
	#pragma unroll
	for (int i = 0; i < 25; i++) st[i] = make_uint2(0, 0);

	st[0].x = c_vh_header[0] + (t & 7) + 1; st[0].y = c_vh_header[1];
	st[1].x = c_vh_header[2];  st[1].y = c_vh_header[3];
	st[2].x = c_vh_header[4];  st[2].y = c_vh_header[5];
	st[3].x = c_vh_header[6];  st[3].y = c_vh_header[7];
	st[4].x = c_vh_header[8];  st[4].y = c_vh_header[9];
	st[5].x = c_vh_header[10]; st[5].y = c_vh_header[11];
	st[6].x = c_vh_header[12]; st[6].y = c_vh_header[13];
	st[7].x = c_vh_header[14]; st[7].y = c_vh_header[15];
	st[8].x = c_vh_header[16]; st[8].y = c_vh_header[17];

	sha3_keccakf_1600(st);

	uint2 *out = kstates + 25 * t;
	#pragma unroll
	for (int i = 0; i < 25; i++) out[i] = st[i];
}

// ---------------------------------------------------------------------------
// 2) SHA3-256(header || nonce) -> running 32-byte hash. One thread per nonce.
__global__ void verthash_gpu_sha3_256(uint2 *iohashes, const uint in18, const uint firstNonce)
{
	const uint gid = blockDim.x * blockIdx.x + threadIdx.x;
	const uint nonce = firstNonce + gid;

	uint2 st[25];
	#pragma unroll
	for (int i = 0; i < 25; i++) st[i] = make_uint2(0, 0);

	st[0].x = c_vh_header[0];  st[0].y = c_vh_header[1];
	st[1].x = c_vh_header[2];  st[1].y = c_vh_header[3];
	st[2].x = c_vh_header[4];  st[2].y = c_vh_header[5];
	st[3].x = c_vh_header[6];  st[3].y = c_vh_header[7];
	st[4].x = c_vh_header[8];  st[4].y = c_vh_header[9];
	st[5].x = c_vh_header[10]; st[5].y = c_vh_header[11];
	st[6].x = c_vh_header[12]; st[6].y = c_vh_header[13];
	st[7].x = c_vh_header[14]; st[7].y = c_vh_header[15];
	st[8].x = c_vh_header[16]; st[8].y = c_vh_header[17];

	st[9].x ^= in18; st[9].y ^= nonce;    // bytes 72..79
	st[10].x ^= 0x00000006U;              // byte 80 (0x06 pad)
	st[16].y ^= 0x80000000U;              // byte 135 (rate 136, final bit)

	sha3_keccakf_1600(st);

	iohashes[4 * gid + 0] = st[0];
	iohashes[4 * gid + 1] = st[1];
	iohashes[4 * gid + 2] = st[2];
	iohashes[4 * gid + 3] = st[3];
}

// ---------------------------------------------------------------------------
// 3) IO/mix. 4 lanes cooperate per nonce. WORK_SIZE threads/block.
#define VH_WORK_SIZE 64

struct vh_sha3_state_t { union { uint u[128]; uint2 u2[64]; }; };

__global__ void
__launch_bounds__(VH_WORK_SIZE)
verthash_gpu_io(uint2 *iohashes, const uint2 *__restrict__ kstates,
                const uint2 *__restrict__ memory, const uint firstNonce,
                uint *results, const uint target)
{
	const uint globalThId = blockDim.x * blockIdx.x + threadIdx.x;
	const uint lgr4id = (globalThId & (VH_WORK_SIZE - 1)) >> 2;  // local 4-lane group
	const uint gr4id  = globalThId >> 2;                          // nonce index
	const uint gr4e   = globalThId & 3;                           // lane 0..3

	__shared__ vh_sha3_state_t sha3St[VH_WORK_SIZE / 4];

	// --- SHA3-512 final-8: lane gr4e finishes states 2*gr4e and 2*gr4e+1 ---
	const uint nonce = firstNonce + gr4id;
	#pragma unroll
	for (int s3s = 0; s3s < 2; ++s3s) {
		const uint2 *ksrc = kstates + (2 * gr4e + s3s) * 25;
		uint2 st[25];
		#pragma unroll
		for (int i = 0; i < 25; ++i) st[i] = ksrc[i];

		st[0].x ^= c_vh_header[18]; st[0].y ^= nonce;   // final-8 block: bytes 72..79
		st[1].x ^= 0x00000006U;                          // byte 8 of block 2 (= byte 80)
		st[8].y ^= 0x80000000U;                          // byte 71 (rate 72, final bit)

		sha3_keccakf_1600(st);

		#pragma unroll
		for (int i = 0; i < 8; ++i)
			sha3St[lgr4id].u2[(gr4e * 16) + (s3s * 8) + i] = st[i];
	}
	// The subset lives in shared memory, written by the 4 lanes of this group and
	// read every IO iteration. Each 4-lane group is contained in one warp (4 | 32),
	// so a warp barrier suffices for visibility -- no block-wide __syncthreads.
	__syncwarp(0xffffffff);

	// --- IO/mix stage ---
	uint2 up1 = iohashes[globalThId];              // running hash words (2*gr4e, 2*gr4e+1)
	uint acc = 0x811c9dc5U;
	const uint mdiv = c_vh_mdiv;

	for (uint i = 0; i < 4096; ++i) {
		const uint s3idx  = i & 127;
		const uint rfac   = i >> 7;
		const uint seek   = vh_rotl32(sha3St[lgr4id].u[s3idx], rfac);
		const uint offset = (vh_fnv1a(seek, acc) % mdiv) << 1;   // uint2 units

		const uint2 v = memory[offset + gr4e];

		up1.x = vh_fnv1a(up1.x, v.x);
		up1.y = vh_fnv1a(up1.y, v.y);

		// cross-lane accumulator sync via width-4 warp shuffle: every lane folds
		// blob_off[0..7] (= lanes 0..3, {x,y}) in order. No block barrier, no
		// extra shared memory -- the shuffle carries its own intra-warp sync.
		acc = vh_fnv1a(acc, __shfl_sync(0xffffffff, v.x, 0, 4));
		acc = vh_fnv1a(acc, __shfl_sync(0xffffffff, v.y, 0, 4));
		acc = vh_fnv1a(acc, __shfl_sync(0xffffffff, v.x, 1, 4));
		acc = vh_fnv1a(acc, __shfl_sync(0xffffffff, v.y, 1, 4));
		acc = vh_fnv1a(acc, __shfl_sync(0xffffffff, v.x, 2, 4));
		acc = vh_fnv1a(acc, __shfl_sync(0xffffffff, v.y, 2, 4));
		acc = vh_fnv1a(acc, __shfl_sync(0xffffffff, v.x, 3, 4));
		acc = vh_fnv1a(acc, __shfl_sync(0xffffffff, v.y, 3, 4));
	}

	iohashes[globalThId] = up1;

	// lane 3 holds words 6,7; up1.y is word 7 (MSW). Superset filter on '<='.
	if (gr4e == 3 && up1.y <= target) {
		uint slot = atomicAdd(results, 1u);
		results[slot + 1] = gr4id;
	}
}

// ===========================================================================
// Host launchers.
extern "C" {

void verthash_cuda_set_header(const uint32_t header19[19])
{
	cudaMemcpyToSymbol(c_vh_header, header19, sizeof(uint32_t) * 19);
}

void verthash_cuda_set_mdiv(uint32_t mdiv)
{
	cudaMemcpyToSymbol(c_vh_mdiv, &mdiv, sizeof(uint32_t));
}

void verthash_cuda_precompute(uint2 *d_kstates)
{
	verthash_gpu_precompute<<<1, 8>>>(d_kstates);
}

// nonces MUST be a multiple of 256 (exact grids; the 4-lane IO kernel launches
// nonces*4 threads with no bounds guard). The host rounds throughput down.
void verthash_cuda_hash(uint2 *d_iohashes, const uint2 *d_kstates, const uint2 *d_memory,
                        uint32_t in18, uint32_t firstNonce, uint32_t nonces,
                        uint32_t *d_results, uint32_t target)
{
	verthash_gpu_sha3_256<<<nonces / 256, 256>>>(d_iohashes, in18, firstNonce);
	verthash_gpu_io<<<(nonces * 4) / VH_WORK_SIZE, VH_WORK_SIZE>>>(
		d_iohashes, d_kstates, d_memory, firstNonce, d_results, target);
}

} // extern "C"
