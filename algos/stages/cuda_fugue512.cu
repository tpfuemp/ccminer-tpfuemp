/*
 * fugue512 x13 kernel implementation — thin wrapper.
 *
 * The Fugue-512 device implementation (mixtab constants, TIX4/CMIX36/SMIX
 * macros, shared-table fill and fugue512_hash_64) lives in
 * cuda/fugue512_device.cuh (docs/coding-guideline.md §3). The donor's
 * texture apparatus (mixTab0Tex/d_textures) is gone: the shared fill reads
 * the same table from constant memory.
 */

#include <cuda_helper.h>

#define TPB 256

#include "cuda/fugue512_device.cuh"

/***************************************************/
// GPU Hash Function
__global__
__launch_bounds__(TPB)
void fugue512_gpu_hash_64(uint32_t threads, uint64_t *g_hash)
{
	__shared__ uint32_t mixtabs[1024];

	// load shared mem (with 256 threads)
	fugue512_load_shared(mixtabs);

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads)
	{
		const size_t hashPosition = thread;
		uint64_t *pHash = &g_hash[hashPosition<<3];
		uint32_t Hash[16];

		#pragma unroll 4
		for(int i = 0; i < 4; i++)
			AS_UINT4(&Hash[i*4]) = AS_UINT4(&pHash[i*2]);

		fugue512_hash_64(mixtabs, Hash);

		#pragma unroll 4
		for(int i = 0; i < 4; i++)
			AS_UINT4(&pHash[i*2]) = AS_UINT4(&Hash[i*4]);
	}
}

/***************************************************/
// Terminal variant: compute fugue, compare the high 64 bits of the result
// against the target on-device, and record up to two found nonces (thread
// indices) via an atomicExch chain into resNonce -- eliding the d_hash store
// plus the separate cuda_check_hash / cuda_check_hash_suppl passes. Used where
// fugue is the last stage of a fixed chain (x13). Not truncated (computes the
// full fugue like the plain kernel), so it stays bit-identical to the CPU
// reference which re-verifies every hit.
__global__
__launch_bounds__(TPB)
void fugue512_gpu_hash_64_final(uint32_t threads, uint64_t *g_hash, uint32_t *resNonce, const uint64_t target)
{
	__shared__ uint32_t mixtabs[1024];

	fugue512_load_shared(mixtabs);

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads)
	{
		uint64_t *pHash = &g_hash[thread<<3];
		uint32_t Hash[16];

		#pragma unroll 4
		for(int i = 0; i < 4; i++)
			AS_UINT4(&Hash[i*4]) = AS_UINT4(&pHash[i*2]);

		fugue512_hash_64(mixtabs, Hash);

		if (*(uint64_t*)&Hash[6] <= target) {
			uint32_t tmp = atomicExch(&resNonce[0], thread);
			if (tmp != UINT32_MAX)
				resNonce[1] = tmp;
		}
	}
}

/* Unit self-test for cuda/fugue512_device.cuh (docs/coding-guideline.md §7
 * layer 1), defined in cuda/xfamily_selftest.cu. */
extern bool fugue512_device_selftest(int thr_id);

__host__
void fugue512_cpu_init(int thr_id, uint32_t threads)
{
	fugue512_device_selftest(thr_id);
}

__host__
void fugue512_cpu_free(int thr_id)
{
}

__host__
void fugue512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
	dim3 grid((threads + TPB-1) / TPB);
	dim3 block(TPB);

	fugue512_gpu_hash_64 <<<grid, block>>> (threads, (uint64_t*)d_hash);
}

__host__
void fugue512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t *d_resNonce, const uint64_t target)
{
	dim3 grid((threads + TPB-1) / TPB);
	dim3 block(TPB);

	fugue512_gpu_hash_64_final <<<grid, block>>> (threads, (uint64_t*)d_hash, d_resNonce, target);
}
