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

/* Legacy-name forwarders (x13 fugue) for the not-yet-migrated consumers
 * (x17/skydoge/hmq17, x21s, ghostrider, evohash, bastion); each drops out as
 * its family switches to the bare name. */
__host__ void x13_fugue512_cpu_init(int thr_id, uint32_t threads) { fugue512_cpu_init(thr_id, threads); }
__host__ void x13_fugue512_cpu_free(int thr_id) { fugue512_cpu_free(thr_id); }
__host__ void x13_fugue512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order) { fugue512_cpu_hash_64(thr_id, threads, startNounce, d_nonceVector, d_hash, order); }
