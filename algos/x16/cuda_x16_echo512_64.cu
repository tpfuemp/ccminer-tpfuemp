/**
 * ECHO-512 64-byte launcher (optimised alexis formulation) — thin wrapper.
 *
 * The alexis ECHO-512 device implementation (c_echo_AES tables,
 * echo_round_alexis, echo512_hash_64_alexis) lives in
 * cuda/echo512_device.cuh (docs/coding-guideline.md §3).
 */

#include <cuda_helper.h>

#include "cuda/echo512_device.cuh"

__global__ __launch_bounds__(128, 5) /* will force 80 registers */
static void x16_echo512_gpu_hash_64(uint32_t threads, uint32_t* g_hash)
{
	__shared__ uint32_t sharedMemory[4][256];

	echo_aes_gpu_init128(sharedMemory);
	__syncthreads(); // barrier: shared AES table filled cooperatively

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t *Hash = &g_hash[thread<<4];
		uint32_t hash[16];

		*(uint2x4*)&hash[ 0] = __ldg4((uint2x4*)&Hash[ 0]);
		*(uint2x4*)&hash[ 8] = __ldg4((uint2x4*)&Hash[ 8]);

		echo512_hash_64_alexis(sharedMemory, hash);

		*(uint2x4*)&Hash[ 0] = *(uint2x4*)&hash[ 0];
		*(uint2x4*)&Hash[ 8] = *(uint2x4*)&hash[ 8];
	}
}

/* Unit self-test for the alexis section of cuda/echo512_device.cuh
 * (docs/coding-guideline.md §7 layer 1), defined in cuda/xfamily_selftest.cu.
 * No cpu_init exists for this stage, so the launcher runs it once (static
 * guard inside). */
extern bool echo512_alexis_device_selftest(int thr_id);

__host__
void echo512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash)
{
	echo512_alexis_device_selftest(thr_id);

	const uint32_t threadsperblock = 128;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	x16_echo512_gpu_hash_64 <<<grid, block>>> (threads, d_hash);
}

/* Legacy forwarder — consumers not yet migrated (x17, skydoge, x21s,
 * ghostrider) still call this name; remove once they call echo512_cpu_hash_64
 * directly. */
__host__
void x16_echo512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash)
{
	echo512_cpu_hash_64(thr_id, threads, d_hash);
}
