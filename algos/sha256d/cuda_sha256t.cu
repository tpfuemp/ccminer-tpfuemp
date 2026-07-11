/*
 * sha256(-t) CUDA implementation.
 * tpruvot 2017
 */

#include <stdio.h>
#include <stdint.h>
#include <memory.h>

#include <cuda_helper.h>
#include <miner.h>

#include "cuda/sha256_device.cuh"

__constant__ static uint32_t __align__(8) c_midstate76[8];
__constant__ static uint32_t __align__(8) c_dataEnd80[4];

__constant__ static uint32_t __align__(8) c_target[2];

static uint32_t* d_resNonces[MAX_GPUS] = { 0 };

extern bool sha256_device_selftest(int thr_id);

// ------------------------------------------------------------------------------------------------

__device__ __forceinline__
uint64_t cuda_swab32ll(uint64_t x) {
	return MAKE_ULONGLONG(cuda_swab32(_LODWORD(x)), cuda_swab32(_HIDWORD(x)));
}

__global__
void sha256t_gpu_hash_shared(const uint32_t threads, const uint32_t startNonce, uint32_t *resNonces)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	__shared__ uint32_t s_K[64*4];
	if (threadIdx.x < 64U) s_K[threadIdx.x] = c_sha256_K[threadIdx.x];

	if (thread < threads)
	{
		const uint32_t nonce = startNonce + thread;

		uint32_t dat[16];
		AS_UINT2(dat) = AS_UINT2(c_dataEnd80);
		dat[ 2] = c_dataEnd80[2];
		dat[ 3] = nonce;
		dat[ 4] = 0x80000000;
		dat[15] = 0x280;
		#pragma unroll
		for (int i=5; i<15; i++) dat[i] = 0;

		uint32_t buf[8];
		#pragma unroll
		for (int i=0; i<8; i+=2) AS_UINT2(&buf[i]) = AS_UINT2(&c_midstate76[i]);

		sha256_transform_full(dat, buf, s_K);

		// second sha256

		#pragma unroll
		for (int i=0; i<8; i++) dat[i] = buf[i];
		dat[8] = 0x80000000;
		#pragma unroll
		for (int i=9; i<15; i++) dat[i] = 0;
		dat[15] = 0x100;

		#pragma unroll
		for (int i=0; i<8; i++) buf[i] = c_sha256_H[i];

		sha256_transform_full(dat, buf, s_K);

		// last sha256

		#pragma unroll
		for (int i=0; i<8; i++) dat[i] = buf[i];
		dat[8] = 0x80000000;
		#pragma unroll
		for (int i=9; i<15; i++) dat[i] = 0;
		dat[15] = 0x100;

		#pragma unroll
		for (int i=0; i<8; i++) buf[i] = c_sha256_H[i];

		// truncated final rounds: only buf[6]/buf[7] are valid, compared
		// directly against the target; candidates are fully re-hashed on the CPU
		sha256_final_to_target(dat, buf, s_K);

		// valid nonces
		uint64_t high = cuda_swab32ll(((uint64_t*)buf)[3]);
		if (high <= c_target[0]) {
			resNonces[1] = atomicExch(resNonces, nonce);
		}
	}
}

__host__
void sha256t_init(int thr_id)
{
	cuda_get_arch(thr_id);
	sha256_device_selftest(thr_id);
	CUDA_SAFE_CALL(cudaMalloc(&d_resNonces[thr_id], 2*sizeof(uint32_t)));
}

__host__
void sha256t_free(int thr_id)
{
	if (d_resNonces[thr_id]) cudaFree(d_resNonces[thr_id]);
	d_resNonces[thr_id] = NULL;
}

__host__
void sha256t_setBlock_80(uint32_t *pdata, uint32_t *ptarget)
{
	uint32_t _ALIGN(64) in[16], buf[8], end[4];
	for (int i=0;i<16;i++) in[i] = cuda_swab32(pdata[i]);
	for (int i=0;i<8;i++) buf[i] = h_sha256_H[i];
	for (int i=0;i<4;i++) end[i] = cuda_swab32(pdata[16+i]);
	sha256_transform_full(in, buf, h_sha256_K);

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_midstate76, buf, 32, 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_dataEnd80,  end, sizeof(end), 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_target, &ptarget[6], 8, 0, cudaMemcpyHostToDevice));
}

__host__
void sha256t_hash_80(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *resNonces)
{
	const uint32_t threadsperblock = 128;

	dim3 grid(threads/threadsperblock);
	dim3 block(threadsperblock);

	CUDA_SAFE_CALL(cudaMemset(d_resNonces[thr_id], 0xFF, 2 * sizeof(uint32_t)));
	cudaDeviceSynchronize();
	sha256t_gpu_hash_shared <<<grid, block>>> (threads, startNonce, d_resNonces[thr_id]);
	cudaDeviceSynchronize();

	CUDA_SAFE_CALL(cudaMemcpy(resNonces, d_resNonces[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
	if (resNonces[0] == resNonces[1]) {
		resNonces[1] = UINT32_MAX;
	}
}
