/*
 * Whirlpool-512 shared device library (docs/coding-guideline.md §3),
 * djm34/tpruvot/SP/Provos Alexis formulation (see x15/cuda_x15_whirlpool.cu
 * for authorship and license).
 *
 * MODE-SWITCHED tables: the round tables are UPLOADED, not static-init —
 * plain Whirlpool (mode 0, x15/x16 chains) and legacy whirlpool1 (mode 1,
 * standalone whirlpool algo) share the same device symbols. Every consumer
 * TU must call whirlpool512_init_tables(mode) once (host, per-TU copy of the
 * symbols) before launching kernels that use these functions.
 *
 * whirlpool512_hash_64 needs the 7x256 uint2 shared table: kernels declare
 * __shared__ uint2 sharedMemory[7][256], call
 * whirlpool512_load_shared(sharedMemory) (cooperative, threads < 256) and
 * MUST __syncthreads() before hashing.
 */

#ifndef CUDA_WHIRLPOOL512_DEVICE_CUH
#define CUDA_WHIRLPOOL512_DEVICE_CUH

#include <cuda_helper.h>
#ifndef CUDA_LYRA_VECTOR_H
#include <cuda_vectors.h>
#include <cuda_vector_uint2x4.h>
#endif

#include "x15/cuda_whirlpool_tables.cuh"

#define whirl_xor3x(a,b,c) (a^b^c)

__device__ static uint64_t whirl_b0[256];
__device__ static uint64_t whirl_b7[256];

__constant__ static uint2 whirl_precomputed_round_key_64[72];

/**
 * Round constants.
 */
__device__ static uint2 whirl_InitVector_RC[10];

//--------START OF WHIRLPOOL DEVICE MACROS---------------------------------------------------------------------------
__device__ __forceinline__
void static TRANSFER(uint2 *const __restrict__ dst,const uint2 *const __restrict__ src){
	dst[0] = src[ 0];
	dst[1] = src[ 1];
	dst[2] = src[ 2];
	dst[3] = src[ 3];
	dst[4] = src[ 4];
	dst[5] = src[ 5];
	dst[6] = src[ 6];
	dst[7] = src[ 7];
}

__device__ __forceinline__
static uint2 d_ROUND_ELT_LDG(const uint2 sharedMemory[7][256],const uint2 *const __restrict__ in,const int i0, const int i1, const int i2, const int i3, const int i4, const int i5, const int i6, const int i7){
	uint2 ret = __ldg((uint2*)&whirl_b0[__byte_perm(in[i0].x, 0, 0x4440)]);
	ret ^= sharedMemory[1][__byte_perm(in[i1].x, 0, 0x4441)];
	ret ^= sharedMemory[2][__byte_perm(in[i2].x, 0, 0x4442)];
	ret ^= sharedMemory[3][__byte_perm(in[i3].x, 0, 0x4443)];
	ret ^= sharedMemory[4][__byte_perm(in[i4].y, 0, 0x4440)];
	ret ^= ROR24(__ldg((uint2*)&whirl_b0[__byte_perm(in[i5].y, 0, 0x4441)]));
	ret ^= ROR8(__ldg((uint2*)&whirl_b7[__byte_perm(in[i6].y, 0, 0x4442)]));
	ret ^= __ldg((uint2*)&whirl_b7[__byte_perm(in[i7].y, 0, 0x4443)]);
	return ret;
}

__device__ __forceinline__
static uint2 d_ROUND_ELT(const uint2 sharedMemory[7][256],const uint2 *const __restrict__ in,const int i0, const int i1, const int i2, const int i3, const int i4, const int i5, const int i6, const int i7){

	uint2 ret = __ldg((uint2*)&whirl_b0[__byte_perm(in[i0].x, 0, 0x4440)]);
	ret ^= sharedMemory[1][__byte_perm(in[i1].x, 0, 0x4441)];
	ret ^= sharedMemory[2][__byte_perm(in[i2].x, 0, 0x4442)];
	ret ^= sharedMemory[3][__byte_perm(in[i3].x, 0, 0x4443)];
	ret ^= sharedMemory[4][__byte_perm(in[i4].y, 0, 0x4440)];
	ret ^= sharedMemory[5][__byte_perm(in[i5].y, 0, 0x4441)];
	ret ^= ROR8(__ldg((uint2*)&whirl_b7[__byte_perm(in[i6].y, 0, 0x4442)]));
	ret ^= __ldg((uint2*)&whirl_b7[__byte_perm(in[i7].y, 0, 0x4443)]);
	return ret;
}

__device__ __forceinline__
static uint2 d_ROUND_ELT1_LDG(const uint2 sharedMemory[7][256],const uint2 *const __restrict__ in,const int i0, const int i1, const int i2, const int i3, const int i4, const int i5, const int i6, const int i7, const uint2 c0){

	uint2 ret = __ldg((uint2*)&whirl_b0[__byte_perm(in[i0].x, 0, 0x4440)]);
	ret ^= sharedMemory[1][__byte_perm(in[i1].x, 0, 0x4441)];
	ret ^= sharedMemory[2][__byte_perm(in[i2].x, 0, 0x4442)];
	ret ^= sharedMemory[3][__byte_perm(in[i3].x, 0, 0x4443)];
	ret ^= sharedMemory[4][__byte_perm(in[i4].y, 0, 0x4440)];
	ret ^= ROR24(__ldg((uint2*)&whirl_b0[__byte_perm(in[i5].y, 0, 0x4441)]));
	ret ^= ROR8(__ldg((uint2*)&whirl_b7[__byte_perm(in[i6].y, 0, 0x4442)]));
	ret ^= __ldg((uint2*)&whirl_b7[__byte_perm(in[i7].y, 0, 0x4443)]);
	ret ^= c0;
	return ret;
}

__device__ __forceinline__
static uint2 d_ROUND_ELT1(const uint2 sharedMemory[7][256],const uint2 *const __restrict__ in,const int i0, const int i1, const int i2, const int i3, const int i4, const int i5, const int i6, const int i7, const uint2 c0){
	uint2 ret = __ldg((uint2*)&whirl_b0[__byte_perm(in[i0].x, 0, 0x4440)]);
	ret ^= sharedMemory[1][__byte_perm(in[i1].x, 0, 0x4441)];
	ret ^= sharedMemory[2][__byte_perm(in[i2].x, 0, 0x4442)];
	ret ^= sharedMemory[3][__byte_perm(in[i3].x, 0, 0x4443)];
	ret ^= sharedMemory[4][__byte_perm(in[i4].y, 0, 0x4440)];
	ret ^= sharedMemory[5][__byte_perm(in[i5].y, 0, 0x4441)];
	ret ^= ROR8(__ldg((uint2*)&whirl_b7[__byte_perm(in[i6].y, 0, 0x4442)]));//sharedMemory[6][__byte_perm(in[i6].y, 0, 0x4442)]
	ret ^= __ldg((uint2*)&whirl_b7[__byte_perm(in[i7].y, 0, 0x4443)]);//sharedMemory[7][__byte_perm(in[i7].y, 0, 0x4443)]
	ret ^= c0;
	return ret;
}

//--------END OF WHIRLPOOL DEVICE MACROS-----------------------------------------------------------------------------

/* Cooperative fill of the 7 rotated tables into shared memory (threads
 * < 256); callers __syncthreads() before use. */
__device__ __forceinline__
void whirlpool512_load_shared(uint2 sharedMemory[7][256])
{
	if (threadIdx.x < 256) {
		const uint2 tmp = __ldg((uint2*)&whirl_b0[threadIdx.x]);
		sharedMemory[0][threadIdx.x] = tmp;
		sharedMemory[1][threadIdx.x] = ROL8(tmp);
		sharedMemory[2][threadIdx.x] = ROL16(tmp);
		sharedMemory[3][threadIdx.x] = ROL24(tmp);
		sharedMemory[4][threadIdx.x] = SWAPUINT2(tmp);
		sharedMemory[5][threadIdx.x] = ROR24(tmp);
		sharedMemory[6][threadIdx.x] = ROR16(tmp);
	}
}

/* Whirlpool-512 of a 64-byte input, in place (uint2 hash[8], d_hash word
 * order in and out) — body of x15_whirlpool_gpu_hash_64. */
__device__ __forceinline__
void whirlpool512_hash_64(const uint2 sharedMemory[7][256], uint2 *const hash)
{
	uint2 n[8], h[8];
	uint2 tmp[8] = {
		{0xC0EE0B30,0x672990AF},{0x28282828,0x28282828},{0x28282828,0x28282828},{0x28282828,0x28282828},
		{0x28282828,0x28282828},{0x28282828,0x28282828},{0x28282828,0x28282828},{0x28282828,0x28282828}
	};

	#pragma unroll 8
	for(int i=0;i<8;i++)
		n[i]=hash[i];

		tmp[ 0]^= d_ROUND_ELT(sharedMemory,n, 0, 7, 6, 5, 4, 3, 2, 1);
		tmp[ 1]^= d_ROUND_ELT_LDG(sharedMemory,n, 1, 0, 7, 6, 5, 4, 3, 2);
		tmp[ 2]^= d_ROUND_ELT(sharedMemory,n, 2, 1, 0, 7, 6, 5, 4, 3);
		tmp[ 3]^= d_ROUND_ELT_LDG(sharedMemory,n, 3, 2, 1, 0, 7, 6, 5, 4);
		tmp[ 4]^= d_ROUND_ELT(sharedMemory,n, 4, 3, 2, 1, 0, 7, 6, 5);
		tmp[ 5]^= d_ROUND_ELT_LDG(sharedMemory,n, 5, 4, 3, 2, 1, 0, 7, 6);
		tmp[ 6]^= d_ROUND_ELT(sharedMemory,n, 6, 5, 4, 3, 2, 1, 0, 7);
		tmp[ 7]^= d_ROUND_ELT_LDG(sharedMemory,n, 7, 6, 5, 4, 3, 2, 1, 0);
		for (int i=1; i <10; i++){
			TRANSFER(n, tmp);
			tmp[ 0] = d_ROUND_ELT1_LDG(sharedMemory,n, 0, 7, 6, 5, 4, 3, 2, 1, whirl_precomputed_round_key_64[(i-1)*8+0]);
			tmp[ 1] = d_ROUND_ELT1(    sharedMemory,n, 1, 0, 7, 6, 5, 4, 3, 2, whirl_precomputed_round_key_64[(i-1)*8+1]);
			tmp[ 2] = d_ROUND_ELT1(    sharedMemory,n, 2, 1, 0, 7, 6, 5, 4, 3, whirl_precomputed_round_key_64[(i-1)*8+2]);
			tmp[ 3] = d_ROUND_ELT1_LDG(sharedMemory,n, 3, 2, 1, 0, 7, 6, 5, 4, whirl_precomputed_round_key_64[(i-1)*8+3]);
			tmp[ 4] = d_ROUND_ELT1(    sharedMemory,n, 4, 3, 2, 1, 0, 7, 6, 5, whirl_precomputed_round_key_64[(i-1)*8+4]);
			tmp[ 5] = d_ROUND_ELT1(    sharedMemory,n, 5, 4, 3, 2, 1, 0, 7, 6, whirl_precomputed_round_key_64[(i-1)*8+5]);
			tmp[ 6] = d_ROUND_ELT1(    sharedMemory,n, 6, 5, 4, 3, 2, 1, 0, 7, whirl_precomputed_round_key_64[(i-1)*8+6]);
			tmp[ 7] = d_ROUND_ELT1_LDG(sharedMemory,n, 7, 6, 5, 4, 3, 2, 1, 0, whirl_precomputed_round_key_64[(i-1)*8+7]);
		}

		TRANSFER(h, tmp);
		#pragma unroll 8
		for (int i=0; i<8; i++)
			hash[ i] = h[i] = h[i] ^ hash[i];

		#pragma unroll 6
		for (int i=1; i<7; i++)
			n[i]=vectorize(0);

		n[0] = vectorize(0x80);
		n[7] = vectorize(0x2000000000000);

		#pragma unroll 8
		for (int i=0; i < 8; i++) {
			n[i] = n[i] ^ h[i];
		}

//		#pragma unroll 10
		for (int i=0; i < 10; i++) {
			tmp[ 0] = whirl_InitVector_RC[i];
			tmp[ 0]^= d_ROUND_ELT(sharedMemory, h, 0, 7, 6, 5, 4, 3, 2, 1);
			tmp[ 1] = d_ROUND_ELT(sharedMemory, h, 1, 0, 7, 6, 5, 4, 3, 2);
			tmp[ 2] = d_ROUND_ELT_LDG(sharedMemory, h, 2, 1, 0, 7, 6, 5, 4, 3);
			tmp[ 3] = d_ROUND_ELT(sharedMemory, h, 3, 2, 1, 0, 7, 6, 5, 4);
			tmp[ 4] = d_ROUND_ELT_LDG(sharedMemory, h, 4, 3, 2, 1, 0, 7, 6, 5);
			tmp[ 5] = d_ROUND_ELT(sharedMemory, h, 5, 4, 3, 2, 1, 0, 7, 6);
			tmp[ 6] = d_ROUND_ELT(sharedMemory, h, 6, 5, 4, 3, 2, 1, 0, 7);
			tmp[ 7] = d_ROUND_ELT(sharedMemory, h, 7, 6, 5, 4, 3, 2, 1, 0);
			TRANSFER(h, tmp);
			tmp[ 0] = d_ROUND_ELT1(sharedMemory,n, 0, 7, 6, 5, 4, 3, 2, 1, tmp[0]);
			tmp[ 1] = d_ROUND_ELT1_LDG(sharedMemory,n, 1, 0, 7, 6, 5, 4, 3, 2, tmp[1]);
			tmp[ 2] = d_ROUND_ELT1(sharedMemory,n, 2, 1, 0, 7, 6, 5, 4, 3, tmp[2]);
			tmp[ 3] = d_ROUND_ELT1(sharedMemory,n, 3, 2, 1, 0, 7, 6, 5, 4, tmp[3]);
			tmp[ 4] = d_ROUND_ELT1_LDG(sharedMemory,n, 4, 3, 2, 1, 0, 7, 6, 5, tmp[4]);
			tmp[ 5] = d_ROUND_ELT1(sharedMemory,n, 5, 4, 3, 2, 1, 0, 7, 6, tmp[5]);
			tmp[ 6] = d_ROUND_ELT1_LDG(sharedMemory,n, 6, 5, 4, 3, 2, 1, 0, 7, tmp[6]);
			tmp[ 7] = d_ROUND_ELT1(sharedMemory,n, 7, 6, 5, 4, 3, 2, 1, 0, tmp[7]);
			TRANSFER(n, tmp);
		}

	hash[0] = whirl_xor3x(hash[0], n[0], vectorize(0x80));
	hash[1] = hash[1]^ n[1];
	hash[2] = hash[2]^ n[2];
	hash[3] = hash[3]^ n[3];
	hash[4] = hash[4]^ n[4];
	hash[5] = hash[5]^ n[5];
	hash[6] = hash[6]^ n[6];
	hash[7] = whirl_xor3x(hash[7], n[7], vectorize(0x2000000000000));
}

/* Per-TU host-side table upload (mode 0 = plain Whirlpool for x15/x16,
 * mode 1 = legacy whirlpool1) — from x15_whirlpool_cpu_init. */
static void whirlpool512_init_tables(int mode)
{
	uint64_t* table0 = NULL;

	switch (mode) {
	case 0: /* x15 with rotated T1-T7 (based on T0) */
		table0 = (uint64_t*)plain_T0;
		cudaMemcpyToSymbol(whirl_InitVector_RC, plain_RC, 10*sizeof(uint64_t),0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(whirl_precomputed_round_key_64, plain_precomputed_round_key_64, 72*sizeof(uint64_t),0, cudaMemcpyHostToDevice);
		break;
	case 1: /* old whirlpool */
		table0 = (uint64_t*)old1_T0;
		cudaMemcpyToSymbol(whirl_InitVector_RC, old1_RC, 10*sizeof(uint64_t),0,cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(whirl_precomputed_round_key_64, old1_precomputed_round_key_64, 72*sizeof(uint64_t),0, cudaMemcpyHostToDevice);
		break;
	default:
		applog(LOG_ERR,"Bad whirlpool mode");
		exit(0);
	}
	cudaMemcpyToSymbol(whirl_b0, table0, 256*sizeof(uint64_t),0, cudaMemcpyHostToDevice);
	uint64_t table7[256];
	for(int i=0;i<256;i++){
		table7[i] = ROTR64(table0[i],8);
	}
	cudaMemcpyToSymbol(whirl_b7, table7, 256*sizeof(uint64_t),0, cudaMemcpyHostToDevice);
}

#endif /* CUDA_WHIRLPOOL512_DEVICE_CUH */
