/*
 * Streebog GOST R 34.10-2012 stripped CUDA implementation for final hash
 *
 * https://tools.ietf.org/html/rfc6986
 * https://en.wikipedia.org/wiki/Streebog
 *
 * ==========================(LICENSE BEGIN)============================
 *
 * @author   Tanguy Pruvot - 2017
 * @author   Alexis Provos - 2016
 */

// Further improved with shared memory partial utilization
// Tested under CUDA7.5 toolkit for cp 5.0/5.2

#include <miner.h>
#include <cuda_helper.h>
#include <cuda_vectors.h>
#include <cuda_vector_uint2x4.h>

#include "x11/streebog_arrays.cuh"

//#define FULL_UNROLL
__device__ __forceinline__
static void GOST_FS(const uint2 shared[8][256],const uint2 *const __restrict__ state,uint2* return_state)
{
	return_state[0] = __ldg(&T02[__byte_perm(state[7].x,0,0x44440)])
			^ shared[1][__byte_perm(state[6].x,0,0x44440)]
			^ shared[2][__byte_perm(state[5].x,0,0x44440)]
			^ shared[3][__byte_perm(state[4].x,0,0x44440)]
			^ shared[4][__byte_perm(state[3].x,0,0x44440)]
			^ shared[5][__byte_perm(state[2].x,0,0x44440)]
			^ shared[6][__byte_perm(state[1].x,0,0x44440)]
			^ __ldg(&T72[__byte_perm(state[0].x,0,0x44440)]);

	return_state[1] =  __ldg(&T02[__byte_perm(state[7].x,0,0x44441)])
			^ __ldg(&T12[__byte_perm(state[6].x,0,0x44441)])
			^ shared[2][__byte_perm(state[5].x,0,0x44441)]
			^ shared[3][__byte_perm(state[4].x,0,0x44441)]
			^ shared[4][__byte_perm(state[3].x,0,0x44441)]
			^ shared[5][__byte_perm(state[2].x,0,0x44441)]
			^ shared[6][__byte_perm(state[1].x,0,0x44441)]
			^ __ldg(&T72[__byte_perm(state[0].x,0,0x44441)]);

	return_state[2] = __ldg(&T02[__byte_perm(state[7].x,0,0x44442)])
			^ __ldg(&T12[__byte_perm(state[6].x,0,0x44442)])
			^ shared[2][__byte_perm(state[5].x,0,0x44442)]
			^ shared[3][__byte_perm(state[4].x,0,0x44442)]
			^ shared[4][__byte_perm(state[3].x,0,0x44442)]
			^ shared[5][__byte_perm(state[2].x,0,0x44442)]
			^ __ldg(&T72[__byte_perm(state[0].x,0,0x44442)])
			^ shared[6][__byte_perm(state[1].x,0,0x44442)];

	return_state[3] = __ldg(&T02[__byte_perm(state[7].x,0,0x44443)])
			^ shared[1][__byte_perm(state[6].x,0,0x44443)]
			^ shared[2][__byte_perm(state[5].x,0,0x44443)]
			^ shared[3][__byte_perm(state[4].x,0,0x44443)]
			^ __ldg(&T42[__byte_perm(state[3].x,0,0x44443)])
			^ shared[5][__byte_perm(state[2].x,0,0x44443)]
			^ __ldg(&T72[__byte_perm(state[0].x,0,0x44443)])
			^ shared[6][__byte_perm(state[1].x,0,0x44443)];

	return_state[4] = __ldg(&T02[__byte_perm(state[7].y,0,0x44440)])
			^ shared[1][__byte_perm(state[6].y,0,0x44440)]
			^ __ldg(&T22[__byte_perm(state[5].y,0,0x44440)])
			^ shared[3][__byte_perm(state[4].y,0,0x44440)]
			^ shared[4][__byte_perm(state[3].y,0,0x44440)]
			^ __ldg(&T62[__byte_perm(state[1].y,0,0x44440)])
			^ shared[5][__byte_perm(state[2].y,0,0x44440)]
			^ __ldg(&T72[__byte_perm(state[0].y,0,0x44440)]);

	return_state[5] = __ldg(&T02[__byte_perm(state[7].y,0,0x44441)])
			^ shared[2][__byte_perm(state[5].y,0,0x44441)]
			^ __ldg(&T12[__byte_perm(state[6].y,0,0x44441)])
			^ shared[3][__byte_perm(state[4].y,0,0x44441)]
			^ shared[4][__byte_perm(state[3].y,0,0x44441)]
			^ shared[5][__byte_perm(state[2].y,0,0x44441)]
			^ __ldg(&T62[__byte_perm(state[1].y,0,0x44441)])
			^ __ldg(&T72[__byte_perm(state[0].y,0,0x44441)]);

	return_state[6] = __ldg(&T02[__byte_perm(state[7].y,0,0x44442)])
			^ shared[1][__byte_perm(state[6].y,0,0x44442)]
			^ shared[2][__byte_perm(state[5].y,0,0x44442)]
			^ shared[3][__byte_perm(state[4].y,0,0x44442)]
			^ shared[4][__byte_perm(state[3].y,0,0x44442)]
			^ shared[5][__byte_perm(state[2].y,0,0x44442)]
			^ __ldg(&T62[__byte_perm(state[1].y,0,0x44442)])
			^ __ldg(&T72[__byte_perm(state[0].y,0,0x44442)]);

	return_state[7] = __ldg(&T02[__byte_perm(state[7].y,0,0x44443)])
			^ __ldg(&T12[__byte_perm(state[6].y,0,0x44443)])
			^ shared[2][__byte_perm(state[5].y,0,0x44443)]
			^ shared[3][__byte_perm(state[4].y,0,0x44443)]
			^ shared[4][__byte_perm(state[3].y,0,0x44443)]
			^ shared[5][__byte_perm(state[2].y,0,0x44443)]
			^ __ldg(&T62[__byte_perm(state[1].y,0,0x44443)])
			^ __ldg(&T72[__byte_perm(state[0].y,0,0x44443)]);
}

__device__ __forceinline__
static void GOST_FS_LDG(const uint2 shared[8][256],const uint2 *const __restrict__ state,uint2* return_state)
{
	return_state[0] =  __ldg(&T02[__byte_perm(state[7].x,0,0x44440)])
			^ __ldg(&T12[__byte_perm(state[6].x,0,0x44440)])
			^ shared[2][__byte_perm(state[5].x,0,0x44440)]
			^ shared[3][__byte_perm(state[4].x,0,0x44440)]
			^ shared[4][__byte_perm(state[3].x,0,0x44440)]
			^ shared[5][__byte_perm(state[2].x,0,0x44440)]
			^ shared[6][__byte_perm(state[1].x,0,0x44440)]
			^ __ldg(&T72[__byte_perm(state[0].x,0,0x44440)]);

	return_state[1] =  __ldg(&T02[__byte_perm(state[7].x,0,0x44441)])
			^ __ldg(&T12[__byte_perm(state[6].x,0,0x44441)])
			^ shared[2][__byte_perm(state[5].x,0,0x44441)]
			^ shared[3][__byte_perm(state[4].x,0,0x44441)]
			^ shared[4][__byte_perm(state[3].x,0,0x44441)]
			^ shared[5][__byte_perm(state[2].x,0,0x44441)]
			^ __ldg(&T72[__byte_perm(state[0].x,0,0x44441)])
			^ shared[6][__byte_perm(state[1].x,0,0x44441)];

	return_state[2] =  __ldg(&T02[__byte_perm(state[7].x,0,0x44442)])
			^ __ldg(&T12[__byte_perm(state[6].x,0,0x44442)])
			^ shared[2][__byte_perm(state[5].x,0,0x44442)]
			^ shared[3][__byte_perm(state[4].x,0,0x44442)]
			^ shared[4][__byte_perm(state[3].x,0,0x44442)]
			^ shared[5][__byte_perm(state[2].x,0,0x44442)]
			^ shared[6][__byte_perm(state[1].x,0,0x44442)]
			^ __ldg(&T72[__byte_perm(state[0].x,0,0x44442)]);

	return_state[3] = __ldg(&T02[__byte_perm(state[7].x,0,0x44443)])
			^ __ldg(&T12[__byte_perm(state[6].x,0,0x44443)])
			^ shared[2][__byte_perm(state[5].x,0,0x44443)]
			^ shared[3][__byte_perm(state[4].x,0,0x44443)]
			^ shared[4][__byte_perm(state[3].x,0,0x44443)]
			^ shared[5][__byte_perm(state[2].x,0,0x44443)]
			^ shared[6][__byte_perm(state[1].x,0,0x44443)]
			^ __ldg(&T72[__byte_perm(state[0].x,0,0x44443)]);

	return_state[4] = __ldg(&T02[__byte_perm(state[7].y,0,0x44440)])
			^ shared[1][__byte_perm(state[6].y,0,0x44440)]
			^ __ldg(&T22[__byte_perm(state[5].y,0,0x44440)])
			^ shared[3][__byte_perm(state[4].y,0,0x44440)]
			^ shared[4][__byte_perm(state[3].y,0,0x44440)]
			^ shared[5][__byte_perm(state[2].y,0,0x44440)]
			^ __ldg(&T72[__byte_perm(state[0].y,0,0x44440)])
			^ __ldg(&T62[__byte_perm(state[1].y,0,0x44440)]);

	return_state[5] =  __ldg(&T02[__byte_perm(state[7].y,0,0x44441)])
			^ __ldg(&T12[__byte_perm(state[6].y,0,0x44441)])
			^ shared[2][__byte_perm(state[5].y,0,0x44441)]
			^ shared[3][__byte_perm(state[4].y,0,0x44441)]
			^ shared[4][__byte_perm(state[3].y,0,0x44441)]
			^ shared[5][__byte_perm(state[2].y,0,0x44441)]
			^ __ldg(&T72[__byte_perm(state[0].y,0,0x44441)])
			^ __ldg(&T62[__byte_perm(state[1].y,0,0x44441)]);

	return_state[6] = __ldg(&T02[__byte_perm(state[7].y,0,0x44442)])
			^ __ldg(&T12[__byte_perm(state[6].y,0,0x44442)])
			^ __ldg(&T22[__byte_perm(state[5].y,0,0x44442)])
			^ shared[3][__byte_perm(state[4].y,0,0x44442)]
			^ shared[4][__byte_perm(state[3].y,0,0x44442)]
			^ shared[5][__byte_perm(state[2].y,0,0x44442)]
			^ __ldg(&T72[__byte_perm(state[0].y,0,0x44442)])
			^ __ldg(&T62[__byte_perm(state[1].y,0,0x44442)]);

	return_state[7] = __ldg(&T02[__byte_perm(state[7].y,0,0x44443)])
			^ shared[1][__byte_perm(state[6].y,0,0x44443)]
			^ __ldg(&T22[__byte_perm(state[5].y,0,0x44443)])
			^ shared[3][__byte_perm(state[4].y,0,0x44443)]
			^ shared[4][__byte_perm(state[3].y,0,0x44443)]
			^ shared[5][__byte_perm(state[2].y,0,0x44443)]
			^ __ldg(&T72[__byte_perm(state[0].y,0,0x44443)])
			^ __ldg(&T62[__byte_perm(state[1].y,0,0x44443)]);
}

__device__ __forceinline__
static void GOST_E12(const uint2 shared[8][256],uint2 *const __restrict__ K, uint2 *const __restrict__ state)
{
	uint2 t[ 8];
	//#pragma unroll 12
	for(int i=0; i<12; i++){
		GOST_FS(shared,state, t);

		#pragma unroll 8
		for(int j=0;j<8;j++)
			K[ j] ^= *(uint2*)&CC[i][j];

		#pragma unroll 8
		for(int j=0;j<8;j++)
			state[ j] = t[ j];

		GOST_FS_LDG(shared,K, t);

		#pragma unroll 8
		for(int j=0;j<8;j++)
			state[ j]^= t[ j];

		#pragma unroll 8
		for(int j=0;j<8;j++)
			K[ j] = t[ j];
	}
}


#define TPB 256
__global__
#if __CUDA_ARCH__ > 500
__launch_bounds__(TPB, 2)
#else
__launch_bounds__(TPB, 3)
#endif
void streebog_gpu_hash_64_alexis(uint64_t *g_hash){

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	uint2 buf[8], t[8], temp[8], K0[8], hash[8];

	__shared__ uint2 shared[7][256];
	shared[0][threadIdx.x] = __ldg(&T02[threadIdx.x]);
	shared[1][threadIdx.x] = __ldg(&T12[threadIdx.x]);
	shared[2][threadIdx.x] = __ldg(&T22[threadIdx.x]);
	shared[3][threadIdx.x] = __ldg(&T32[threadIdx.x]);
	shared[4][threadIdx.x] = __ldg(&T42[threadIdx.x]);
	shared[5][threadIdx.x] = __ldg(&T52[threadIdx.x]);
	shared[6][threadIdx.x] = __ldg(&T62[threadIdx.x]);
	//shared[7][threadIdx.x] = __ldg(&T72[threadIdx.x]);

//	if (thread < threads)
//	{
	uint64_t* inout = &g_hash[thread<<3];

	*(uint2x4*)&hash[0] = __ldg4((uint2x4*)&inout[0]);
	*(uint2x4*)&hash[4] = __ldg4((uint2x4*)&inout[4]);

	__syncthreads();

	#pragma unroll
	for(int i = 0; i < 8; i++) buf[i] = vectorize(0x74a5d4ce2efc83b3) ^ hash[i];

	#pragma nounroll
	for(int i = 0; i < 12; i++) {
		GOST_FS(shared, buf, temp);
		#pragma unroll
		for(uint32_t j = 0; j < 8; j++) buf[j] = temp[j] ^ *(uint2*)&precomputed_values[i][j];
	}

	#pragma unroll
	for(int j = 0; j < 8; j++) buf[j] ^= hash[j];

	#pragma unroll
	for(int j = 0; j < 8; j++) K0[j] = buf[j];
	K0[7].y ^= 0x00020000;

	GOST_FS(shared, K0, t);

	#pragma unroll
	for(int i = 0; i < 8; i++) K0[i] = t[i];

	t[7].y ^= 0x01000000;

	GOST_E12(shared, K0, t);

	#pragma unroll
	for(int j = 0; j < 8; j++) buf[j] ^= t[j];

	buf[7].y ^= 0x01000000;

	GOST_FS(shared, buf,K0);

	buf[7].y ^= 0x00020000;

	#pragma unroll
	for(int j = 0; j < 8; j++) t[j] = K0[j];

	t[7].y ^= 0x00020000;

	GOST_E12(shared, K0, t);

	#pragma unroll
	for(int j = 0; j < 8; j++) buf[j] ^= t[j];

	GOST_FS(shared, buf,K0); // K = F(h)

	hash[7]+= vectorize(0x0100000000000000);

	#pragma unroll
	for(int j = 0; j < 8; j++) t[j] = K0[j] ^ hash[j];

	GOST_E12(shared, K0, t);

	*(uint2x4*)&inout[ 0] = *(uint2x4*)&t[ 0] ^ *(uint2x4*)&hash[0] ^ *(uint2x4*)&buf[0];
	*(uint2x4*)&inout[ 4] = *(uint2x4*)&t[ 4] ^ *(uint2x4*)&hash[4] ^ *(uint2x4*)&buf[4];
}

__host__
void streebog_cpu_hash_64_alexis(int thr_id, uint32_t threads, uint32_t *d_hash)
{
	dim3 grid((threads + TPB-1) / TPB);
	dim3 block(TPB);

	streebog_gpu_hash_64_alexis<<<grid, block>>>((uint64_t*)d_hash);
}

__constant__ uint64_t target64[4];

__host__
void skunk_streebog_set_target(uint32_t* ptarget)
{
	cudaMemcpyToSymbol(target64, ptarget, 4*sizeof(uint64_t), 0, cudaMemcpyHostToDevice);
}

#define TPB 256
__global__
__launch_bounds__(TPB, 2)
void skunk_streebog_gpu_final_64(uint64_t *g_hash, uint32_t* resNonce)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	uint2 buf[8], t[8], temp[8], K0[8], hash[8];

	__shared__ uint2 shared[8][256];
	shared[0][threadIdx.x] = __ldg(&T02[threadIdx.x]);
	shared[1][threadIdx.x] = __ldg(&T12[threadIdx.x]);
	shared[2][threadIdx.x] = __ldg(&T22[threadIdx.x]);
	shared[3][threadIdx.x] = __ldg(&T32[threadIdx.x]);
	shared[4][threadIdx.x] = __ldg(&T42[threadIdx.x]);
	shared[5][threadIdx.x] = __ldg(&T52[threadIdx.x]);
	shared[6][threadIdx.x] = __ldg(&T62[threadIdx.x]);
	shared[7][threadIdx.x] = __ldg(&T72[threadIdx.x]);

//	if (thread < threads)
//	{
	uint64_t* inout = &g_hash[thread<<3];
	*(uint2x4*)&hash[0] = __ldg4((uint2x4*)&inout[0]);
	*(uint2x4*)&hash[4] = __ldg4((uint2x4*)&inout[4]);

	__threadfence_block();

	K0[0] = vectorize(0x74a5d4ce2efc83b3);

	#pragma unroll 8
	for(uint32_t i=0;i<8;i++){
		buf[ i] = hash[ i] ^ K0[ 0];
	}
	//#pragma unroll 12
	for(int i=0; i<12; i++){
		GOST_FS(shared, buf, temp);
		#pragma unroll 8
		for(uint32_t j=0;j<8;j++){
			buf[ j] = temp[ j] ^ *(uint2*)&precomputed_values[i][j];
		}
	}
	#pragma unroll 8
	for(int j=0;j<8;j++){
		buf[ j]^= hash[ j];
	}
	#pragma unroll 8
	for(int j=0;j<8;j++){
		K0[ j] = buf[ j];
	}

	K0[7].y ^= 0x00020000;

	GOST_FS(shared, K0, t);

	#pragma unroll 8
	for(uint32_t i=0;i<8;i++)
		K0[ i] = t[ i];

	t[7].y ^= 0x01000000;
	GOST_E12(shared, K0, t);

	#pragma unroll 8
	for(int j=0;j<8;j++)
		buf[ j] ^= t[ j];

	buf[7].y ^= 0x01000000;

	GOST_FS(shared, buf,K0);

	buf[7].y ^= 0x00020000;

	#pragma unroll 8
	for(uint32_t j=0;j<8;j++)
		t[ j] = K0[ j];

	t[7].y ^= 0x00020000;
	GOST_E12(shared, K0, t);

	#pragma unroll 8
	for(uint32_t j=0;j<8;j++)
		buf[ j] ^= t[ j];

	GOST_FS(shared, buf,K0); // K = F(h)

	hash[7]+= vectorize(0x0100000000000000);

	#pragma unroll 8
	for(uint32_t j=0;j<8;j++)
		t[ j] = K0[ j] ^ hash[ j];

//	#pragma unroll
	for(uint32_t i=0; i<10; i++){
		GOST_FS(shared, t, temp);

		#pragma unroll 8
		for(uint32_t j=0;j<8;j++){
			t[ j] = temp[ j];
			K0[ j] = K0[ j] ^ *(uint2*)&CC[ i][ j];
		}

		GOST_FS(shared, K0, temp);

		#pragma unroll 8
		for(uint32_t j=0;j<8;j++){
			K0[ j] = temp[ j];
			t[ j]^= temp[ j];
		}
	}

	GOST_FS(shared, t, temp);

	#pragma unroll 8
	for(uint32_t j=0;j<8;j++){
		t[ j] = temp[ j];
		K0[ j] = K0[ j] ^ *(uint2*)&CC[10][ j];
	}

	GOST_FS(shared, K0, temp);

	#pragma unroll 8
	for(int i=7;i>=0;i--){
		t[i].x = t[i].x ^ temp[i].x;
		temp[i].x = temp[i].x ^ ((uint32_t*)&CC[11])[i<<1];
	}

	uint2 last[2];

#define T0(x) shared[0][x]
#define T1(x) shared[1][x]
#define T2(x) shared[2][x]
#define T3(x) shared[3][x]
#define T4(x) shared[4][x]
#define T5(x) shared[5][x]
#define T6(x) shared[6][x]
#define T7(x) shared[7][x]

	last[ 0] = T0(__byte_perm(t[7].x,0,0x44443)) ^ T1(__byte_perm(t[6].x,0,0x44443))
		 ^ T2(__byte_perm(t[5].x,0,0x44443)) ^ T3(__byte_perm(t[4].x,0,0x44443))
		 ^ T4(__byte_perm(t[3].x,0,0x44443)) ^ T5(__byte_perm(t[2].x,0,0x44443))
		 ^ T6(__byte_perm(t[1].x,0,0x44443)) ^ T7(__byte_perm(t[0].x,0,0x44443));

	last[ 1] = T0(__byte_perm(temp[7].x,0,0x44443)) ^ T1(__byte_perm(temp[6].x,0,0x44443))
		 ^ T2(__byte_perm(temp[5].x,0,0x44443)) ^ T3(__byte_perm(temp[4].x,0,0x44443))
		 ^ T4(__byte_perm(temp[3].x,0,0x44443)) ^ T5(__byte_perm(temp[2].x,0,0x44443))
		 ^ T6(__byte_perm(temp[1].x,0,0x44443)) ^ T7(__byte_perm(temp[0].x,0,0x44443));

	if(devectorize(buf[3] ^ hash[3] ^ last[ 0] ^ last[ 1]) <= target64[3]){
		uint32_t tmp = atomicExch(&resNonce[0], thread);
		if (tmp != UINT32_MAX)
			resNonce[1] = tmp;
	}
}

__host__
void skunk_cuda_streebog(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t* d_resNonce)
{
	dim3 grid((threads + TPB-1) / TPB);
	dim3 block(TPB);

	skunk_streebog_gpu_final_64 <<< grid, block >>> ((uint64_t*)d_hash, d_resNonce);
}