#include <stdio.h>
#include <stdint.h>
#include <memory.h>

#define TPB52 32

#include "cuda_lyra2_vectors.h"
#include "cuda/blake2b_device.cuh"

#define Nrow 4
#define Ncol 4
#define memshift 3

__device__ uint2x4 *DState;

__device__ __forceinline__ uint2 LD4S(const int index)
{
	extern __shared__ uint2 shared_mem[];

	return shared_mem[(index * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x];
}

__device__ __forceinline__ void ST4S(const int index, const uint2 data)
{
	extern __shared__ uint2 shared_mem[];

	shared_mem[(index * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x] = data;
}



__device__ __forceinline__ void reduceDuplexRowSetupV2(uint2 state[4])
{
	int i, j;
	uint2 state1[Ncol][3], state0[Ncol][3], state2[3];

#pragma unroll
	for (int i = 0; i < Ncol; i++)
	{
#pragma unroll
		for (j = 0; j < 3; j++)
			state0[Ncol - i - 1][j] = state[j];
		round_lyra(state);
	}

	//#pragma unroll 4
	for (i = 0; i < Ncol; i++)
	{
#pragma unroll
		for (j = 0; j < 3; j++)
			state[j] ^= state0[i][j];

		round_lyra(state);

#pragma unroll
		for (j = 0; j < 3; j++)
			state1[Ncol - i - 1][j] = state0[i][j];

#pragma unroll
		for (j = 0; j < 3; j++)
			state1[Ncol - i - 1][j] ^= state[j];
	}

	uint32_t s0 = 0;
	uint32_t s2 = 33;
	for (i = 0; i < Ncol; i++)
	{
#pragma unroll
		for (j = 0; j < 3; j++)
			state[j] ^= state1[i][j] + state0[i][j];

		round_lyra(state);

#pragma unroll
		for (j = 0; j < 3; j++)
			state2[j] = state1[i][j];

#pragma unroll
		for (j = 0; j < 3; j++)
			state2[j] ^= state[j];

#pragma unroll
		for (j = 0; j < 3; j++)
			ST4S(s2 + j, state2[j]);

		//���O�̃X���b�h����f�[�^��Ⴄ(�����Ɉ��̃X���b�h�Ƀf�[�^�𑗂�)
		uint2 Data0 = state[0];
		uint2 Data1 = state[1];
		uint2 Data2 = state[2];
		WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

		if (threadIdx.x == 0)
		{
			state0[i][0] ^= Data2;
			state0[i][1] ^= Data0;
			state0[i][2] ^= Data1;
		}
		else
		{
			state0[i][0] ^= Data0;
			state0[i][1] ^= Data1;
			state0[i][2] ^= Data2;
		}

#pragma unroll
		for (j = 0; j < 3; j++)
			ST4S(s0 + j, state0[i][j]);

#pragma unroll
		for (j = 0; j < 3; j++)
			state0[i][j] = state2[j];

		s0 += memshift;
		s2 -= memshift;
	}

	s2 += 24;
	for (i = 0; i < Ncol; i++)
	{
#pragma unroll
		for (j = 0; j < 3; j++)
			state[j] ^= state1[i][j] + state0[Ncol - i - 1][j];

		round_lyra(state);

#pragma unroll
		for (j = 0; j < 3; j++)
			state0[Ncol - i - 1][j] ^= state[j];
#pragma unroll
		for (j = 0; j < 3; j++)
			ST4S(s2 + j, state0[Ncol - i - 1][j]);

		//���O�̃X���b�h����f�[�^��Ⴄ(�����Ɉ��̃X���b�h�Ƀf�[�^�𑗂�)
		uint2 Data0 = state[0];
		uint2 Data1 = state[1];
		uint2 Data2 = state[2];
		WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

		if (threadIdx.x == 0)
		{
			state1[i][0] ^= Data2;
			state1[i][1] ^= Data0;
			state1[i][2] ^= Data1;
		}
		else
		{
			state1[i][0] ^= Data0;
			state1[i][1] ^= Data1;
			state1[i][2] ^= Data2;
		}

#pragma unroll
		for (j = 0; j < 3; j++)
			ST4S(s0 + j, state1[i][j]);

		s0 += memshift;
		s2 -= memshift;
	}
}

__device__ void reduceDuplexRowtV2(uint2 state[4])
{
	uint32_t rowInOut = WarpShuffle(state[0].x, 0, 4) & 3;

	uint2 state2[3], state1[3], last[3];
	uint32_t s1 = 36;
	uint32_t s2 = 12 * rowInOut;
	uint32_t s3 = 0;

	for (int i = 0; i < Ncol; i++)
	{
#pragma unroll
		for (int j = 0; j < 3; j++)
			state2[j] = LD4S(s2 + j);

#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= LD4S(s1 + j) + state2[j];

		round_lyra(state);

		//���O�̃X���b�h����f�[�^��Ⴄ(�����Ɉ��̃X���b�h�Ƀf�[�^�𑗂�)
		uint2 Data0 = state[0];
		uint2 Data1 = state[1];
		uint2 Data2 = state[2];
		WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

		if (threadIdx.x == 0)
		{
			state2[0] ^= Data2;
			state2[1] ^= Data0;
			state2[2] ^= Data1;
		}
		else
		{
			state2[0] ^= Data0;
			state2[1] ^= Data1;
			state2[2] ^= Data2;
		}

#pragma unroll
		for (int j = 0; j < 3; j++)
		{
			ST4S(s2 + j, state2[j]);
			ST4S(s3 + j, LD4S(s3 + j) ^ state[j]);
		}

		s1 += memshift;
		s2 += memshift;
		s3 += memshift;
	}
	s1 = 0;
	rowInOut = WarpShuffle(state[0].x, 0, 4) & 3;
	s2 = 12 * rowInOut;

	for (int i = 0; i < Ncol; i++)
	{
#pragma unroll
		for (int j = 0; j < 3; j++)
			state2[j] = LD4S(s2 + j);

#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= LD4S(s1 + j) + state2[j];

		round_lyra(state);

		//���O�̃X���b�h����f�[�^��Ⴄ(�����Ɉ��̃X���b�h�Ƀf�[�^�𑗂�)
		uint2 Data0 = state[0];
		uint2 Data1 = state[1];
		uint2 Data2 = state[2];
		WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

		if (threadIdx.x == 0)
		{
			state2[0] ^= Data2;
			state2[1] ^= Data0;
			state2[2] ^= Data1;
		}
		else
		{
			state2[0] ^= Data0;
			state2[1] ^= Data1;
			state2[2] ^= Data2;
		}

#pragma unroll
		for (int j = 0; j < 3; j++)
		{
			ST4S(s2 + j, state2[j]);
			ST4S(s3 + j, LD4S(s3 + j) ^ state[j]);
		}

		s1 += memshift;
		s2 += memshift;
		s3 += memshift;
	}

	rowInOut = WarpShuffle(state[0].x, 0, 4) & 3;
	s2 = 12 * rowInOut;

	for (int i = 0; i < Ncol; i++)
	{
#pragma unroll
		for (int j = 0; j < 3; j++)
			state2[j] = LD4S(s2 + j);

#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= LD4S(s1 + j) + state2[j];

		round_lyra(state);

		//���O�̃X���b�h����f�[�^��Ⴄ(�����Ɉ��̃X���b�h�Ƀf�[�^�𑗂�)
		uint2 Data0 = state[0];
		uint2 Data1 = state[1];
		uint2 Data2 = state[2];
		WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

		if (threadIdx.x == 0)
		{
			state2[0] ^= Data2;
			state2[1] ^= Data0;
			state2[2] ^= Data1;
		}
		else
		{
			state2[0] ^= Data0;
			state2[1] ^= Data1;
			state2[2] ^= Data2;
		}

#pragma unroll
		for (int j = 0; j < 3; j++)
		{
			ST4S(s2 + j, state2[j]);
			ST4S(s3 + j, LD4S(s3 + j) ^ state[j]);
		}

		s1 += memshift;
		s2 += memshift;
		s3 += memshift;
	}

	rowInOut = WarpShuffle(state[0].x, 0, 4) & 3;
	s2 = 12 * rowInOut;

#pragma unroll
	for (int j = 0; j < 3; j++)
		last[j] = LD4S(s2 + j);

#pragma unroll
	for (int j = 0; j < 3; j++)
		state[j] ^= LD4S(s1 + j) + last[j];

	round_lyra(state);

	//���O�̃X���b�h����f�[�^��Ⴄ(�����Ɉ��̃X���b�h�Ƀf�[�^�𑗂�)
	uint2 Data0 = state[0];
	uint2 Data1 = state[1];
	uint2 Data2 = state[2];
	WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

	if (threadIdx.x == 0)
	{
		last[0] ^= Data2;
		last[1] ^= Data0;
		last[2] ^= Data1;
	}
	else
	{
		last[0] ^= Data0;
		last[1] ^= Data1;
		last[2] ^= Data2;
	}

	if (rowInOut == 3)
	{
#pragma unroll
		for (int j = 0; j < 3; j++)
			last[j] ^= state[j];
	}
	s1 += memshift;
	s2 += memshift;

	for (int i = 1; i < Ncol; i++)
	{
#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= LD4S(s1 + j) + LD4S(s2 + j);

		round_lyra(state);

		s1 += memshift;
		s2 += memshift;
	}

#pragma unroll
	for (int j = 0; j < 3; j++)
		state[j] ^= last[j];
}

__constant__ uint28 blake2b_IV[2] = {
	0xf3bcc908lu, 0x6a09e667lu,
	0x84caa73blu, 0xbb67ae85lu,
	0xfe94f82blu, 0x3c6ef372lu,
	0x5f1d36f1lu, 0xa54ff53alu,
	0xade682d1lu, 0x510e527flu,
	0x2b3e6c1flu, 0x9b05688clu,
	0xfb41bd6blu, 0x1f83d9ablu,
	0x137e2179lu, 0x5be0cd19lu
};

__constant__ uint28 Mask[2] = {
	0x00000020lu, 0x00000000lu,
	0x00000020lu, 0x00000000lu,
	0x00000020lu, 0x00000000lu,
	0x00000001lu, 0x00000000lu,
	0x00000004lu, 0x00000000lu,
	0x00000004lu, 0x00000000lu,
	0x00000080lu, 0x00000000lu,
	0x00000000lu, 0x01000000lu
};

__global__ __launch_bounds__(64, 1)
void lyra2v2_gpu_hash_32_1(uint32_t threads, uint32_t startNounce, uint2 *outputHash)
{
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;

	uint28 state[4];

	if (thread < threads)
	{
		/*
		state[0].x = state[1].x = __ldg(&outputHash[thread + threads * 0]);
		state[0].y = state[1].y = __ldg(&outputHash[thread + threads * 1]);
		state[0].z = state[1].z = __ldg(&outputHash[thread + threads * 2]);
		state[0].w = state[1].w = __ldg(&outputHash[thread + threads * 3]);
		*/
		state[0].x = state[1].x = __ldg(&outputHash[thread * 8 + 0]);
		state[0].y = state[1].y = __ldg(&outputHash[thread * 8 + 1]);
		state[0].z = state[1].z = __ldg(&outputHash[thread * 8 + 2]);
		state[0].w = state[1].w = __ldg(&outputHash[thread * 8 + 3]);
		state[2] = blake2b_IV[0];
		state[3] = blake2b_IV[1];

#pragma unroll 2
		for (int i = 0; i<12; i++)
			round_lyra(state);

		state[0] ^= Mask[0];
		state[1] ^= Mask[1];

#pragma unroll 2
		for (int i = 0; i<12; i++)
			round_lyra(state);

		DState[blockDim.x * gridDim.x * 0 + blockDim.x * blockIdx.x + threadIdx.x] = state[0];
		DState[blockDim.x * gridDim.x * 1 + blockDim.x * blockIdx.x + threadIdx.x] = state[1];
		DState[blockDim.x * gridDim.x * 2 + blockDim.x * blockIdx.x + threadIdx.x] = state[2];
		DState[blockDim.x * gridDim.x * 3 + blockDim.x * blockIdx.x + threadIdx.x] = state[3];

	} //thread
}

__global__ __launch_bounds__(TPB52, 1)
void lyra2v2_gpu_hash_32_2(uint32_t threads, uint32_t startNounce, uint64_t *outputHash)
{
	const uint32_t thread = blockDim.y * blockIdx.x + threadIdx.y;

	if (thread < threads)
	{
		uint2 state[4];
		state[0] = ((uint2*)DState)[(0 * gridDim.x * blockDim.y + thread) * blockDim.x + threadIdx.x];
		state[1] = ((uint2*)DState)[(1 * gridDim.x * blockDim.y + thread) * blockDim.x + threadIdx.x];
		state[2] = ((uint2*)DState)[(2 * gridDim.x * blockDim.y + thread) * blockDim.x + threadIdx.x];
		state[3] = ((uint2*)DState)[(3 * gridDim.x * blockDim.y + thread) * blockDim.x + threadIdx.x];

		reduceDuplexRowSetupV2(state);

		reduceDuplexRowtV2(state);

		((uint2*)DState)[(0 * gridDim.x * blockDim.y + thread) * blockDim.x + threadIdx.x] = state[0];
		((uint2*)DState)[(1 * gridDim.x * blockDim.y + thread) * blockDim.x + threadIdx.x] = state[1];
		((uint2*)DState)[(2 * gridDim.x * blockDim.y + thread) * blockDim.x + threadIdx.x] = state[2];
		((uint2*)DState)[(3 * gridDim.x * blockDim.y + thread) * blockDim.x + threadIdx.x] = state[3];
	} //thread
}

__global__ __launch_bounds__(64, 1)
void lyra2v2_gpu_hash_32_3(uint32_t threads, uint32_t startNounce, uint2 *outputHash)
{
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;

	uint28 state[4];

	if (thread < threads)
	{
		state[0] = __ldg4(&DState[blockDim.x * gridDim.x * 0 + blockDim.x * blockIdx.x + threadIdx.x]);
		state[1] = __ldg4(&DState[blockDim.x * gridDim.x * 1 + blockDim.x * blockIdx.x + threadIdx.x]);
		state[2] = __ldg4(&DState[blockDim.x * gridDim.x * 2 + blockDim.x * blockIdx.x + threadIdx.x]);
		state[3] = __ldg4(&DState[blockDim.x * gridDim.x * 3 + blockDim.x * blockIdx.x + threadIdx.x]);

#pragma unroll 2
		for (int i = 0; i < 12; i++)
			round_lyra(state);

		/*
		outputHash[thread + threads * 0] = state[0].x;
		outputHash[thread + threads * 1] = state[0].y;
		outputHash[thread + threads * 2] = state[0].z;
		outputHash[thread + threads * 3] = state[0].w;
		*/
		outputHash[thread * 8 + 0] = state[0].x;
		outputHash[thread * 8 + 1] = state[0].y;
		outputHash[thread * 8 + 2] = state[0].z;
		outputHash[thread * 8 + 3] = state[0].w;

	} //thread
}

__host__
void lyra2v2_cpu_init(int thr_id, uint32_t threads, uint64_t *d_matrix)
{
	// just assign the device pointer allocated in main loop
	cudaMemcpyToSymbol(DState, &d_matrix, sizeof(uint64_t*), 0, cudaMemcpyHostToDevice);
}

__host__
void lyra2v2_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *g_hash, int order)
{
	const uint32_t tpb = TPB52;

	// the matrix lives in dynamic shared memory: memshift * Nrow * Ncol uint2
	// per thread of the block
	const size_t shared_mem = memshift * Nrow * Ncol * sizeof(uint2) * tpb;

	dim3 grid1((threads * 4 + tpb - 1) / tpb);
	dim3 block1(4, tpb >> 2);

	dim3 grid2((threads + 64 - 1) / 64);
	dim3 block2(64);

	lyra2v2_gpu_hash_32_1 << <grid2, block2 >> > (threads, startNounce, (uint2*)g_hash);

	lyra2v2_gpu_hash_32_2 << <grid1, block1, shared_mem >> > (threads, startNounce, g_hash);

	lyra2v2_gpu_hash_32_3 << <grid2, block2 >> > (threads, startNounce, (uint2*)g_hash);
	//MyStreamSynchronize(NULL, order, thr_id);
}
