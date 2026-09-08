#include <memory.h>

#define SPH_C32(x)    ((uint32_t)(x ## U))
#define SPH_T32(x)    ((x) & SPH_C32(0xFFFFFFFF))

#include "cuda_helper.h"

static uint32_t *h_GNonces[MAX_GPUS];
static uint32_t *d_GNonces[MAX_GPUS];

__constant__ uint32_t pTarget[8];

#define C32e(x) \
	  ((SPH_C32(x) >> 24) \
	| ((SPH_C32(x) >>  8) & SPH_C32(0x0000FF00)) \
	| ((SPH_C32(x) <<  8) & SPH_C32(0x00FF0000)) \
	| ((SPH_C32(x) << 24) & SPH_C32(0xFF000000)))

#define PC32up(j, r)   ((uint32_t)((j) + (r)))
#define PC32dn(j, r)   0
#define QC32up(j, r)   0xFFFFFFFF
#define QC32dn(j, r)   (((uint32_t)(r) << 24) ^ SPH_T32(~((uint32_t)(j) << 24)))

#define B32_0(x)    __byte_perm(x, 0, 0x4440)
//((x) & 0xFF)
#define B32_1(x)    __byte_perm(x, 0, 0x4441)
//(((x) >> 8) & 0xFF)
#define B32_2(x)    __byte_perm(x, 0, 0x4442)
//(((x) >> 16) & 0xFF)
#define B32_3(x)    __byte_perm(x, 0, 0x4443)
//((x) >> 24)

/* All eight tables are read from shared memory; each block stages them in
 * once. */
#define T0up(x) (*((uint32_t*)mixtabs + (    (x))))
#define T0dn(x) (*((uint32_t*)mixtabs + (256+(x))))
#define T1up(x) (*((uint32_t*)mixtabs + (512+(x))))
#define T1dn(x) (*((uint32_t*)mixtabs + (768+(x))))
#define T2up(x) (*((uint32_t*)mixtabs + (1024+(x))))
#define T2dn(x) (*((uint32_t*)mixtabs + (1280+(x))))
#define T3up(x) (*((uint32_t*)mixtabs + (1536+(x))))
#define T3dn(x) (*((uint32_t*)mixtabs + (1792+(x))))

/* The tables, resident in device memory, in T0up..T3dn order. */
static __device__ uint32_t d_T[8][256];

#define RSTT(d0, d1, a, b0, b1, b2, b3, b4, b5, b6, b7) do { \
	t[d0] = T0up(B32_0(a[b0])) \
		^ T1up(B32_1(a[b1])) \
		^ T2up(B32_2(a[b2])) \
		^ T3up(B32_3(a[b3])) \
		^ T0dn(B32_0(a[b4])) \
		^ T1dn(B32_1(a[b5])) \
		^ T2dn(B32_2(a[b6])) \
		^ T3dn(B32_3(a[b7])); \
	t[d1] = T0dn(B32_0(a[b0])) \
		^ T1dn(B32_1(a[b1])) \
		^ T2dn(B32_2(a[b2])) \
		^ T3dn(B32_3(a[b3])) \
		^ T0up(B32_0(a[b4])) \
		^ T1up(B32_1(a[b5])) \
		^ T2up(B32_2(a[b6])) \
		^ T3up(B32_3(a[b7])); \
	} while (0)


extern uint32_t T0up_cpu[];
extern uint32_t T0dn_cpu[];
extern uint32_t T1up_cpu[];
extern uint32_t T1dn_cpu[];
extern uint32_t T2up_cpu[];
extern uint32_t T2dn_cpu[];
extern uint32_t T3up_cpu[];
extern uint32_t T3dn_cpu[];

__device__ __forceinline__
void groestl256_perm_P(uint32_t thread,uint32_t *a, char *mixtabs)
{
	#pragma unroll 10
	for (int r = 0; r<10; r++)
	{
		uint32_t t[16];

		a[0x0] ^= PC32up(0x00, r);
		a[0x2] ^= PC32up(0x10, r);
		a[0x4] ^= PC32up(0x20, r);
		a[0x6] ^= PC32up(0x30, r);
		a[0x8] ^= PC32up(0x40, r);
		a[0xA] ^= PC32up(0x50, r);
		a[0xC] ^= PC32up(0x60, r);
		a[0xE] ^= PC32up(0x70, r);
		RSTT(0x0, 0x1, a, 0x0, 0x2, 0x4, 0x6, 0x9, 0xB, 0xD, 0xF);
		RSTT(0x2, 0x3, a, 0x2, 0x4, 0x6, 0x8, 0xB, 0xD, 0xF, 0x1);
		RSTT(0x4, 0x5, a, 0x4, 0x6, 0x8, 0xA, 0xD, 0xF, 0x1, 0x3);
		RSTT(0x6, 0x7, a, 0x6, 0x8, 0xA, 0xC, 0xF, 0x1, 0x3, 0x5);
		RSTT(0x8, 0x9, a, 0x8, 0xA, 0xC, 0xE, 0x1, 0x3, 0x5, 0x7);
		RSTT(0xA, 0xB, a, 0xA, 0xC, 0xE, 0x0, 0x3, 0x5, 0x7, 0x9);
		RSTT(0xC, 0xD, a, 0xC, 0xE, 0x0, 0x2, 0x5, 0x7, 0x9, 0xB);
		RSTT(0xE, 0xF, a, 0xE, 0x0, 0x2, 0x4, 0x7, 0x9, 0xB, 0xD);

		#pragma unroll 16
		for (int k = 0; k<16; k++)
			a[k] = t[k];
	}
}

__device__ __forceinline__
void groestl256_perm_Q(uint32_t thread, uint32_t *a, char *mixtabs)
{
	#pragma unroll
	for (int r = 0; r<10; r++)
	{
		uint32_t t[16];

		a[0x0] ^= QC32up(0x00, r);
		a[0x1] ^= QC32dn(0x00, r);
		a[0x2] ^= QC32up(0x10, r);
		a[0x3] ^= QC32dn(0x10, r);
		a[0x4] ^= QC32up(0x20, r);
		a[0x5] ^= QC32dn(0x20, r);
		a[0x6] ^= QC32up(0x30, r);
		a[0x7] ^= QC32dn(0x30, r);
		a[0x8] ^= QC32up(0x40, r);
		a[0x9] ^= QC32dn(0x40, r);
		a[0xA] ^= QC32up(0x50, r);
		a[0xB] ^= QC32dn(0x50, r);
		a[0xC] ^= QC32up(0x60, r);
		a[0xD] ^= QC32dn(0x60, r);
		a[0xE] ^= QC32up(0x70, r);
		a[0xF] ^= QC32dn(0x70, r);
		RSTT(0x0, 0x1, a, 0x2, 0x6, 0xA, 0xE, 0x1, 0x5, 0x9, 0xD);
		RSTT(0x2, 0x3, a, 0x4, 0x8, 0xC, 0x0, 0x3, 0x7, 0xB, 0xF);
		RSTT(0x4, 0x5, a, 0x6, 0xA, 0xE, 0x2, 0x5, 0x9, 0xD, 0x1);
		RSTT(0x6, 0x7, a, 0x8, 0xC, 0x0, 0x4, 0x7, 0xB, 0xF, 0x3);
		RSTT(0x8, 0x9, a, 0xA, 0xE, 0x2, 0x6, 0x9, 0xD, 0x1, 0x5);
		RSTT(0xA, 0xB, a, 0xC, 0x0, 0x4, 0x8, 0xB, 0xF, 0x3, 0x7);
		RSTT(0xC, 0xD, a, 0xE, 0x2, 0x6, 0xA, 0xD, 0x1, 0x5, 0x9);
		RSTT(0xE, 0xF, a, 0x0, 0x4, 0x8, 0xC, 0xF, 0x3, 0x7, 0xB);

		#pragma unroll
		for (int k = 0; k<16; k++)
			a[k] = t[k];
	}
}

// Do NOT raise minBlocks: this kernel is latency-bound over the shared T-table and needs
// its registers to keep loads in flight. Trading them for occupancy measures slower.
__global__ __launch_bounds__(256,1)
void groestl256_gpu_hash_32(uint32_t threads, uint32_t startNounce, uint64_t *outputHash, uint32_t *resNonces)
{
	extern __shared__ char mixtabs[];

	if (threadIdx.x < 256) {
		*((uint32_t*)mixtabs + (threadIdx.x)) = __ldg(&d_T[0][threadIdx.x]);
		*((uint32_t*)mixtabs + (256 + threadIdx.x)) = __ldg(&d_T[1][threadIdx.x]);
		*((uint32_t*)mixtabs + (512 + threadIdx.x)) = __ldg(&d_T[2][threadIdx.x]);
		*((uint32_t*)mixtabs + (768 + threadIdx.x)) = __ldg(&d_T[3][threadIdx.x]);
		*((uint32_t*)mixtabs + (1024 + threadIdx.x)) = __ldg(&d_T[4][threadIdx.x]);
		*((uint32_t*)mixtabs + (1280 + threadIdx.x)) = __ldg(&d_T[5][threadIdx.x]);
		*((uint32_t*)mixtabs + (1536 + threadIdx.x)) = __ldg(&d_T[6][threadIdx.x]);
		*((uint32_t*)mixtabs + (1792 + threadIdx.x)) = __ldg(&d_T[7][threadIdx.x]);
	}

	__syncthreads();

	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		// GROESTL
		uint32_t message[16];
		uint32_t state[16];

		#pragma unroll
		for (int k = 0; k<4; k++)
			LOHI(message[2*k], message[2*k+1], outputHash[k*threads+thread]);

		#pragma unroll
		for (int k = 9; k<15; k++)
			message[k] = 0;

		message[8] = 0x80;
		message[15] = 0x01000000;

		#pragma unroll 16
		for (int u = 0; u<16; u++)
			state[u] = message[u];

		state[15] ^= 0x10000;

		// Perm

		groestl256_perm_P(thread, state, mixtabs);
		state[15] ^= 0x10000;
		groestl256_perm_Q(thread, message, mixtabs);
		#pragma unroll 16
		for (int u = 0; u<16; u++) state[u] ^= message[u];
		#pragma unroll 16
		for (int u = 0; u<16; u++) message[u] = state[u];
		groestl256_perm_P(thread, message, mixtabs);
		state[14] ^= message[14];
		state[15] ^= message[15];

		uint32_t nonce = startNounce + thread;
		if (state[15] <= pTarget[7]) {
			// Keep the two lowest candidates in one pass. atomicMin returns the previous minimum, so
			// the displaced value moves to slot 1; reading resNonces[0] outside the atomic loses one.
			const uint32_t prev = atomicMin(&resNonces[0], nonce);
			if (prev != UINT32_MAX)
				atomicMin(&resNonces[1], max(prev, nonce));
		}
	}
}


__host__
void groestl256_cpu_init(int thr_id, uint32_t threads)
{
	/* Upload the tables once per device, in T0up..T3dn order. */
	const size_t tab = sizeof(uint32_t) * 256;
	cudaMemcpyToSymbol(d_T, T0up_cpu, tab, 0 * tab, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_T, T0dn_cpu, tab, 1 * tab, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_T, T1up_cpu, tab, 2 * tab, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_T, T1dn_cpu, tab, 3 * tab, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_T, T2up_cpu, tab, 4 * tab, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_T, T2dn_cpu, tab, 5 * tab, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_T, T3up_cpu, tab, 6 * tab, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_T, T3dn_cpu, tab, 7 * tab, cudaMemcpyHostToDevice);

	cudaMalloc(&d_GNonces[thr_id], 2*sizeof(uint32_t));
	cudaMallocHost(&h_GNonces[thr_id], 2*sizeof(uint32_t));
}

__host__
void groestl256_cpu_free(int thr_id)
{
	cudaFree(d_GNonces[thr_id]);
	cudaFreeHost(h_GNonces[thr_id]);
}

__host__
uint32_t groestl256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_outputHash, int order)
{
	uint32_t result = UINT32_MAX;
	cudaMemset(d_GNonces[thr_id], 0xff, 2*sizeof(uint32_t));
	const uint32_t threadsperblock = 256;

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	size_t shared_size = 8 * 256 * sizeof(uint32_t);
	groestl256_gpu_hash_32<<<grid, block, shared_size>>>(threads, startNounce, d_outputHash, d_GNonces[thr_id]);

	MyStreamSynchronize(NULL, order, thr_id);

	// get first found nonce
	cudaMemcpy(h_GNonces[thr_id], d_GNonces[thr_id], 1*sizeof(uint32_t), cudaMemcpyDeviceToHost);
	result = *h_GNonces[thr_id];

	return result;
}

__host__
uint32_t groestl256_getSecNonce(int thr_id, int num)
{
	uint32_t results[2];
	memset(results, 0xFF, sizeof(results));
	cudaMemcpy(results, d_GNonces[thr_id], sizeof(results), cudaMemcpyDeviceToHost);
	if (results[1] == results[0])
		return UINT32_MAX;
	return results[num];
}

__host__
void groestl256_setTarget(const void *pTargetIn)
{
	cudaMemcpyToSymbol(pTarget, pTargetIn, 32, 0, cudaMemcpyHostToDevice);
}
