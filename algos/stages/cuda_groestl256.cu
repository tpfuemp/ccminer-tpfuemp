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

/* The tables, resident in device memory, in T0up..T3dn order. */
static __device__ uint32_t d_T[8][256];

/* Shared layout: four uint2 tables carry all eight (bytes 4..7 use the half swap), each in R
 * copies; lane l reads copy l % R. groestl256_shared_bytes() must match G256_R per arch. */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define G256_R 8
#else
#define G256_R 4
#endif
#define G256_T(k, ix) (((uint2*)mixtabs)[(((k) * 256 + (ix)) * G256_R) + (threadIdx.x % G256_R)])
#define RSTT(d0, d1, a, b0, b1, b2, b3, b4, b5, b6, b7) do { \
	const uint2 u0 = G256_T(0, B32_0(a[b0])), u1 = G256_T(1, B32_1(a[b1])); \
	const uint2 u2 = G256_T(2, B32_2(a[b2])), u3 = G256_T(3, B32_3(a[b3])); \
	const uint2 w0 = G256_T(0, B32_0(a[b4])), w1 = G256_T(1, B32_1(a[b5])); \
	const uint2 w2 = G256_T(2, B32_2(a[b6])), w3 = G256_T(3, B32_3(a[b7])); \
	t[d0] = u0.x ^ u1.x ^ u2.x ^ u3.x ^ w0.y ^ w1.y ^ w2.y ^ w3.y; \
	t[d1] = u0.y ^ u1.y ^ u2.y ^ u3.y ^ w0.x ^ w1.x ^ w2.x ^ w3.x; \
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

	for (int e = threadIdx.x; e < 4 * 256 * G256_R; e += blockDim.x) {
		const int k = e / (256 * G256_R), x = (e / G256_R) % 256;
		((uint2*)mixtabs)[e] = make_uint2(__ldg(&d_T[2 * k][x]), __ldg(&d_T[2 * k + 1][x]));
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


bool groestl256_device_selftest(int thr_id);

/* bytes of the replicated table for this device; must match G256_R of the arch it runs */
static size_t groestl256_shared_bytes(int thr_id)
{
	return (size_t)4 * 256 * ((device_sm[device_map[thr_id]] >= 800) ? 8 : 4) * sizeof(uint2);
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
	// the sm_80+ layout needs more than the default 48 KB of dynamic shared memory
	cudaFuncSetAttribute(groestl256_gpu_hash_32, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)groestl256_shared_bytes(thr_id));
	groestl256_device_selftest(thr_id);
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

	const size_t shared_size = groestl256_shared_bytes(thr_id);
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

/* ------------------------------------------------------------------ self-test
 * Init-time KAT (the stage returns nonces): each slot alone must report at target v and not
 * at v - 1 (v = sph_groestl256 bytes 28..31), and a ragged batch must return the two lowest
 * candidates. Clobbers pTarget, which scanhash sets before every launch. Fail-closed. */
extern "C" {
#include "sph/sph_groestl.h"
}
#include "cuda/stage_selftest.cuh"

#define GROESTL256_ST_N     256
#define GROESTL256_ST_EXACT 64

static bool groestl256_st_launch(int thr_id, uint32_t threads, uint32_t start, uint64_t *d,
	uint32_t target, uint32_t *res)
{
	const uint32_t t[8] = { 0, 0, 0, 0, 0, 0, 0, target };
	groestl256_setTarget(t);
	res[0] = groestl256_cpu_hash_32(thr_id, threads, start, d, -1);
	res[1] = groestl256_getSecNonce(thr_id, 1);
	return cudaGetLastError() == cudaSuccess;
}

static int groestl256_st_cmp(const void *a, const void *b)
{
	const uint32_t x = *(const uint32_t*)a, y = *(const uint32_t*)b;
	return (x > y) - (x < y);
}

__host__
bool groestl256_device_selftest(int thr_id)
{
	static bool tested = false, passed = false;
	if (tested) return passed;
	tested = true;

	const int n = GROESTL256_ST_N, nr = GROESTL256_ST_N - 57;
	const uint32_t start = 0x7fffff00u;
	uint8_t *in = (uint8_t*)malloc((size_t)n * 32);
	uint64_t *soa = (uint64_t*)malloc((size_t)n * 32);
	uint32_t v[GROESTL256_ST_N];
	uint64_t *d_aos = NULL, *d_soa = NULL;
	if (!in || !soa || cudaMalloc(&d_aos, (size_t)n * 32) != cudaSuccess || cudaMalloc(&d_soa, (size_t)nr * 32) != cudaSuccess) {
		free(in); free(soa); cudaFree(d_aos);
		return selftest_gate(thr_id, "groestl256", selftest_cuda_fault());
	}
	stkat_fill(in, n, 32, 0x47523536u);
	for (int i = 0; i < n; i++) {
		uint8_t dg[32];
		sph_groestl256_context c;
		sph_groestl256_init(&c);
		sph_groestl256(&c, in + i * 32, 32);
		sph_groestl256_close(&c, dg);
		memcpy(&v[i], dg + 28, 4);
	}
	stkat_to_soa32(in, soa, nr);
	bool cuda_ok = cudaMemcpy(d_aos, in, (size_t)n * 32, cudaMemcpyHostToDevice) == cudaSuccess
		&& cudaMemcpy(d_soa, soa, (size_t)nr * 32, cudaMemcpyHostToDevice) == cudaSuccess;

	int bad_exact = 0, first = -1;
	uint32_t res[2];
	for (int i = 0; cuda_ok && i < GROESTL256_ST_EXACT; i++) {
		cuda_ok = groestl256_st_launch(thr_id, 1, start + i, d_aos + 4 * i, v[i], res);
		bool ok = res[0] == start + (uint32_t)i && res[1] == UINT32_MAX;
		if (cuda_ok && v[i] != 0) {
			cuda_ok = groestl256_st_launch(thr_id, 1, start + i, d_aos + 4 * i, v[i] - 1, res);
			ok = ok && res[0] == UINT32_MAX;
		}
		if (!ok) { if (first < 0) first = i; bad_exact++; }
	}

	/* targets: none, all, and the values of a few ranks */
	uint32_t sorted[GROESTL256_ST_N];
	memcpy(sorted, v, sizeof(uint32_t) * nr);
	qsort(sorted, nr, sizeof(uint32_t), groestl256_st_cmp);
	const uint32_t targets[] = { sorted[0] ? sorted[0] - 1 : 0, UINT32_MAX,
		sorted[0], sorted[1], sorted[2], sorted[7], sorted[31], sorted[nr / 2] };
	int bad_batch = 0;
	for (int t = 0; cuda_ok && t < (int)(sizeof(targets) / sizeof(targets[0])); t++) {
		uint32_t e[2] = { UINT32_MAX, UINT32_MAX };
		for (int i = 0; i < nr; i++)
			if (v[i] <= targets[t]) {
				if (e[0] == UINT32_MAX) e[0] = start + i;
				else if (e[1] == UINT32_MAX) e[1] = start + i;
			}
		cuda_ok = groestl256_st_launch(thr_id, nr, start, d_soa, targets[t], res);
		if (res[0] != e[0] || res[1] != e[1]) bad_batch++;
	}

	cudaFree(d_aos); cudaFree(d_soa); free(in); free(soa);
	if (!cuda_ok)
		return selftest_gate(thr_id, "groestl256", selftest_cuda_fault());
	passed = bad_exact == 0 && bad_batch == 0;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "groestl256 device self-test FAILED (exact: %d of %d slots wrong, first %d; batch: %d targets wrong)",
			bad_exact, GROESTL256_ST_EXACT, first, bad_batch);
	else
		gpulog(LOG_DEBUG, thr_id, "groestl256 device self-test passed");
	passed = selftest_gate(thr_id, "groestl256", passed);
	return passed;
}
