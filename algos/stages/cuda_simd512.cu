/*
 * SIMD-512 multi-kernel stage (shared x-family stage; layout B — lives in the
 * unbranded algos/stages/, docs/coding-guideline.md).
 *
 * Deliberately-kept fusion boundary (§6.1): no register-resident device fn — it
 * is an expand -> d_temp4 -> compress pipeline, so the fused kernel treats it as
 * a stage boundary. The FFT message expansion + the compress round machinery
 * live in cuda_simd512_func.cuh; this TU is the standalone 64-byte launcher
 * (simd512_cpu_hash_64 = expand_64 then compress_64) + init/free.
 *
 * Based on the 2 Christians, klaus_t, Tanguy Pruvot, tsiv and SP (2013-2016);
 * Provos Alexis 2016; optimised by sp 2018 (+20% on the GTX 1080 Ti).
 *
 * NOTE: the unused simd+next-stage FUSED compress variants (echo/whirlpool/
 * hamsi/fugue + the _final target-screen kernels) and their embedded device
 * machinery were removed (zero consumers repo-wide) — a suprminer-era
 * simd->next fusion nobody in this tree wired up.
 */

#include "miner.h"
#include "cuda_helper_alexis.h"
#include "cuda_vectors_alexis.h"

#ifdef __INTELLISENSE__
/* just for vstudio code colors */
#define __CUDA_ARCH__ 500
#endif

#define TPB50_1 128
#define TPB50_2 128
#define TPB52_1 128
#define TPB52_2 128

static uint4 *d_temp4[MAX_GPUS];
#include "cuda_simd512_func.cuh"

__global__ __launch_bounds__(128,5)
static void simd512_gpu_compress_64(uint32_t threads, uint32_t *g_hash,const uint4 *const __restrict__ g_fft4)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x)>>3;
	const uint32_t thr_offset = thread << 6; // thr_id * 128 (je zwei elemente)
	uint32_t IV[32];
	if (thread < threads){

		uint32_t *Hash = &g_hash[thread<<4];
//		Compression1(Hash, thread, g_fft4, g_state);
		uint32_t A[32];

		*(uint2x4*)&IV[ 0] = *(uint2x4*)&c_IV_512[ 0];
		*(uint2x4*)&IV[ 8] = *(uint2x4*)&c_IV_512[ 8];
		*(uint2x4*)&IV[16] = *(uint2x4*)&c_IV_512[16];
		*(uint2x4*)&IV[24] = *(uint2x4*)&c_IV_512[24];

		*(uint2x4*)&A[ 0] = __ldg4((uint2x4*)&Hash[ 0]);
		*(uint2x4*)&A[ 8] = __ldg4((uint2x4*)&Hash[ 8]);

		#pragma unroll 16
		for(uint32_t i=0;i<16;i++)
			A[ i] = A[ i] ^ IV[ i];

		#pragma unroll 16
		for(uint32_t i=16;i<32;i++)
			A[ i] = IV[ i];

		Round8(A, thr_offset, g_fft4);

		STEP8_IF(&IV[ 0],32, 4,13,&A[ 0],&A[ 8],&A[16],&A[24]);
		STEP8_IF(&IV[ 8],33,13,10,&A[24],&A[ 0],&A[ 8],&A[16]);
		STEP8_IF(&IV[16],34,10,25,&A[16],&A[24],&A[ 0],&A[ 8]);
		STEP8_IF(&IV[24],35,25, 4,&A[ 8],&A[16],&A[24],&A[ 0]);

		#pragma unroll 32
		for(uint32_t i=0;i<32;i++){
			IV[ i] = A[ i];
		}

		A[ 0] ^= 512;

		Round8_0_final(A, 3,23,17,27);
		Round8_1_final(A,28,19,22, 7);
		Round8_2_final(A,29, 9,15, 5);
		Round8_3_final(A, 4,13,10,25);
		STEP8_IF(&IV[ 0],32, 4,13, &A[ 0], &A[ 8], &A[16], &A[24]);
		STEP8_IF(&IV[ 8],33,13,10, &A[24], &A[ 0], &A[ 8], &A[16]);
		STEP8_IF(&IV[16],34,10,25, &A[16], &A[24], &A[ 0], &A[ 8]);
		STEP8_IF(&IV[24],35,25, 4, &A[ 8], &A[16], &A[24], &A[ 0]);

		*(uint2x4*)&Hash[ 0] = *(uint2x4*)&A[ 0];
		*(uint2x4*)&Hash[ 8] = *(uint2x4*)&A[ 8];
	}
}

__host__
int simd512_cpu_init(int thr_id, uint32_t threads)
{
	cudaMalloc(&d_temp4[thr_id], 64*sizeof(uint4)*threads);

	return 0;
}

__host__
void simd512_cpu_free(int thr_id){
	cudaFree(d_temp4[thr_id]);
}

__host__
void simd512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
	int dev_id = device_map[thr_id];

	uint32_t tpb = TPB52_1;
	if (device_sm[dev_id] <= 500) tpb = TPB50_1;
	const dim3 grid1((8*threads + tpb - 1) / tpb);
	const dim3 block1(tpb);

//	tpb = TPB52_2;
//	if (device_sm[dev_id] <= 500) tpb = TPB50_2;
//	const dim3 grid2((threads + tpb - 1) / tpb);
//	const dim3 block2(tpb);

	simd512_gpu_expand_64 <<<grid1, block1>>> (threads, d_hash, d_temp4[thr_id]);
	simd512_gpu_compress_64 <<< grid1, block1 >>> (threads, d_hash, d_temp4[thr_id]);
}

/* Legacy forwarders — the not-yet-migrated x11-family consumers (c11/sib/fresh/
 * s3/veltor/0x10/phi/timetravel/bitcore/x11evo + x13/x15/x17/qubit/... ) still
 * call the x11_simd512_* names; removed once they call the bare simd512_* ones. */
__host__ int  x11_simd512_cpu_init(int thr_id, uint32_t threads) { return simd512_cpu_init(thr_id, threads); }
__host__ void x11_simd512_cpu_free(int thr_id) { simd512_cpu_free(thr_id); }
__host__ void x11_simd512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order) {
	simd512_cpu_hash_64(thr_id, threads, startNounce, d_nonceVector, d_hash, order);
}
