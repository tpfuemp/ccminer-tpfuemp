// Auf Groestlcoin spezialisierte Version von Groestl inkl. Bitslice

#include <stdio.h>
#include <memory.h>

#include "cuda_helper.h"

#ifdef __INTELLISENSE__
#define __CUDA_ARCH__ 500
#define __byte_perm(x,y,n) x
#endif

#include "miner.h"
#include "cuda/selftest_gate.cuh"

__constant__ uint32_t pTarget[8]; // Single GPU
__constant__ uint32_t groestlcoin_gpu_msg[32];

static uint32_t *d_resultNonce[MAX_GPUS];

#include "cuda/groestl512_device.cuh"

#define SWAB32(x) cuda_swab32(x)

/* Per-nonce hash, shared by the mining kernel and the init self-test.
 * out_state is valid only on the first of each group of 4 threads. */
__device__ __forceinline__
void groestlcoin_quad_hash(uint32_t nounce, uint32_t *out_state)
{
	// GROESTL
	uint32_t paddedInput[8];

	#pragma unroll 8
	for(int k=0;k<8;k++) paddedInput[k] = groestlcoin_gpu_msg[4*k+threadIdx.x%4];

	if ((threadIdx.x % 4) == 3)
		paddedInput[4] = SWAB32(nounce);  // 4*4+3 = 19

	uint32_t msgBitsliced[8];
	to_bitslice_quad(paddedInput, msgBitsliced);

	uint32_t state[8];
	for (int round=0; round<2; round++)
	{
		groestl512_progressMessage_quad(state, msgBitsliced);

		if (round < 1)
		{
			// Two chained rounds, padding included.
			msgBitsliced[ 0] = __byte_perm(state[ 0], 0x00800100, 0x4341 + ((threadIdx.x%4)==3)*0x2000);
			msgBitsliced[ 1] = __byte_perm(state[ 1], 0x00800100, 0x4341);
			msgBitsliced[ 2] = __byte_perm(state[ 2], 0x00800100, 0x4341);
			msgBitsliced[ 3] = __byte_perm(state[ 3], 0x00800100, 0x4341);
			msgBitsliced[ 4] = __byte_perm(state[ 4], 0x00800100, 0x4341);
			msgBitsliced[ 5] = __byte_perm(state[ 5], 0x00800100, 0x4341);
			msgBitsliced[ 6] = __byte_perm(state[ 6], 0x00800100, 0x4341);
			msgBitsliced[ 7] = __byte_perm(state[ 7], 0x00800100, 0x4341 + ((threadIdx.x%4)==0)*0x0010);
		}
	}

	// Only the first of each 4 threads receives the result hash
	from_bitslice_quad(state, out_state);
}

__global__ __launch_bounds__(256, 4)
void groestlcoin_gpu_hash_quad(uint32_t threads, uint32_t startNounce, uint32_t *resNounce)
{
	// durch 4 dividieren, weil jeweils 4 Threads zusammen ein Hash berechnen
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x) / 4;
	if (thread < threads)
	{
		uint32_t nounce = startNounce + thread;

		uint32_t out_state[16];
		groestlcoin_quad_hash(nounce, out_state);

		if (threadIdx.x % 4 == 0)
		{
			int i, position = -1;
			bool rc = true;

			#pragma unroll 8
			for (i = 7; i >= 0; i--) {
				if (out_state[i] > pTarget[i]) {
					if(position < i) {
						position = i;
						rc = false;
					}
				 }
				 if (out_state[i] < pTarget[i]) {
					if(position < i) {
						position = i;
						rc = true;
					}
				 }
			}

			// Atomic: several lanes can report in one launch, and a
			// read-compare-write would drop the lower nonce.
			if (rc)
				atomicMin(resNounce, nounce);
		}
	}
}

/* Init self-test. The shared xfamily self-test covers the quad-bitslice
 * permutation; this covers the rest -- two chained compressions, padding and
 * the nonce patch -- against groestlhash(). */
__global__
void groestlcoin_selftest_kernel(uint32_t nounce, uint32_t *out)
{
	uint32_t out_state[16];
	groestlcoin_quad_hash(nounce, out_state);

	if (threadIdx.x % 4 == 0) {
		#pragma unroll 8
		for (int i = 0; i < 8; i++) out[i] = out_state[i];
	}
}

/* One header + nonce through the device path, arranged as setBlock does.
 * Returns false only when CUDA itself failed. */
static bool groestlcoin_selftest_hash(const uint32_t *pdata, uint32_t nounce, uint32_t *out8)
{
	uint32_t endiandata[20], msgBlock[32] = { 0 };

	for (int k = 0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	memcpy(&msgBlock[0], endiandata, 80);
	msgBlock[20] = 0x80;
	msgBlock[31] = 0x01000000;

	if (cudaMemcpyToSymbol(groestlcoin_gpu_msg, msgBlock, 128) != cudaSuccess)
		return selftest_cuda_fault();

	uint32_t *d_out = NULL;
	if (cudaMalloc(&d_out, 8 * sizeof(uint32_t)) != cudaSuccess)
		return selftest_cuda_fault();

	groestlcoin_selftest_kernel <<< 1, 4 >>> (nounce, d_out);

	bool ok = cudaDeviceSynchronize() == cudaSuccess
	       && cudaMemcpy(out8, d_out, 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost) == cudaSuccess;
	cudaFree(d_out);

	return ok ? true : selftest_cuda_fault();
}

/* CPU reference, nonce placed as scanhash's candidate re-verify places it. */
static void groestlcoin_selftest_oracle(const uint32_t *pdata, uint32_t nounce, uint32_t *vhash)
{
	uint32_t endiandata[20];

	for (int k = 0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);
	endiandata[19] = swab32(nounce);

	groestlhash(vhash, endiandata);
}

__host__
bool groestlcoin_device_selftest(int thr_id)
{
	uint32_t pdata[20], gpu[8], cpu[8], gpu_flipped[8];
	bool kat0 = false, kat1 = false, neg = false;

	for (int i = 0; i < 20; i++)
		pdata[i] = 0x04030201u * (uint32_t)(i + 1);

	/* two nonces: a digest that ignored the nonce patch would still pass one */
	if (groestlcoin_selftest_hash(pdata, 0x00000000u, gpu)) {
		groestlcoin_selftest_oracle(pdata, 0x00000000u, cpu);
		kat0 = memcmp(gpu, cpu, 32) == 0;
	}
	if (groestlcoin_selftest_hash(pdata, 0xdeadbeefu, gpu)) {
		groestlcoin_selftest_oracle(pdata, 0xdeadbeefu, cpu);
		kat1 = memcmp(gpu, cpu, 32) == 0;
	}

	/* Negative leg on the device: a flipped header bit must change the digest.
	 * Comparing a host-side constant would be vacuous. */
	pdata[0] ^= 1u;
	if (groestlcoin_selftest_hash(pdata, 0xdeadbeefu, gpu_flipped))
		neg = memcmp(gpu_flipped, gpu, 32) != 0;
	pdata[0] ^= 1u;

	const bool passed = kat0 && kat1 && neg;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "groestl self-test FAILED (kat %d%d neg %d)",
			(int)kat0, (int)kat1, (int)neg);

	return selftest_gate(thr_id, "groestl", passed);
}

__host__
void groestlcoin_cpu_init(int thr_id, uint32_t threads)
{
	// populates cuda_arch[] for this device (global init state, kept)
	cuda_get_arch(thr_id);

	CUDA_SAFE_CALL(cudaMalloc(&d_resultNonce[thr_id], sizeof(uint32_t)));

	groestlcoin_device_selftest(thr_id);
}

__host__
void groestlcoin_cpu_free(int thr_id)
{
	cudaFree(d_resultNonce[thr_id]);
}

__host__
void groestlcoin_cpu_setBlock(int thr_id, void *data, void *pTargetIn)
{
	uint32_t msgBlock[32] = { 0 };

	memcpy(&msgBlock[0], data, 80);

	// Erweitere die Nachricht auf den Nachrichtenblock (padding)
	// Unsere Nachricht hat 80 Byte
	msgBlock[20] = 0x80;
	msgBlock[31] = 0x01000000;

	// groestl512 braucht hierfür keinen CPU-Code (die einzige Runde wird
	// auf der GPU ausgeführt)

	// Blockheader setzen (korrekte Nonce und Hefty Hash fehlen da drin noch)
	cudaMemcpyToSymbol(groestlcoin_gpu_msg, msgBlock, 128);

	cudaMemset(d_resultNonce[thr_id], 0xFF, sizeof(uint32_t));
	cudaMemcpyToSymbol(pTarget, pTargetIn, 32);
}

__host__
void groestlcoin_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *resNonce)
{
	uint32_t threadsperblock = 256;

	// Compute 3.0 benutzt die registeroptimierte Quad Variante mit Warp Shuffle
	// mit den Quad Funktionen brauchen wir jetzt 4 threads pro Hash, daher Faktor 4 bei der Blockzahl
	int factor = 4;

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid(factor*((threads + threadsperblock-1)/threadsperblock));
	dim3 block(threadsperblock);

	cudaMemset(d_resultNonce[thr_id], 0xFF, sizeof(uint32_t));
	groestlcoin_gpu_hash_quad <<<grid, block>>> (threads, startNounce, d_resultNonce[thr_id]);

	// Strategisches Sleep Kommando zur Senkung der CPU Last
	// MyStreamSynchronize(NULL, 0, thr_id);

	cudaMemcpy(resNonce, d_resultNonce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);
}
