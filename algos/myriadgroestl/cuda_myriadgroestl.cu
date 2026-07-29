// Auf Myriadcoin spezialisierte Version von Groestl inkl. Bitslice

#include <stdio.h>
#include <memory.h>

#include "cuda_helper.h"

#ifdef __INTELLISENSE__
#define __CUDA_ARCH__ 500
#define __funnelshift_r(x,y,n) (x >> n)
#define atomicExch(p,x) x
#endif

#include "cuda/groestl512_device.cuh"
#include "cuda/sha256_device.cuh"   /* c_sha256_K / h_sha256_K */

// globaler Speicher für alle HeftyHashes aller Threads
static uint32_t *d_outputHashes[MAX_GPUS];
static uint32_t *d_resultNonces[MAX_GPUS];

__constant__ uint32_t pTarget[2]; // Same for all GPU
__constant__ uint32_t myriadgroestl_gpu_msg[32];

/* K comes from cuda/sha256_device.cuh; this is the algo-specific K+W fold for the
 * fully constant second block, expanded at init. */
__constant__ uint32_t myr_sha256_gpu_constantTable2[64];

const uint32_t myr_sha256_cpu_w2Table[] = {
	0x80000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
	0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000200,
	0x80000000, 0x01400000, 0x00205000, 0x00005088, 0x22000800, 0x22550014, 0x05089742, 0xa0000020,
	0x5a880000, 0x005c9400, 0x0016d49d, 0xfa801f00, 0xd33225d0, 0x11675959, 0xf6e6bfda, 0xb30c1549,
	0x08b2b050, 0x9d7c4c27, 0x0ce2a393, 0x88e6e1ea, 0xa52b4335, 0x67a16f49, 0xd732016f, 0x4eeb2e91,
	0x5dbf55e5, 0x8eee2335, 0xe2bc5ec2, 0xa83f4394, 0x45ad78f7, 0x36f3d0cd, 0xd99c05e8, 0xb0511dc7,
	0x69bc7ac4, 0xbd11375b, 0xe3ba71e5, 0x3b209ff2, 0x18feee17, 0xe25ad9e7, 0x13375046, 0x0515089d,
	0x4f0d0f04, 0x2627484e, 0x310128d2, 0xc668b434, 0x420841cc, 0x62d311b8, 0xe59ba771, 0x85a7a484
};

#define SWAB32(x) cuda_swab32(x)

/* ROTR32 comes from cuda_helper.h (identical definition). */

#define R(x, n)         ((x) >> (n))
#define Ch(x, y, z)     ((x & (y ^ z)) ^ z)
#define Maj(x, y, z)    ((x & (y | z)) | (y & z))
#define S0(x)           (ROTR32(x, 2) ^ ROTR32(x, 13) ^ ROTR32(x, 22))
#define S1(x)           (ROTR32(x, 6) ^ ROTR32(x, 11) ^ ROTR32(x, 25))
#define s0(x)           (ROTR32(x, 7) ^ ROTR32(x, 18) ^ R(x, 3))
#define s1(x)           (ROTR32(x, 17) ^ ROTR32(x, 19) ^ R(x, 10))

__device__ __forceinline__
void myriadgroestl_gpu_sha256(uint32_t *message)
{
	uint32_t W1[16];
	#pragma unroll
	for(int k=0; k<16; k++)
		W1[k] = SWAB32(message[k]);

	uint32_t regs[8] = {
		0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
		0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
	};

	// Progress W1
	#pragma unroll
	for(int j=0; j<16; j++)
	{
		uint32_t T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + c_sha256_K[j] + W1[j];
		uint32_t T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);

		#pragma unroll 7
		for (int k=6; k >= 0; k--) regs[k+1] = regs[k];
		regs[0] = T1 + T2;
		regs[4] += T1;
	}

	// Progress W2...W3
	uint32_t W2[16];

	////// PART 1
	#pragma unroll
	for(int j=0; j<2; j++)
		W2[j] = s1(W1[14+j]) + W1[9+j] + s0(W1[1+j]) + W1[j];

	#pragma unroll 5
	for(int j=2; j<7;j++)
		W2[j] = s1(W2[j-2]) + W1[9+j] + s0(W1[1+j]) + W1[j];

	#pragma unroll
	for(int j=7; j<15; j++)
		W2[j] = s1(W2[j-2]) + W2[j-7] + s0(W1[1+j]) + W1[j];

	W2[15] = s1(W2[13]) + W2[8] + s0(W2[0]) + W1[15];

	// Round function
	#pragma unroll
	for(int j=0; j<16; j++)
	{
		uint32_t T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + c_sha256_K[j + 16] + W2[j];
		uint32_t T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);

		#pragma unroll 7
		for (int l=6; l >= 0; l--) regs[l+1] = regs[l];
		regs[0] = T1 + T2;
		regs[4] += T1;
	}

	////// PART 2
	#pragma unroll
	for(int j=0; j<2; j++)
		W1[j] = s1(W2[14+j]) + W2[9+j] + s0(W2[1+j]) + W2[j];
	#pragma unroll 5
	for(int j=2; j<7; j++)
		W1[j] = s1(W1[j-2]) + W2[9+j] + s0(W2[1+j]) + W2[j];

	#pragma unroll
	for(int j=7; j<15; j++)
		W1[j] = s1(W1[j-2]) + W1[j-7] + s0(W2[1+j]) + W2[j];

	W1[15] = s1(W1[13]) + W1[8] + s0(W1[0]) + W2[15];

	// Round function
	#pragma unroll
	for(int j=0; j<16; j++)
	{
		uint32_t T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + c_sha256_K[j + 32] + W1[j];
		uint32_t T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);

		#pragma unroll 7
		for (int l=6; l >= 0; l--) regs[l+1] = regs[l];
		regs[0] = T1 + T2;
		regs[4] += T1;
	}

	////// PART 3
	#pragma unroll
	for(int j=0; j<2; j++)
		W2[j] = s1(W1[14+j]) + W1[9+j] + s0(W1[1+j]) + W1[j];

	#pragma unroll 5
	for(int j=2; j<7; j++)
		W2[j] = s1(W2[j-2]) + W1[9+j] + s0(W1[1+j]) + W1[j];

	#pragma unroll
	for(int j=7; j<15; j++)
		W2[j] = s1(W2[j-2]) + W2[j-7] + s0(W1[1+j]) + W1[j];

	W2[15] = s1(W2[13]) + W2[8] + s0(W2[0]) + W1[15];

	// Round function
	#pragma unroll
	for(int j=0; j<16; j++)
	{
		uint32_t T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + c_sha256_K[j + 48] + W2[j];
		uint32_t T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);

		#pragma unroll 7
		for (int l=6; l >= 0; l--) regs[l+1] = regs[l];
		regs[0] = T1 + T2;
		regs[4] += T1;
	}

	uint32_t hash[8] = {
		0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
		0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
	};

	#pragma unroll 8
	for(int k=0; k<8; k++)
		hash[k] += regs[k];

	/////
	///// 2nd Round (wegen Msg-Padding)
	/////
	#pragma unroll
	for(int k=0; k<8; k++)
		regs[k] = hash[k];

	// Progress W1
	#pragma unroll
	for(int j=0; j<64; j++)
	{
		uint32_t T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + myr_sha256_gpu_constantTable2[j];
		uint32_t T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);

		#pragma unroll 7
		for (int k=6; k >= 0; k--) regs[k+1] = regs[k];
		regs[0] = T1 + T2;
		regs[4] += T1;
	}

	/* only the top two words are needed: the caller compares them against the target */
	message[6] = SWAB32(hash[6] + regs[6]);
	message[7] = SWAB32(hash[7] + regs[7]);
}

__global__
//__launch_bounds__(256, 6) // we want <= 40 regs
void myriadgroestl_gpu_hash_sha(uint32_t threads, uint32_t startNounce, uint32_t *hashBuffer, uint32_t *resNonces)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint32_t nonce = startNounce + thread;

		uint32_t out_state[16];
		uint32_t *inpHash = &hashBuffer[16 * thread];

		#pragma unroll 16
		for (int i=0; i < 16; i++)
			out_state[i] = inpHash[i];

		myriadgroestl_gpu_sha256(out_state);

		// Proper 64-bit compare. The old `s7 <= t1 && s6 <= t0` also demanded the low
		// word be under target when the high word was strictly below, losing a
		// t1*(2^32-t0)/(t1*2^32+t0) share of valid nonces — nil when t1==0, but a third
		// of them at low difficulty and nearly all under --benchmark.
		if (out_state[7] < pTarget[1] || (out_state[7] == pTarget[1] && out_state[6] <= pTarget[0]))
		{
			uint32_t tmp = atomicExch(&resNonces[0], nonce);
			if (tmp != UINT32_MAX)
				resNonces[1] = tmp;
		}
	}
}

__global__
__launch_bounds__(256, 4)
void myriadgroestl_gpu_hash_quad(uint32_t threads, uint32_t startNounce, uint32_t *hashBuffer)
{
	// durch 4 dividieren, weil jeweils 4 Threads zusammen ein Hash berechnen
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x) / 4;
	if (thread < threads)
	{
		// GROESTL
		uint32_t paddedInput[8];
		#pragma unroll 8
		for(int k=0; k<8; k++)
			paddedInput[k] = myriadgroestl_gpu_msg[4*k+threadIdx.x%4];

		uint32_t nounce = startNounce + thread;
		if ((threadIdx.x % 4) == 3)
			paddedInput[4] = SWAB32(nounce);  // 4*4+3 = 19

		uint32_t msgBitsliced[8];
		to_bitslice_quad(paddedInput, msgBitsliced);

		uint32_t state[8];
		groestl512_progressMessage_quad(state, msgBitsliced);

		uint32_t out_state[16];
		from_bitslice_quad(state, out_state);

		if ((threadIdx.x & 0x03) == 0)
		{
			uint32_t *outpHash = &hashBuffer[16 * thread];
			#pragma unroll 16
			for(int k=0; k<16; k++) outpHash[k] = out_state[k];
		}
	}
}

// Setup Function
__host__
void myriadgroestl_cpu_init(int thr_id, uint32_t threads)
{
	uint32_t temp[64];
	for(int i=0; i<64; i++)
		temp[i] = myr_sha256_cpu_w2Table[i] + h_sha256_K[i];

	cudaMemcpyToSymbol( myr_sha256_gpu_constantTable2, temp, sizeof(uint32_t) * 64 );

	cuda_get_arch(thr_id);

	cudaMalloc(&d_outputHashes[thr_id], (size_t) 64 * threads);
	cudaMalloc(&d_resultNonces[thr_id], 2 * sizeof(uint32_t));
}

__host__
void myriadgroestl_cpu_free(int thr_id)
{
	cudaFree(d_outputHashes[thr_id]);
	cudaFree(d_resultNonces[thr_id]);
}

__host__
void myriadgroestl_cpu_setBlock(int thr_id, void *data, uint32_t *pTargetIn)
{
	uint32_t msgBlock[32] = { 0 };
	memcpy(&msgBlock[0], data, 80);
	msgBlock[20] = 0x80;
	msgBlock[31] = 0x01000000;

	cudaMemcpyToSymbol(myriadgroestl_gpu_msg, msgBlock, 128);
	cudaMemcpyToSymbol(pTarget, &pTargetIn[6], 2 * sizeof(uint32_t));
}

__host__
void myriadgroestl_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *resNounce)
{
	uint32_t threadsperblock = 256;

	cudaMemset(d_resultNonces[thr_id], 0xFF, 2 * sizeof(uint32_t));

	// Compute 3.0 benutzt die registeroptimierte Quad Variante mit Warp Shuffle
	// mit den Quad Funktionen brauchen wir jetzt 4 threads pro Hash, daher Faktor 4 bei der Blockzahl
	const int factor = 4;

	dim3 grid(factor*((threads + threadsperblock-1)/threadsperblock));
	dim3 block(threadsperblock);

	myriadgroestl_gpu_hash_quad <<< grid, block >>> (threads, startNounce, d_outputHashes[thr_id]);

	dim3 grid2((threads + threadsperblock-1)/threadsperblock);
	myriadgroestl_gpu_hash_sha <<< grid2, block >>> (threads, startNounce, d_outputHashes[thr_id], d_resultNonces[thr_id]);

	cudaMemcpy(resNounce, d_resultNonces[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
}
