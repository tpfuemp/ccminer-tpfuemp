/*
 * ECHO-512 shared device library (docs/coding-guideline.md §3), tpruvot/x11
 * formulation over the cuda/aes_device.cuh tables (static-init).
 *
 * cuda_echo_round needs the 1024-word shared AES table: kernels declare
 * __shared__ uint32_t sharedMemory[1024], call echo_gpu_init(sharedMemory)
 * (cooperative, threads < 128) and MUST __syncthreads() before hashing —
 * the shavite/echo shared-AES-table race lesson.
 *
 * Extracted verbatim from x11/cuda_x11_echo.cu. The alexis x16 variant
 * (algos/stages/cuda_echo512_64.cu, algos/stages/cuda_echo512_aes.cuh tables) is a
 * separate formulation and stays in its TU until the x16 pilot.
 */

#ifndef CUDA_ECHO512_DEVICE_CUH
#define CUDA_ECHO512_DEVICE_CUH

#include <cuda_helper.h>
#ifndef CUDA_LYRA_VECTOR_H
#include <cuda_vectors.h>
#include <cuda_vector_uint2x4.h>
#endif
#include "cuda/aes_device.cuh"

__device__ __forceinline__ void AES_2ROUND(
	const uint32_t* __restrict__ sharedMemory,
	uint32_t &x0, uint32_t &x1, uint32_t &x2, uint32_t &x3,
	uint32_t &k0)
{
	uint32_t y0, y1, y2, y3;

	aes_round(sharedMemory,
		x0, x1, x2, x3,
		k0,
		y0, y1, y2, y3);

	aes_round(sharedMemory,
		y0, y1, y2, y3,
		x0, x1, x2, x3);

	// hier werden wir ein carry brauchen (oder auch nicht)
	k0++;
}

__device__ __forceinline__
void cuda_echo_round(
	const uint32_t *const __restrict__ sharedMemory, uint32_t *const __restrict__  hash)
{
	const uint32_t P[48] = {
	0xe7e9f5f5, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
	0xa4213d7e, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
	//8-12
	0x01425eb8,	0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
	0x65978b09, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
	//21-25
	0x2cb6b661, 0x6b23b3b3, 0xcf93a7cf, 0x9d9d3751,
	0x9ac2dea3, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
	//34-38
	0x579f9f33, 0xfbfbfbfb, 0xfbfbfbfb, 0xefefd3c7,
	0xdbfde1dd, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
	0x34514d9e, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
	0xb134347e, 0xea6f7e7e, 0xbd7731bd, 0x8a8a1968,
	0x14b8a457, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
	0x265f4382, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af
	//58-61
	};
	uint32_t k0;
	uint32_t h[16];

	#pragma unroll 16
	for (int i = 0; i < 16; i++)
	{
		h[i] = hash[i];
	}

	k0 = 512 + 8;

	#pragma unroll 4
	for (int idx = 0; idx < 16; idx += 4)
	{
		AES_2ROUND(sharedMemory,
			h[idx + 0], h[idx + 1], h[idx + 2], h[idx + 3], k0);
	}
	k0 += 4;

	uint32_t W[64];

	#pragma unroll 4
	for (int i = 0; i < 4; i++)
	{
		uint32_t a = P[i];
		uint32_t b = P[i + 4];
		uint32_t c = h[i + 8];
		uint32_t d = P[i + 8];

		uint32_t ab = a ^ b;
		uint32_t bc = b ^ c;
		uint32_t cd = c ^ d;


		uint32_t t =  (ab & 0x80808080);
		uint32_t t2 = (bc & 0x80808080);
		uint32_t t3 = (cd & 0x80808080);

		uint32_t abx = (t >> 7) * 27 ^ ((ab^t) << 1);
		uint32_t bcx = (t2 >> 7) * 27 ^ ((bc^t2) << 1);
		uint32_t cdx = (t3 >> 7) * 27 ^ ((cd^t3) << 1);

		W[0 + i] = abx ^ bc ^ d;
		W[0 + i + 4] = bcx ^ a ^ cd;
		W[0 + i + 8] = cdx ^ ab ^ d;
		W[0 + i + 12] = abx ^ bcx ^ cdx ^ ab ^ c;

		a = P[12 + i];
		b = h[i + 4];
		c = P[12 + i + 4];
		d = P[12 + i + 8];

		ab = a ^ b;
		bc = b ^ c;
		cd = c ^ d;


		t = (ab & 0x80808080);
		t2 = (bc & 0x80808080);
		t3 = (cd & 0x80808080);

		abx = (t >> 7) * 27 ^ ((ab^t) << 1);
		bcx = (t2 >> 7) * 27 ^ ((bc^t2) << 1);
		cdx = (t3 >> 7) * 27 ^ ((cd^t3) << 1);

		W[16 + i] = abx ^ bc ^ d;
		W[16 + i + 4] = bcx ^ a ^ cd;
		W[16 + i + 8] = cdx ^ ab ^ d;
		W[16 + i + 12] = abx ^ bcx ^ cdx ^ ab ^ c;

		a = h[i];
		b = P[24 + i + 0];
		c = P[24 + i + 4];
		d = P[24 + i + 8];

		ab = a ^ b;
		bc = b ^ c;
		cd = c ^ d;


		t = (ab & 0x80808080);
		t2 = (bc & 0x80808080);
		t3 = (cd & 0x80808080);

		abx = (t >> 7) * 27 ^ ((ab^t) << 1);
		bcx = (t2 >> 7) * 27 ^ ((bc^t2) << 1);
		cdx = (t3 >> 7) * 27 ^ ((cd^t3) << 1);

		W[32 + i] = abx ^ bc ^ d;
		W[32 + i + 4] = bcx ^ a ^ cd;
		W[32 + i + 8] = cdx ^ ab ^ d;
		W[32 + i + 12] = abx ^ bcx ^ cdx ^ ab ^ c;

		a = P[36 + i ];
		b = P[36 + i + 4];
		c = P[36 + i + 8];
		d = h[i + 12];

		ab = a ^ b;
		bc = b ^ c;
		cd = c ^ d;

		t = (ab & 0x80808080);
		t2 = (bc & 0x80808080);
		t3 = (cd & 0x80808080);

		abx = (t >> 7) * 27 ^ ((ab^t) << 1);
		bcx = (t2 >> 7) * 27 ^ ((bc^t2) << 1);
		cdx = (t3 >> 7) * 27 ^ ((cd^t3) << 1);

		W[48 + i] = abx ^ bc ^ d;
		W[48 + i + 4] = bcx ^ a ^ cd;
		W[48 + i + 8] = cdx ^ ab ^ d;
		W[48 + i + 12] = abx ^ bcx ^ cdx ^ ab ^ c;

	}

	for (int k = 1; k < 10; k++)
	{

		// Big Sub Words
		#pragma unroll 4
		for (int idx = 0; idx < 64; idx += 16)
		{
			AES_2ROUND(sharedMemory,
				W[idx + 0], W[idx + 1], W[idx + 2], W[idx + 3],
				k0);
			AES_2ROUND(sharedMemory,
				W[idx + 4], W[idx + 5], W[idx + 6], W[idx + 7],
				k0);
			AES_2ROUND(sharedMemory,
				W[idx + 8], W[idx + 9], W[idx + 10], W[idx + 11],
				k0);
			AES_2ROUND(sharedMemory,
				W[idx + 12], W[idx + 13], W[idx + 14], W[idx + 15],
				k0);
		}

		// Shift Rows
		#pragma unroll 4
		for (int i = 0; i < 4; i++)
		{
			uint32_t t;

			/// 1, 5, 9, 13
			t = W[4 + i];
			W[4 + i] = W[20 + i];
			W[20 + i] = W[36 + i];
			W[36 + i] = W[52 + i];
			W[52 + i] = t;

			// 2, 6, 10, 14
			t = W[8 + i];
			W[8 + i] = W[40 + i];
			W[40 + i] = t;
			t = W[24 + i];
			W[24 + i] = W[56 + i];
			W[56 + i] = t;

			// 15, 11, 7, 3
			t = W[60 + i];
			W[60 + i] = W[44 + i];
			W[44 + i] = W[28 + i];
			W[28 + i] = W[12 + i];
			W[12 + i] = t;
		}

		// Mix Columns
		#pragma unroll 4
		for (int i = 0; i < 4; i++) // Schleife über je 2*uint32_t
		{
			#pragma unroll 4
			for (int idx = 0; idx < 64; idx += 16) // Schleife über die elemnte
			{
				uint32_t a = W[idx + i];
				uint32_t b = W[idx + i + 4];
				uint32_t c = W[idx + i + 8];
				uint32_t d = W[idx + i + 12];

				uint32_t ab = a ^ b;
				uint32_t bc = b ^ c;
				uint32_t cd = c ^ d;

				uint32_t t, t2, t3;
				t = (ab & 0x80808080);
				t2 = (bc & 0x80808080);
				t3 = (cd & 0x80808080);

				uint32_t abx = (t >> 7) * 27 ^ ((ab^t) << 1);
				uint32_t bcx = (t2 >> 7) * 27 ^ ((bc^t2) << 1);
				uint32_t cdx = (t3 >> 7) * 27 ^ ((cd^t3) << 1);

				W[idx + i] = abx ^ bc ^ d;
				W[idx + i + 4] = bcx ^ a ^ cd;
				W[idx + i + 8] = cdx ^ ab ^ d;
				W[idx + i + 12] = abx ^ bcx ^ cdx ^ ab ^ c;
			}
		}
	}

	#pragma unroll
	for (int i = 0; i < 16; i += 4)
	{
		W[i] ^= W[32 + i] ^ 512;
		W[i + 1] ^= W[32 + i + 1];
		W[i + 2] ^= W[32 + i + 2];
		W[i + 3] ^= W[32 + i + 3];
	}

	#pragma unroll
	for (int i = 0; i < 16; i++)
		hash[i] ^= W[i];
}

__device__ __forceinline__
void echo_gpu_init(uint32_t *const __restrict__ sharedMemory)
{
	/* each thread startup will fill a uint32 */
	if (threadIdx.x < 128) {
		sharedMemory[threadIdx.x] = d_AES0[threadIdx.x];
		sharedMemory[threadIdx.x + 256] = d_AES1[threadIdx.x];
		sharedMemory[threadIdx.x + 512] = d_AES2[threadIdx.x];
		sharedMemory[threadIdx.x + 768] = d_AES3[threadIdx.x];

		sharedMemory[threadIdx.x + 64 * 2] = d_AES0[threadIdx.x + 64 * 2];
		sharedMemory[threadIdx.x + 64 * 2 + 256] = d_AES1[threadIdx.x + 64 * 2];
		sharedMemory[threadIdx.x + 64 * 2 + 512] = d_AES2[threadIdx.x + 64 * 2];
		sharedMemory[threadIdx.x + 64 * 2 + 768] = d_AES3[threadIdx.x + 64 * 2];
	}
}

/* Single ECHO permutation round (tpruvot flat-table formulation) — the body
 * cuda_echo_round_80 iterates. Not __forceinline__: called in a loop, keep it
 * a real call to avoid code bloat. */
__device__
static void echo_round(uint32_t* const sharedMemory, uint32_t *W, uint32_t &k0)
{
	// Big Sub Words
	#pragma unroll 16
	for (int idx = 0; idx < 16; idx++) {
		AES_2ROUND(sharedMemory, W[(idx << 2) + 0], W[(idx << 2) + 1], W[(idx << 2) + 2], W[(idx << 2) + 3], k0);
	}

	// Shift Rows
	#pragma unroll 4
	for (int i = 0; i < 4; i++)
	{
		uint32_t t[4];
		/// 1, 5, 9, 13
		t[0] = W[i +  4];
		t[1] = W[i +  8];
		t[2] = W[i + 24];
		t[3] = W[i + 60];

		W[i +  4] = W[i + 20];
		W[i +  8] = W[i + 40];
		W[i + 24] = W[i + 56];
		W[i + 60] = W[i + 44];

		W[i + 20] = W[i + 36];
		W[i + 40] = t[1];
		W[i + 56] = t[2];
		W[i + 44] = W[i + 28];

		W[i + 28] = W[i + 12];
		W[i + 12] = t[3];
		W[i + 36] = W[i + 52];
		W[i + 52] = t[0];
	}

	// Mix Columns
	#pragma unroll 4
	for (int i = 0; i < 4; i++)
	{
		#pragma unroll 4
		for (int idx = 0; idx < 64; idx += 16)
		{
			uint32_t a[4];
			a[0] = W[idx + i];
			a[1] = W[idx + i + 4];
			a[2] = W[idx + i + 8];
			a[3] = W[idx + i + 12];

			uint32_t ab = a[0] ^ a[1];
			uint32_t bc = a[1] ^ a[2];
			uint32_t cd = a[2] ^ a[3];

			uint32_t t, t2, t3;
			t  = (ab & 0x80808080);
			t2 = (bc & 0x80808080);
			t3 = (cd & 0x80808080);

			uint32_t abx = (t  >> 7) * 27U ^ ((ab^t)  << 1);
			uint32_t bcx = (t2 >> 7) * 27U ^ ((bc^t2) << 1);
			uint32_t cdx = (t3 >> 7) * 27U ^ ((cd^t3) << 1);

			W[idx + i] = bc ^ a[3] ^ abx;
			W[idx + i +  4] = a[0] ^ cd ^ bcx;
			W[idx + i +  8] = ab ^ a[3] ^ cdx;
			W[idx + i + 12] = ab ^ a[2] ^ (abx ^ bcx ^ cdx);
		}
	}
}

/* ECHO-512 of an 80-byte header (data[0..18] + nonce), digest OUT in hash[0..15].
 * The 80-byte first-stage path for x16-family algorithms. */
__device__ __forceinline__
void cuda_echo_round_80(uint32_t *const __restrict__ sharedMemory, uint32_t *const __restrict__ data, const uint32_t nonce, uint32_t *hash)
{
	uint32_t h[29]; // <= 127 bytes input

	#pragma unroll 8
	for (int i = 0; i < 18; i += 2)
		AS_UINT2(&h[i]) = AS_UINT2(&data[i]);
	h[18] = data[18];
	h[19] = cuda_swab32(nonce);
	h[20] = 0x80;
	h[21] = h[22] = h[23] = h[24] = h[25] = h[26] = 0;
	//((uint8_t*)h)[80] = 0x80;
	//((uint8_t*)h)[128-17] = 0x02;
	//((uint8_t*)h)[128-16] = 0x80;
	//((uint8_t*)h)[128-15] = 0x02;
	h[27] = 0x2000000;
	h[28] = 0x280;
	//h[29] = h[30] = h[31] = 0;

	uint32_t k0 = 640; // bitlen
	uint32_t W[64];

	#pragma unroll 8
	for (int i = 0; i < 32; i+=4) {
		W[i] = 512; // L
		W[i+1] = 0; // H
		W[i+2] = 0; // X
		W[i+3] = 0;
	}

	uint32_t Z[16];
	#pragma unroll
	for (int i = 0;  i<16; i++) Z[i] = W[i];
	#pragma unroll
	for (int i = 32; i<61; i++) W[i] = h[i - 32];
	#pragma unroll
	for (int i = 61; i<64; i++) W[i] = 0;

	for (int i = 0; i < 10; i++)
		echo_round(sharedMemory, W, k0);

	#pragma unroll 16
	for (int i = 0; i < 16; i++) {
		Z[i] ^= h[i] ^ W[i] ^ W[i + 32];
	}

	#pragma unroll 8
	for (int i = 0; i < 16; i += 2)
		AS_UINT2(&hash[i]) = AS_UINT2(&Z[i]);
}


/* ======================================================================== *
 * ECHO-512 alexis formulation (x16 chains on sm >= 500), from
 * algos/stages/cuda_echo512_aes.cuh + algos/stages/cuda_echo512_64.cu. Its own AES
 * tables (c_echo_AES0/c_echo_AES3, static-init; table reads mix __ldg with
 * shared memory — INTENSIVE_GMF baked in). Kernels declare
 * __shared__ uint32_t sharedMemory[4][256], fill it with one of the
 * echo_aes_gpu_init* variants matching their block size and MUST
 * __syncthreads() before hashing.
 * ======================================================================== */

#define INTENSIVE_GMF

#ifdef __CUDA_ARCH__
__device__ __forceinline__
uint32_t ECHO_ROL8(const uint32_t a) {
	return __byte_perm(a, 0, 0x2103);
}
__device__ __forceinline__
uint32_t ECHO_ROR8(const uint32_t a) {
	return __byte_perm(a, 0, 0x0321);
}
__device__ __forceinline__
uint32_t ECHO_ROL16(const uint32_t a) {
	return __byte_perm(a, 0, 0x1032);
}
#else
#define ECHO_ROL8(u)  ROTL32(u, 8)
#define ECHO_ROR8(u)  ROTR32(u, 8)
#define ECHO_ROL16(u) ROTL32(u,16)
#endif

static __device__ uint32_t c_echo_AES0[256] = {
	0xA56363C6, 0x847C7CF8, 0x997777EE, 0x8D7B7BF6, 0x0DF2F2FF, 0xBD6B6BD6, 0xB16F6FDE, 0x54C5C591,
	0x50303060, 0x03010102, 0xA96767CE, 0x7D2B2B56, 0x19FEFEE7, 0x62D7D7B5, 0xE6ABAB4D, 0x9A7676EC,
	0x45CACA8F, 0x9D82821F, 0x40C9C989, 0x877D7DFA, 0x15FAFAEF, 0xEB5959B2, 0xC947478E, 0x0BF0F0FB,
	0xECADAD41, 0x67D4D4B3, 0xFDA2A25F, 0xEAAFAF45, 0xBF9C9C23, 0xF7A4A453, 0x967272E4, 0x5BC0C09B,
	0xC2B7B775, 0x1CFDFDE1, 0xAE93933D, 0x6A26264C, 0x5A36366C, 0x413F3F7E, 0x02F7F7F5, 0x4FCCCC83,
	0x5C343468, 0xF4A5A551, 0x34E5E5D1, 0x08F1F1F9, 0x937171E2, 0x73D8D8AB, 0x53313162, 0x3F15152A,
	0x0C040408, 0x52C7C795, 0x65232346, 0x5EC3C39D, 0x28181830, 0xA1969637, 0x0F05050A, 0xB59A9A2F,
	0x0907070E, 0x36121224, 0x9B80801B, 0x3DE2E2DF, 0x26EBEBCD, 0x6927274E, 0xCDB2B27F, 0x9F7575EA,
	0x1B090912, 0x9E83831D, 0x742C2C58, 0x2E1A1A34, 0x2D1B1B36, 0xB26E6EDC, 0xEE5A5AB4, 0xFBA0A05B,
	0xF65252A4, 0x4D3B3B76, 0x61D6D6B7, 0xCEB3B37D, 0x7B292952, 0x3EE3E3DD, 0x712F2F5E, 0x97848413,
	0xF55353A6, 0x68D1D1B9, 0x00000000, 0x2CEDEDC1, 0x60202040, 0x1FFCFCE3, 0xC8B1B179, 0xED5B5BB6,
	0xBE6A6AD4, 0x46CBCB8D, 0xD9BEBE67, 0x4B393972, 0xDE4A4A94, 0xD44C4C98, 0xE85858B0, 0x4ACFCF85,
	0x6BD0D0BB, 0x2AEFEFC5, 0xE5AAAA4F, 0x16FBFBED, 0xC5434386, 0xD74D4D9A, 0x55333366, 0x94858511,
	0xCF45458A, 0x10F9F9E9, 0x06020204, 0x817F7FFE, 0xF05050A0, 0x443C3C78, 0xBA9F9F25, 0xE3A8A84B,
	0xF35151A2, 0xFEA3A35D, 0xC0404080, 0x8A8F8F05, 0xAD92923F, 0xBC9D9D21, 0x48383870, 0x04F5F5F1,
	0xDFBCBC63, 0xC1B6B677, 0x75DADAAF, 0x63212142, 0x30101020, 0x1AFFFFE5, 0x0EF3F3FD, 0x6DD2D2BF,
	0x4CCDCD81, 0x140C0C18, 0x35131326, 0x2FECECC3, 0xE15F5FBE, 0xA2979735, 0xCC444488, 0x3917172E,
	0x57C4C493, 0xF2A7A755, 0x827E7EFC, 0x473D3D7A, 0xAC6464C8, 0xE75D5DBA, 0x2B191932, 0x957373E6,
	0xA06060C0, 0x98818119, 0xD14F4F9E, 0x7FDCDCA3, 0x66222244, 0x7E2A2A54, 0xAB90903B, 0x8388880B,
	0xCA46468C, 0x29EEEEC7, 0xD3B8B86B, 0x3C141428, 0x79DEDEA7, 0xE25E5EBC, 0x1D0B0B16, 0x76DBDBAD,
	0x3BE0E0DB, 0x56323264, 0x4E3A3A74, 0x1E0A0A14, 0xDB494992, 0x0A06060C, 0x6C242448, 0xE45C5CB8,
	0x5DC2C29F, 0x6ED3D3BD, 0xEFACAC43, 0xA66262C4, 0xA8919139, 0xA4959531, 0x37E4E4D3, 0x8B7979F2,
	0x32E7E7D5, 0x43C8C88B, 0x5937376E, 0xB76D6DDA, 0x8C8D8D01, 0x64D5D5B1, 0xD24E4E9C, 0xE0A9A949,
	0xB46C6CD8, 0xFA5656AC, 0x07F4F4F3, 0x25EAEACF, 0xAF6565CA, 0x8E7A7AF4, 0xE9AEAE47, 0x18080810,
	0xD5BABA6F, 0x887878F0, 0x6F25254A, 0x722E2E5C, 0x241C1C38, 0xF1A6A657, 0xC7B4B473, 0x51C6C697,
	0x23E8E8CB, 0x7CDDDDA1, 0x9C7474E8, 0x211F1F3E, 0xDD4B4B96, 0xDCBDBD61, 0x868B8B0D, 0x858A8A0F,
	0x907070E0, 0x423E3E7C, 0xC4B5B571, 0xAA6666CC, 0xD8484890, 0x05030306, 0x01F6F6F7, 0x120E0E1C,
	0xA36161C2, 0x5F35356A, 0xF95757AE, 0xD0B9B969, 0x91868617, 0x58C1C199, 0x271D1D3A, 0xB99E9E27,
	0x38E1E1D9, 0x13F8F8EB, 0xB398982B, 0x33111122, 0xBB6969D2, 0x70D9D9A9, 0x898E8E07, 0xA7949433,
	0xB69B9B2D, 0x221E1E3C, 0x92878715, 0x20E9E9C9, 0x49CECE87, 0xFF5555AA, 0x78282850, 0x7ADFDFA5,
	0x8F8C8C03, 0xF8A1A159, 0x80898909, 0x170D0D1A, 0xDABFBF65, 0x31E6E6D7, 0xC6424284, 0xB86868D0,
	0xC3414182, 0xB0999929, 0x772D2D5A, 0x110F0F1E, 0xCBB0B07B, 0xFC5454A8, 0xD6BBBB6D, 0x3A16162C
};

static __device__ uint32_t c_echo_AES3[256] = {
	0xC6A56363, 0xF8847C7C, 0xEE997777, 0xF68D7B7B, 0xFF0DF2F2, 0xD6BD6B6B, 0xDEB16F6F, 0x9154C5C5,
	0x60503030, 0x02030101, 0xCEA96767, 0x567D2B2B, 0xE719FEFE, 0xB562D7D7, 0x4DE6ABAB, 0xEC9A7676,
	0x8F45CACA, 0x1F9D8282, 0x8940C9C9, 0xFA877D7D, 0xEF15FAFA, 0xB2EB5959, 0x8EC94747, 0xFB0BF0F0,
	0x41ECADAD, 0xB367D4D4, 0x5FFDA2A2, 0x45EAAFAF, 0x23BF9C9C, 0x53F7A4A4, 0xE4967272, 0x9B5BC0C0,
	0x75C2B7B7, 0xE11CFDFD, 0x3DAE9393, 0x4C6A2626, 0x6C5A3636, 0x7E413F3F, 0xF502F7F7, 0x834FCCCC,
	0x685C3434, 0x51F4A5A5, 0xD134E5E5, 0xF908F1F1, 0xE2937171, 0xAB73D8D8, 0x62533131, 0x2A3F1515,
	0x080C0404, 0x9552C7C7, 0x46652323, 0x9D5EC3C3, 0x30281818, 0x37A19696, 0x0A0F0505, 0x2FB59A9A,
	0x0E090707, 0x24361212, 0x1B9B8080, 0xDF3DE2E2, 0xCD26EBEB, 0x4E692727, 0x7FCDB2B2, 0xEA9F7575,
	0x121B0909, 0x1D9E8383, 0x58742C2C, 0x342E1A1A, 0x362D1B1B, 0xDCB26E6E, 0xB4EE5A5A, 0x5BFBA0A0,
	0xA4F65252, 0x764D3B3B, 0xB761D6D6, 0x7DCEB3B3, 0x527B2929, 0xDD3EE3E3, 0x5E712F2F, 0x13978484,
	0xA6F55353, 0xB968D1D1, 0x00000000, 0xC12CEDED, 0x40602020, 0xE31FFCFC, 0x79C8B1B1, 0xB6ED5B5B,
	0xD4BE6A6A, 0x8D46CBCB, 0x67D9BEBE, 0x724B3939, 0x94DE4A4A, 0x98D44C4C, 0xB0E85858, 0x854ACFCF,
	0xBB6BD0D0, 0xC52AEFEF, 0x4FE5AAAA, 0xED16FBFB, 0x86C54343, 0x9AD74D4D, 0x66553333, 0x11948585,
	0x8ACF4545, 0xE910F9F9, 0x04060202, 0xFE817F7F, 0xA0F05050, 0x78443C3C, 0x25BA9F9F, 0x4BE3A8A8,
	0xA2F35151, 0x5DFEA3A3, 0x80C04040, 0x058A8F8F, 0x3FAD9292, 0x21BC9D9D, 0x70483838, 0xF104F5F5,
	0x63DFBCBC, 0x77C1B6B6, 0xAF75DADA, 0x42632121, 0x20301010, 0xE51AFFFF, 0xFD0EF3F3, 0xBF6DD2D2,
	0x814CCDCD, 0x18140C0C, 0x26351313, 0xC32FECEC, 0xBEE15F5F, 0x35A29797, 0x88CC4444, 0x2E391717,
	0x9357C4C4, 0x55F2A7A7, 0xFC827E7E, 0x7A473D3D, 0xC8AC6464, 0xBAE75D5D, 0x322B1919, 0xE6957373,
	0xC0A06060, 0x19988181, 0x9ED14F4F, 0xA37FDCDC, 0x44662222, 0x547E2A2A, 0x3BAB9090, 0x0B838888,
	0x8CCA4646, 0xC729EEEE, 0x6BD3B8B8, 0x283C1414, 0xA779DEDE, 0xBCE25E5E, 0x161D0B0B, 0xAD76DBDB,
	0xDB3BE0E0, 0x64563232, 0x744E3A3A, 0x141E0A0A, 0x92DB4949, 0x0C0A0606, 0x486C2424, 0xB8E45C5C,
	0x9F5DC2C2, 0xBD6ED3D3, 0x43EFACAC, 0xC4A66262, 0x39A89191, 0x31A49595, 0xD337E4E4, 0xF28B7979,
	0xD532E7E7, 0x8B43C8C8, 0x6E593737, 0xDAB76D6D, 0x018C8D8D, 0xB164D5D5, 0x9CD24E4E, 0x49E0A9A9,
	0xD8B46C6C, 0xACFA5656, 0xF307F4F4, 0xCF25EAEA, 0xCAAF6565, 0xF48E7A7A, 0x47E9AEAE, 0x10180808,
	0x6FD5BABA, 0xF0887878, 0x4A6F2525, 0x5C722E2E, 0x38241C1C, 0x57F1A6A6, 0x73C7B4B4, 0x9751C6C6,
	0xCB23E8E8, 0xA17CDDDD, 0xE89C7474, 0x3E211F1F, 0x96DD4B4B, 0x61DCBDBD, 0x0D868B8B, 0x0F858A8A,
	0xE0907070, 0x7C423E3E, 0x71C4B5B5, 0xCCAA6666, 0x90D84848, 0x06050303, 0xF701F6F6, 0x1C120E0E,
	0xC2A36161, 0x6A5F3535, 0xAEF95757, 0x69D0B9B9, 0x17918686, 0x9958C1C1, 0x3A271D1D, 0x27B99E9E,
	0xD938E1E1, 0xEB13F8F8, 0x2BB39898, 0x22331111, 0xD2BB6969, 0xA970D9D9, 0x07898E8E, 0x33A79494,
	0x2DB69B9B, 0x3C221E1E, 0x15928787, 0xC920E9E9, 0x8749CECE, 0xAAFF5555, 0x50782828, 0xA57ADFDF,
	0x038F8C8C, 0x59F8A1A1, 0x09808989, 0x1A170D0D, 0x65DABFBF, 0xD731E6E6, 0x84C64242, 0xD0B86868,
	0x82C34141, 0x29B09999, 0x5A772D2D, 0x1E110F0F, 0x7BCBB0B0, 0xA8FC5454, 0x6DD6BBBB, 0x2C3A1616
};

__device__ __forceinline__
void echo_aes_gpu_init_mt_256(uint32_t sharedMemory[4][256])
{
	/* each thread startup will fill a uint32 */
	if (threadIdx.x < 256) {
		uint32_t temp = __ldg(&c_echo_AES0[threadIdx.x]);
		sharedMemory[0][threadIdx.x] = temp;
		sharedMemory[1][threadIdx.x] = ECHO_ROL8(temp);
		sharedMemory[2][threadIdx.x] = ECHO_ROL16(temp);
		sharedMemory[3][threadIdx.x] = ECHO_ROR8(temp);
	}
}

__device__ __forceinline__
void echo_aes_gpu_init256(uint32_t sharedMemory[4][256])
{
	/* each thread startup will fill a uint32 */
	uint32_t temp = __ldg(&c_echo_AES0[threadIdx.x]);
	sharedMemory[0][threadIdx.x] = temp;
	sharedMemory[1][threadIdx.x] = ECHO_ROL8(temp);
	sharedMemory[2][threadIdx.x] = ECHO_ROL16(temp);
	sharedMemory[3][threadIdx.x] = ECHO_ROR8(temp);
}

__device__ __forceinline__
void echo_aes_gpu_init128(uint32_t sharedMemory[4][256])
{
	/* each thread startup will fill 2 uint32 */
	uint2 temp = __ldg(&((uint2*)&c_echo_AES0)[threadIdx.x]);

	sharedMemory[0][(threadIdx.x << 1) + 0] = temp.x;
	sharedMemory[0][(threadIdx.x << 1) + 1] = temp.y;
	sharedMemory[1][(threadIdx.x << 1) + 0] = ECHO_ROL8(temp.x);
	sharedMemory[1][(threadIdx.x << 1) + 1] = ECHO_ROL8(temp.y);
	sharedMemory[2][(threadIdx.x << 1) + 0] = ECHO_ROL16(temp.x);
	sharedMemory[2][(threadIdx.x << 1) + 1] = ECHO_ROL16(temp.y);
	sharedMemory[3][(threadIdx.x << 1) + 0] = ECHO_ROR8(temp.x);
	sharedMemory[3][(threadIdx.x << 1) + 1] = ECHO_ROR8(temp.y);
}

__device__ __forceinline__
void echo_aes_gpu_init_lt_256(uint32_t sharedMemory[4][256])
{
	if (threadIdx.x < 128) {
		/* each thread startup will fill 2 uint32 */
		uint2 temp = __ldg(&((uint2*)&c_echo_AES0)[threadIdx.x]);

		sharedMemory[0][(threadIdx.x << 1) + 0] = temp.x;
		sharedMemory[0][(threadIdx.x << 1) + 1] = temp.y;
		sharedMemory[1][(threadIdx.x << 1) + 0] = ECHO_ROL8(temp.x);
		sharedMemory[1][(threadIdx.x << 1) + 1] = ECHO_ROL8(temp.y);
		sharedMemory[2][(threadIdx.x << 1) + 0] = ECHO_ROL16(temp.x);
		sharedMemory[2][(threadIdx.x << 1) + 1] = ECHO_ROL16(temp.y);
		sharedMemory[3][(threadIdx.x << 1) + 0] = ECHO_ROR8(temp.x);
		sharedMemory[3][(threadIdx.x << 1) + 1] = ECHO_ROR8(temp.y);
	}
}

__device__ __forceinline__
static void echo_aes_round(const uint32_t sharedMemory[4][256], const uint32_t x0, const uint32_t x1, const uint32_t x2, const uint32_t x3,
	const uint32_t k0, uint32_t &y0, uint32_t &y1, uint32_t &y2, uint32_t &y3)
{
	y0 = __ldg(&c_echo_AES0[__byte_perm(x0, 0, 0x4440)]);
	y3 = sharedMemory[1][__byte_perm(x0, 0, 0x4441)];
	y2 = sharedMemory[2][__byte_perm(x0, 0, 0x4442)];
	y1 = __ldg(&c_echo_AES3[__byte_perm(x0, 0, 0x4443)]);

	y1 ^= sharedMemory[0][__byte_perm(x1, 0, 0x4440)];
	y0 ^= sharedMemory[1][__byte_perm(x1, 0, 0x4441)];
	y3 ^= sharedMemory[2][__byte_perm(x1, 0, 0x4442)];
#ifdef INTENSIVE_GMF
	y2 ^= __ldg(&c_echo_AES3[__byte_perm(x1, 0, 0x4443)]);
#else
	y2 ^= sharedMemory[3][__byte_perm(x1, 0, 0x4443)];
#endif

	y0 ^= k0;

	y2 ^= __ldg(&c_echo_AES0[__byte_perm(x2, 0, 0x4440)]);
	y1 ^= sharedMemory[1][__byte_perm(x2, 0, 0x4441)];
	y0 ^= sharedMemory[2][__byte_perm(x2, 0, 0x4442)];
	y3 ^= __ldg(&c_echo_AES3[__byte_perm(x2, 0, 0x4443)]);

	y3 ^= sharedMemory[0][__byte_perm(x3, 0, 0x4440)];
	y2 ^= sharedMemory[1][__byte_perm(x3, 0, 0x4441)];
	y1 ^= sharedMemory[2][__byte_perm(x3, 0, 0x4442)];
	y0 ^= __ldg(&c_echo_AES3[__byte_perm(x3, 0, 0x4443)]);
}

__device__ __forceinline__
static void echo_aes_round_LDG(const uint32_t sharedMemory[4][256], const uint32_t x0, const uint32_t x1, const uint32_t x2, const uint32_t x3,
	const uint32_t k0, uint32_t &y0, uint32_t &y1, uint32_t &y2, uint32_t &y3)
{
	y0 = __ldg(&c_echo_AES0[__byte_perm(x0, 0, 0x4440)]);
	y3 = sharedMemory[1][__byte_perm(x0, 0, 0x4441)];
	y2 = sharedMemory[2][__byte_perm(x0, 0, 0x4442)];
	y1 = __ldg(&c_echo_AES3[__byte_perm(x0, 0, 0x4443)]);

	y1 ^= sharedMemory[0][__byte_perm(x1, 0, 0x4440)];
	y0 ^= sharedMemory[1][__byte_perm(x1, 0, 0x4441)];
	y3 ^= sharedMemory[2][__byte_perm(x1, 0, 0x4442)];
	y2 ^= __ldg(&c_echo_AES3[__byte_perm(x1, 0, 0x4443)]);

	y0 ^= k0;

	y2 ^= __ldg(&c_echo_AES0[__byte_perm(x2, 0, 0x4440)]);
	y1 ^= sharedMemory[1][__byte_perm(x2, 0, 0x4441)];
	y0 ^= sharedMemory[2][__byte_perm(x2, 0, 0x4442)];
	y3 ^= __ldg(&c_echo_AES3[__byte_perm(x2, 0, 0x4443)]);

	y3 ^= __ldg(&c_echo_AES0[__byte_perm(x3, 0, 0x4440)]);
	y2 ^= sharedMemory[1][__byte_perm(x3, 0, 0x4441)];
	y1 ^= sharedMemory[2][__byte_perm(x3, 0, 0x4442)];
	y0 ^= __ldg(&c_echo_AES3[__byte_perm(x3, 0, 0x4443)]);
}

__device__ __forceinline__
static void echo_aes_round(const uint32_t sharedMemory[4][256], const uint32_t x0, const uint32_t x1, const uint32_t x2, const uint32_t x3,
	uint32_t &y0, uint32_t &y1, uint32_t &y2, uint32_t &y3)
{
	y0 = sharedMemory[0][__byte_perm(x0, 0, 0x4440)];
	y3 = sharedMemory[1][__byte_perm(x0, 0, 0x4441)];
	y2 = sharedMemory[2][__byte_perm(x0, 0, 0x4442)];
	y1 = __ldg(&c_echo_AES3[__byte_perm(x0, 0, 0x4443)]);

#ifdef INTENSIVE_GMF
	y1 ^= __ldg(&c_echo_AES0[__byte_perm(x1, 0, 0x4440)]);
#else
	y1 ^= sharedMemory[0][__byte_perm(x1, 0, 0x4440)];
#endif
	y0 ^= sharedMemory[1][__byte_perm(x1, 0, 0x4441)];
	y3 ^= sharedMemory[2][__byte_perm(x1, 0, 0x4442)];
	y2 ^= __ldg(&c_echo_AES3[__byte_perm(x1, 0, 0x4443)]);

	y2 ^= sharedMemory[0][__byte_perm(x2, 0, 0x4440)];
	y1 ^= sharedMemory[1][__byte_perm(x2, 0, 0x4441)];
	y0 ^= sharedMemory[2][__byte_perm(x2, 0, 0x4442)];
	y3 ^= __ldg(&c_echo_AES3[__byte_perm(x2, 0, 0x4443)]);

	y3 ^= sharedMemory[0][__byte_perm(x3, 0, 0x4440)];
	y2 ^= sharedMemory[1][__byte_perm(x3, 0, 0x4441)];
	y1 ^= sharedMemory[2][__byte_perm(x3, 0, 0x4442)];
	y0 ^= __ldg(&c_echo_AES3[__byte_perm(x3, 0, 0x4443)]);
}

__device__ __forceinline__
static void echo_aes_round_LDG(const uint32_t sharedMemory[4][256], const uint32_t x0, const uint32_t x1, const uint32_t x2, const uint32_t x3,
	uint32_t &y0, uint32_t &y1, uint32_t &y2, uint32_t &y3)
{
	y0 = sharedMemory[0][__byte_perm(x0, 0, 0x4440)];
	y3 = sharedMemory[1][__byte_perm(x0, 0, 0x4441)];
	y2 = sharedMemory[2][__byte_perm(x0, 0, 0x4442)];
	y1 = __ldg(&c_echo_AES3[__byte_perm(x0, 0, 0x4443)]);

	y1 ^= __ldg(&c_echo_AES0[__byte_perm(x1, 0, 0x4440)]);
	y0 ^= sharedMemory[1][__byte_perm(x1, 0, 0x4441)];
	y3 ^= sharedMemory[2][__byte_perm(x1, 0, 0x4442)];
	y2 ^= __ldg(&c_echo_AES3[__byte_perm(x1, 0, 0x4443)]);

	y2 ^= sharedMemory[0][__byte_perm(x2, 0, 0x4440)];
	y1 ^= sharedMemory[1][__byte_perm(x2, 0, 0x4441)];
	y0 ^= sharedMemory[2][__byte_perm(x2, 0, 0x4442)];
	y3 ^= __ldg(&c_echo_AES3[__byte_perm(x2, 0, 0x4443)]);

	y3 ^= sharedMemory[0][__byte_perm(x3, 0, 0x4440)];
	y2 ^= sharedMemory[1][__byte_perm(x3, 0, 0x4441)];
	y1 ^= sharedMemory[2][__byte_perm(x3, 0, 0x4442)];
	y0 ^= __ldg(&c_echo_AES3[__byte_perm(x3, 0, 0x4443)]);
}

__device__ __forceinline__
static void ECHO_AES_2ROUND(const uint32_t sharedMemory[4][256], uint32_t &x0, uint32_t &x1, uint32_t &x2, uint32_t &x3, uint32_t &k0)
{
	uint32_t y0, y1, y2, y3;

	echo_aes_round(sharedMemory, x0, x1, x2, x3, k0, y0, y1, y2, y3);

	echo_aes_round(sharedMemory, y0, y1, y2, y3, x0, x1, x2, x3);

	// hier werden wir ein carry brauchen (oder auch nicht)
	k0++;
}

__device__ __forceinline__
static void ECHO_AES_2ROUND_LDG(const uint32_t sharedMemory[4][256], uint32_t &x0, uint32_t &x1, uint32_t &x2, uint32_t &x3, uint32_t &k0)
{
	uint32_t y0, y1, y2, y3;

	echo_aes_round_LDG(sharedMemory, x0, x1, x2, x3, k0, y0, y1, y2, y3);

	echo_aes_round_LDG(sharedMemory, y0, y1, y2, y3, x0, x1, x2, x3);

	// hier werden wir ein carry brauchen (oder auch nicht)
	k0++;
}

__device__
static void echo_round_alexis(const uint32_t sharedMemory[4][256], uint32_t *W, uint32_t &k0)
{
	// Big Sub Words
	#pragma unroll 16
	for (int idx = 0; idx < 16; idx++)
		ECHO_AES_2ROUND(sharedMemory,W[(idx<<2) + 0], W[(idx<<2) + 1], W[(idx<<2) + 2], W[(idx<<2) + 3], k0);

	// Shift Rows
	#pragma unroll 4
	for (int i = 0; i < 4; i++){
		uint32_t t[4];
		/// 1, 5, 9, 13
		t[0] = W[i+ 4];
		t[1] = W[i+ 8];
		t[2] = W[i+24];
		t[3] = W[i+60];
		W[i + 4] = W[i + 20];
		W[i + 8] = W[i + 40];
		W[i +24] = W[i + 56];
		W[i +60] = W[i + 44];

		W[i +20] = W[i +36];
		W[i +40] = t[1];
		W[i +56] = t[2];
		W[i +44] = W[i +28];

		W[i +28] = W[i +12];
		W[i +12] = t[3];
		W[i +36] = W[i +52];
		W[i +52] = t[0];
	}
	// Mix Columns
	#pragma unroll 4
	for (int i = 0; i < 4; i++){ // Schleife über je 2*uint32_t
		#pragma unroll 4
		for (int idx = 0; idx < 64; idx += 16){ // Schleife über die elemnte
			uint32_t a[4];
			a[0] = W[idx + i];
			a[1] = W[idx + i + 4];
			a[2] = W[idx + i + 8];
			a[3] = W[idx + i +12];

			uint32_t ab = a[0] ^ a[1];
			uint32_t bc = a[1] ^ a[2];
			uint32_t cd = a[2] ^ a[3];

			uint32_t t, t2, t3;
			t = (ab & 0x80808080);
			t2 = (bc & 0x80808080);
			t3 = (cd & 0x80808080);

			uint32_t abx = (t  >> 7) * 27U ^ ((ab^t) << 1);
			uint32_t bcx = (t2 >> 7) * 27U ^ ((bc^t2) << 1);
			uint32_t cdx = (t3 >> 7) * 27U ^ ((cd^t3) << 1);

			W[idx + i] = bc ^ a[3] ^ abx;
			W[idx + i + 4] = a[0] ^ cd ^ bcx;
			W[idx + i + 8] = ab ^ a[3] ^ cdx;
			W[idx + i +12] = ab ^ a[2] ^ (abx ^ bcx ^ cdx);
		}
	}
}

/* ECHO-512 of a 64-byte input, in place (uint32 hash[16], d_hash word order)
 * — body of x16_echo512_gpu_hash_64. */
__device__ __forceinline__
void echo512_hash_64_alexis(const uint32_t sharedMemory[4][256], uint32_t *hash)
{
	uint32_t k0;
	uint32_t h[16];

	*(uint2x4*)&h[ 0] = *(uint2x4*)&hash[ 0];
	*(uint2x4*)&h[ 8] = *(uint2x4*)&hash[ 8];

		const uint32_t P[48] = {
			0xe7e9f5f5, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af, 0xa4213d7e, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
			//8-12
			0x01425eb8, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af, 0x65978b09, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
			//21-25
			0x2cb6b661, 0x6b23b3b3, 0xcf93a7cf, 0x9d9d3751, 0x9ac2dea3, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
			//34-38
			0x579f9f33, 0xfbfbfbfb, 0xfbfbfbfb, 0xefefd3c7, 0xdbfde1dd, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
			0x34514d9e, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af, 0xb134347e, 0xea6f7e7e, 0xbd7731bd, 0x8a8a1968,
			0x14b8a457, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af, 0x265f4382, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af
			//58-61
		};

		k0 = 520;

		#pragma unroll 4
		for (uint32_t idx = 0; idx < 16; idx += 4) {
			ECHO_AES_2ROUND(sharedMemory, h[idx + 0], h[idx + 1], h[idx + 2], h[idx + 3], k0);
		}
		k0 += 4;

		uint32_t W[64];

		#pragma unroll 4
		for (uint32_t i = 0; i < 4; i++)
		{
			uint32_t a = P[i];
			uint32_t b = P[i + 4];
			uint32_t c = h[i + 8];
			uint32_t d = P[i + 8];

			uint32_t ab = a ^ b;
			uint32_t bc = b ^ c;
			uint32_t cd = c ^ d;


			uint32_t t =  (ab & 0x80808080);
			uint32_t t2 = (bc & 0x80808080);
			uint32_t t3 = (cd & 0x80808080);

			uint32_t abx = (t  >> 7) * 27U ^ ((ab^t) << 1);
			uint32_t bcx = (t2 >> 7) * 27U ^ ((bc^t2) << 1);
			uint32_t cdx = (t3 >> 7) * 27U ^ ((cd^t3) << 1);

			W[i] = abx ^ bc ^ d;
			W[i + 4] = bcx ^ a ^ cd;
			W[i + 8] = cdx ^ ab ^ d;
			W[i +12] = abx ^ bcx ^ cdx ^ ab ^ c;

			a = P[i +12];
			b = h[i + 4];
			c = P[i +16];
			d = P[i +20];

			ab = a ^ b;
			bc = b ^ c;
			cd = c ^ d;


			t = (ab & 0x80808080);
			t2 = (bc & 0x80808080);
			t3 = (cd & 0x80808080);

			abx = (t >> 7) * 27U ^ ((ab^t) << 1);
			bcx = (t2 >> 7) * 27U ^ ((bc^t2) << 1);
			cdx = (t3 >> 7) * 27U ^ ((cd^t3) << 1);

			W[16 + i] = bc ^ d ^ abx;
			W[16 + i + 4] = a ^ cd ^ bcx;
			W[16 + i + 8] = d ^ ab ^ cdx;
			W[16 + i + 12] = c ^ ab ^ abx ^ bcx ^ cdx;

			a = h[i];
			b = P[24 + i + 0];
			c = P[24 + i + 4];
			d = P[24 + i + 8];

			ab = a ^ b;
			bc = b ^ c;
			cd = c ^ d;


			t = (ab & 0x80808080);
			t2 = (bc & 0x80808080);
			t3 = (cd & 0x80808080);

			abx = (t >> 7) * 27U ^ ((ab^t) << 1);
			bcx = (t2 >> 7) * 27U ^ ((bc^t2) << 1);
			cdx = (t3 >> 7) * 27U ^ ((cd^t3) << 1);

			W[32 + i] = bc ^ d ^ abx;
			W[32 + i + 4] = a ^ cd ^ bcx;
			W[32 + i + 8] = d ^ ab ^ cdx;
			W[32 + i + 12] = c ^ ab ^ abx ^ bcx ^ cdx;

			a = P[36 + i ];
			b = P[36 + i + 4];
			c = P[36 + i + 8];
			d = h[i + 12];

			ab = a ^ b;
			bc = b ^ c;
			cd = c ^ d;

			t = (ab & 0x80808080);
			t2 = (bc & 0x80808080);
			t3 = (cd & 0x80808080);

			abx = (t >> 7) * 27U ^ ((ab^t) << 1);
			bcx = (t2 >> 7) * 27U ^ ((bc^t2) << 1);
			cdx = (t3 >> 7) * 27U ^ ((cd^t3) << 1);

			W[48 + i] = bc ^ d ^ abx;
			W[48 + i + 4] = a ^ cd ^ bcx;
			W[48 + i + 8] = d ^ ab ^ cdx;
			W[48 + i + 12] = c ^ ab ^ abx ^ bcx ^ cdx;
		}

		for (int k = 1; k < 10; k++)
			echo_round_alexis(sharedMemory,W,k0);

		#pragma unroll 4
		for (int i = 0; i < 16; i += 4)
		{
			W[i] ^= W[32 + i] ^ 512;
			W[i + 1] ^= W[32 + i + 1];
			W[i + 2] ^= W[32 + i + 2];
			W[i + 3] ^= W[32 + i + 3];
		}
	*(uint2x4*)&hash[ 0] = *(uint2x4*)&hash[ 0] ^ *(uint2x4*)&W[ 0];
	*(uint2x4*)&hash[ 8] = *(uint2x4*)&hash[ 8] ^ *(uint2x4*)&W[ 8];
}

#endif /* CUDA_ECHO512_DEVICE_CUH */
