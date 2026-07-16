/**
 * Blake2-B CUDA Implementation
 *
 * tpruvot@github July 2016
 *
 */

#include <miner.h>

#include <string.h>
#include <stdint.h>

#include <sph/blake2b.h>

#include <cuda_helper.h>
#include <cuda_vector_uint2x4.h>

#define TPB 512
#define NBN 2

static uint32_t *d_resNonces[MAX_GPUS];

__device__ uint64_t d_data[10];

static __constant__ const int8_t blake2b_sigma[12][16] = {
	{ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15 } ,
	{ 14, 10, 4,  8,  9,  15, 13, 6,  1,  12, 0,  2,  11, 7,  5,  3  } ,
	{ 11, 8,  12, 0,  5,  2,  15, 13, 10, 14, 3,  6,  7,  1,  9,  4  } ,
	{ 7,  9,  3,  1,  13, 12, 11, 14, 2,  6,  5,  10, 4,  0,  15, 8  } ,
	{ 9,  0,  5,  7,  2,  4,  10, 15, 14, 1,  11, 12, 6,  8,  3,  13 } ,
	{ 2,  12, 6,  10, 0,  11, 8,  3,  4,  13, 7,  5,  15, 14, 1,  9  } ,
	{ 12, 5,  1,  15, 14, 13, 4,  10, 0,  7,  6,  3,  9,  2,  8,  11 } ,
	{ 13, 11, 7,  14, 12, 1,  3,  9,  5,  0,  15, 4,  8,  6,  2,  10 } ,
	{ 6,  15, 14, 9,  11, 3,  0,  8,  12, 2,  13, 7,  1,  4,  10, 5  } ,
	{ 10, 2,  8,  4,  7,  6,  1,  5,  15, 11, 9,  14, 3,  12, 13, 0  } ,
	{ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15 } ,
	{ 14, 10, 4,  8,  9,  15, 13, 6,  1,  12, 0,  2,  11, 7,  5,  3  }
};

// host mem align
#define A 64

//sph/blake2b.h is not linked properly for VC++
#ifdef _WIN32
#ifndef ROTR64
#define ROTR64(x, y)  (((x) >> (y)) ^ ((x) << (64 - (y))))
#endif

// Little-endian byte access.

#define B2B_GET64(p)                            \
	(((uint64_t) ((uint8_t *) (p))[0]) ^        \
	(((uint64_t) ((uint8_t *) (p))[1]) << 8) ^  \
	(((uint64_t) ((uint8_t *) (p))[2]) << 16) ^ \
	(((uint64_t) ((uint8_t *) (p))[3]) << 24) ^ \
	(((uint64_t) ((uint8_t *) (p))[4]) << 32) ^ \
	(((uint64_t) ((uint8_t *) (p))[5]) << 40) ^ \
	(((uint64_t) ((uint8_t *) (p))[6]) << 48) ^ \
	(((uint64_t) ((uint8_t *) (p))[7]) << 56))

// G Mixing function.

#define B2B_G(a, b, c, d, x, y) {   \
	v[a] = v[a] + v[b] + x;         \
	v[d] = ROTR64(v[d] ^ v[a], 32); \
	v[c] = v[c] + v[d];             \
	v[b] = ROTR64(v[b] ^ v[c], 24); \
	v[a] = v[a] + v[b] + y;         \
	v[d] = ROTR64(v[d] ^ v[a], 16); \
	v[c] = v[c] + v[d];             \
	v[b] = ROTR64(v[b] ^ v[c], 63); }

// Initialization Vector.

static const uint64_t blake2b_iv[8] = {
	0x6A09E667F3BCC908, 0xBB67AE8584CAA73B,
	0x3C6EF372FE94F82B, 0xA54FF53A5F1D36F1,
	0x510E527FADE682D1, 0x9B05688C2B3E6C1F,
	0x1F83D9ABFB41BD6B, 0x5BE0CD19137E2179
};

// Compression function. "last" flag indicates last block.

static void blake2b_compress(blake2b_ctx* ctx, int last)
{
	const uint8_t sigma[12][16] = {
		{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
		{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
		{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
		{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
		{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
		{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
		{ 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
		{ 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
		{ 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
		{ 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
		{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
		{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 }
	};
	int i;
	uint64_t v[16], m[16];

	for (i = 0; i < 8; i++) {           // init work variables
		v[i] = ctx->h[i];
		v[i + 8] = blake2b_iv[i];
	}

	v[12] ^= ctx->t[0];                 // low 64 bits of offset
	v[13] ^= ctx->t[1];                 // high 64 bits
	if (last)                           // last block flag set ?
		v[14] = ~v[14];

	for (i = 0; i < 16; i++)            // get little-endian words
		m[i] = B2B_GET64(&ctx->b[8 * i]);

	for (i = 0; i < 12; i++) {          // twelve rounds
		B2B_G(0, 4, 8, 12, m[sigma[i][0]], m[sigma[i][1]]);
		B2B_G(1, 5, 9, 13, m[sigma[i][2]], m[sigma[i][3]]);
		B2B_G(2, 6, 10, 14, m[sigma[i][4]], m[sigma[i][5]]);
		B2B_G(3, 7, 11, 15, m[sigma[i][6]], m[sigma[i][7]]);
		B2B_G(0, 5, 10, 15, m[sigma[i][8]], m[sigma[i][9]]);
		B2B_G(1, 6, 11, 12, m[sigma[i][10]], m[sigma[i][11]]);
		B2B_G(2, 7, 8, 13, m[sigma[i][12]], m[sigma[i][13]]);
		B2B_G(3, 4, 9, 14, m[sigma[i][14]], m[sigma[i][15]]);
	}

	for (i = 0; i < 8; ++i)
		ctx->h[i] ^= v[i] ^ v[i + 8];
}
int blake2b_init(blake2b_ctx* ctx, size_t outlen,
	const void* key, size_t keylen)        // (keylen=0: no key)
{
	size_t i;

	if (outlen == 0 || outlen > 64 || keylen > 64)
		return -1;                      // illegal parameters

	for (i = 0; i < 8; i++)             // state, "param block"
		ctx->h[i] = blake2b_iv[i];
	ctx->h[0] ^= 0x01010000 ^ (keylen << 8) ^ outlen;

	ctx->t[0] = 0;                      // input count low word
	ctx->t[1] = 0;                      // input count high word
	ctx->c = 0;                         // pointer within buffer
	ctx->outlen = outlen;

	for (i = keylen; i < 128; i++)      // zero input block
		ctx->b[i] = 0;
	if (keylen > 0) {
		blake2b_update(ctx, key, keylen);
		ctx->c = 128;                   // at the end
	}

	return 0;
}

// Add "inlen" bytes from "in" into the hash.

void blake2b_update(blake2b_ctx* ctx,
	const void* in, size_t inlen)       // data bytes
{
	size_t i;

	for (i = 0; i < inlen; i++) {
		if (ctx->c == 128) {            // buffer full ?
			ctx->t[0] += ctx->c;        // add counters
			if (ctx->t[0] < ctx->c)     // carry overflow ?
				ctx->t[1]++;            // high word
			blake2b_compress(ctx, 0);   // compress (not last)
			ctx->c = 0;                 // counter to zero
		}
		ctx->b[ctx->c++] = ((const uint8_t*)in)[i];
	}
}

// Generate the message digest (size given in init).
//      Result placed in "out".

void blake2b_final(blake2b_ctx* ctx, void* out)
{
	size_t i;

	ctx->t[0] += ctx->c;                // mark last block offset
	if (ctx->t[0] < ctx->c)             // carry overflow
		ctx->t[1]++;                    // high word

	while (ctx->c < 128)                // fill up with zeros
		ctx->b[ctx->c++] = 0;
	blake2b_compress(ctx, 1);           // final block flag = 1

	// little endian convert and store
	for (i = 0; i < ctx->outlen; i++) {
		((uint8_t*)out)[i] =
			(ctx->h[i >> 3] >> (8 * (i & 7))) & 0xFF;
	}
}
#endif

extern "C" void blake2b_hash(void *output, const void *input)
{
	uint8_t _ALIGN(A) hash[32];
	blake2b_ctx ctx;

	blake2b_init(&ctx, 32, NULL, 0);
	blake2b_update(&ctx, input, 80);
	blake2b_final(&ctx, hash);

	memcpy(output, hash, 32);
}

// ----------------------------------------------------------------

__device__ __forceinline__
static void G(const int r, const int i, uint64_t &a, uint64_t &b, uint64_t &c, uint64_t &d, uint64_t const m[16])
{
	a = a + b + m[ blake2b_sigma[r][2*i] ];
	((uint2*)&d)[0] = SWAPUINT2( ((uint2*)&d)[0] ^ ((uint2*)&a)[0] );
	c = c + d;
	((uint2*)&b)[0] = ROR24( ((uint2*)&b)[0] ^ ((uint2*)&c)[0] );
	a = a + b + m[ blake2b_sigma[r][2*i+1] ];
	((uint2*)&d)[0] = ROR16( ((uint2*)&d)[0] ^ ((uint2*)&a)[0] );
	c = c + d;
	((uint2*)&b)[0] = ROR2( ((uint2*)&b)[0] ^ ((uint2*)&c)[0], 63U);
}

#define ROUND(r) \
	G(r, 0, v[0], v[4], v[ 8], v[12], m); \
	G(r, 1, v[1], v[5], v[ 9], v[13], m); \
	G(r, 2, v[2], v[6], v[10], v[14], m); \
	G(r, 3, v[3], v[7], v[11], v[15], m); \
	G(r, 4, v[0], v[5], v[10], v[15], m); \
	G(r, 5, v[1], v[6], v[11], v[12], m); \
	G(r, 6, v[2], v[7], v[ 8], v[13], m); \
	G(r, 7, v[3], v[4], v[ 9], v[14], m);

__global__
//__launch_bounds__(128, 8) /* to force 64 regs */
void blake2b_gpu_hash(const uint32_t threads, const uint32_t startNonce, uint32_t *resNonce, const uint2 target2)
{
	const uint32_t nonce = (blockDim.x * blockIdx.x + threadIdx.x) + startNonce;

	uint64_t m[16];

	m[0] = d_data[0];
	m[1] = d_data[1];
	m[2] = d_data[2];
	m[3] = d_data[3];
	m[4] = d_data[4];
	m[5] = d_data[5];
	m[6] = d_data[6];
	m[7] = d_data[7];
	m[8] = d_data[8];
	((uint32_t*)m)[18] = AS_U32(&d_data[9]);
	((uint32_t*)m)[19] = nonce;

	m[10] = m[11] = 0;
	m[12] = m[13] = 0;
	m[14] = m[15] = 0;

	uint64_t v[16] = {
		0x6a09e667f2bdc928, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
		0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
		0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
		0x510e527fade68281, 0x9b05688c2b3e6c1f, 0xe07c265404be4294, 0x5be0cd19137e2179
	};

	ROUND( 0);
	ROUND( 1);
	ROUND( 2);
	ROUND( 3);
	ROUND( 4);
	ROUND( 5);
	ROUND( 6);
	ROUND( 7);
	ROUND( 8);
	ROUND( 9);
	ROUND(10);
	ROUND(11);

	uint2 last = vectorize(v[3] ^ v[11] ^ 0xa54ff53a5f1d36f1);
	if (last.y <= target2.y && last.x <= target2.x) {
		resNonce[1] = resNonce[0];
		resNonce[0] = nonce;
	}
}

__host__
uint32_t blake2b_hash_cuda(const int thr_id, const uint32_t threads, const uint32_t startNonce, const uint2 target2, uint32_t &secNonce)
{
	uint32_t resNonces[NBN] = { UINT32_MAX, UINT32_MAX };
	uint32_t result = UINT32_MAX;

	dim3 grid((threads + TPB-1)/TPB);
	dim3 block(TPB);

	/* Check error on Ctrl+C or kill to prevent segfaults on exit */
	if (cudaMemset(d_resNonces[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
		return result;

	blake2b_gpu_hash <<<grid, block, 8>>> (threads, startNonce, d_resNonces[thr_id], target2);
	cudaDeviceSynchronize();

	if (cudaSuccess == cudaMemcpy(resNonces, d_resNonces[thr_id], NBN*sizeof(uint32_t), cudaMemcpyDeviceToHost)) {
		result = resNonces[0];
		secNonce = resNonces[1];
		if (secNonce == result) secNonce = UINT32_MAX;
	}
	return result;
}

__host__
void blake2b_setBlock(uint32_t *data)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_data, data, 80, 0, cudaMemcpyHostToDevice));
}

static bool init[MAX_GPUS] = { 0 };

int scanhash_blake2b(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(A) endiandata[20];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;

	const uint32_t first_nonce = pdata[19];

	int dev_id = device_map[thr_id];
	int intensity = (device_sm[dev_id] >= 500 && !is_windows()) ? 28 : 25;
	if (device_sm[dev_id] >= 520 && is_windows()) intensity = 26;
	if (device_sm[dev_id] < 350) intensity = 22;

	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (!init[thr_id])
	{
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage (linux)
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		CUDA_CALL_OR_RET_X(cudaMalloc(&d_resNonces[thr_id], NBN * sizeof(uint32_t)), -1);
		init[thr_id] = true;
	}

	for (int i=0; i < 20; i++)
		be32enc(&endiandata[i], pdata[i]);

	const uint2 target = make_uint2(ptarget[6], ptarget[7]);
	blake2b_setBlock(endiandata);

	do {
		work->nonces[0] = blake2b_hash_cuda(thr_id, throughput, pdata[19], target, work->nonces[1]);

		*hashes_done = pdata[19] - first_nonce + throughput;

		if (work->nonces[0] != UINT32_MAX)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t _ALIGN(A) vhash[8];
			work->valid_nonces = 0;
			endiandata[19] = work->nonces[0];
			blake2b_hash(vhash, endiandata);
			if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
				work_set_target_ratio(work, vhash);
				work->valid_nonces++;
				pdata[19] = work->nonces[0] + 1;
			} else {
				gpu_increment_reject(thr_id);
			}

			if (work->nonces[1] != UINT32_MAX) {
				endiandata[19] = work->nonces[1];
				blake2b_hash(vhash, endiandata);
				if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
					if (bn_hash_target_ratio(vhash, ptarget) > work->shareratio[0]) {
						work->sharediff[1] = work->sharediff[0];
						work->shareratio[1] = work->shareratio[0];
						xchg(work->nonces[1], work->nonces[0]);
						work_set_target_ratio(work, vhash);
					} else {
						bn_set_target_ratio(work, vhash, 1);
					}
					work->valid_nonces++;
					pdata[19] = max(work->nonces[0], work->nonces[1]) + 1; // next scan start
				} else {
					gpu_increment_reject(thr_id);
				}
			}

			if (work->valid_nonces) {
				work->nonces[0] = cuda_swab32(work->nonces[0]);
				work->nonces[1] = cuda_swab32(work->nonces[1]);
				return work->valid_nonces;
			}
		}

		if ((uint64_t) throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}

		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce;

	return 0;
}

// cleanup
extern "C" void free_blake2b(int thr_id)
{
	if (!init[thr_id])
		return;

	//cudaDeviceSynchronize();

	cudaFree(d_resNonces[thr_id]);

	init[thr_id] = false;

	cudaDeviceSynchronize();
}
