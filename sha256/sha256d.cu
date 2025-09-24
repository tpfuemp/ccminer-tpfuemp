/**
 * SHA256d
 * by tpruvot@github - 2017
 */

#include <miner.h>
#include <cuda_helper.h>
#include <openssl/sha.h>


static bool init[MAX_GPUS] = { 0 };
extern void sha256d_init(int thr_id);
extern void sha256d_free(int thr_id);
extern void sha256d_hash_80(int thr_id, uint32_t threads, uint32_t startNonce, const uint32_t* const ms, uint32_t merkle, uint32_t time, uint32_t compacttarget, uint32_t* resNonces);

extern void sha256d_midstate(const uint32_t* data, uint32_t* midstate);

uint32_t rrot(uint32_t x, unsigned int n)
{
	return (x >> n) | (x << (32 - n));
}

// CPU Check
extern "C" void sha256d_hash(void* output, const void* input)
{
	unsigned char _ALIGN(64) hash[64];
	SHA256_CTX sha256;

	SHA256_Init(&sha256);
	SHA256_Update(&sha256, (unsigned char*)input, 80);
	SHA256_Final(hash, &sha256);

	SHA256_Init(&sha256);
	SHA256_Update(&sha256, hash, 32);
	SHA256_Final((unsigned char*)output, &sha256);
}

void sha256d_opt_hash(uint32_t* output, const uint32_t* data, uint32_t nonce, const uint32_t* midstate)
{
	int i;
	uint32_t s0, s1, t1, t2, maj, ch, a, b, c, d, e, f, g, h;
	uint32_t w[64];

	const uint32_t k[64] = {
		0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
		0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
		0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
		0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
		0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
		0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
		0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
		0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
	};
	const uint32_t hc[8] = {
		0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
		0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U
	};

	for (i = 0; i <= 15; ++i)
	{
		w[i] = data[i + 16];
	}
	w[3] = nonce;
	for (i = 16; i <= 63; ++i)
	{
		s0 = rrot(w[i - 15], 7) ^ rrot(w[i - 15], 18) ^ (w[i - 15] >> 3);
		s1 = rrot(w[i - 2], 17) ^ rrot(w[i - 2], 19) ^ (w[i - 2] >> 10);
		w[i] = w[i - 16] + s0 + w[i - 7] + s1;
	}
	a = midstate[0];
	b = midstate[1];
	c = midstate[2];
	d = midstate[3];
	e = midstate[4];
	f = midstate[5];
	g = midstate[6];
	h = midstate[7];
	for (i = 0; i <= 63; ++i)
	{
		s0 = rrot(a, 2) ^ rrot(a, 13) ^ rrot(a, 22);
		maj = (a & b) ^ (a & c) ^ (b & c);
		t2 = s0 + maj;
		s1 = rrot(e, 6) ^ rrot(e, 11) ^ rrot(e, 25);
		ch = (e & f) ^ ((~e) & g);
		t1 = h + s1 + ch + k[i] + w[i];
		h = g;
		g = f;
		f = e;
		e = d + t1;
		d = c;
		c = b;
		b = a;
		a = t1 + t2;
	}
	w[0] = a + midstate[0];
	w[1] = b + midstate[1];
	w[2] = c + midstate[2];
	w[3] = d + midstate[3];
	w[4] = e + midstate[4];
	w[5] = f + midstate[5];
	w[6] = g + midstate[6];
	w[7] = h + midstate[7];
	w[8] = 0x80000000U;
	for (i = 9; i <= 14; ++i)
		w[i] = 0U;
	w[15] = 0x100U;
	for (i = 16; i <= 63; ++i)
	{
		s0 = rrot(w[i - 15], 7) ^ rrot(w[i - 15], 18) ^ (w[i - 15] >> 3);
		s1 = rrot(w[i - 2], 17) ^ rrot(w[i - 2], 19) ^ (w[i - 2] >> 10);
		w[i] = w[i - 16] + s0 + w[i - 7] + s1;
	}
	a = hc[0];
	b = hc[1];
	c = hc[2];
	d = hc[3];
	e = hc[4];
	f = hc[5];
	g = hc[6];
	h = hc[7];
	for (i = 0; i <= 63; ++i)
	{
		s0 = rrot(a, 2) ^ rrot(a, 13) ^ rrot(a, 22);
		maj = (a & b) ^ (a & c) ^ (b & c);
		t2 = s0 + maj;
		s1 = rrot(e, 6) ^ rrot(e, 11) ^ rrot(e, 25);
		ch = (e & f) ^ ((~e) & g);
		t1 = h + s1 + ch + k[i] + w[i];
		h = g;
		g = f;
		f = e;
		e = d + t1;
		d = c;
		c = b;
		b = a;
		a = t1 + t2;
	}
	be32enc(&output[0], a + hc[0]);
	be32enc(&output[1], b + hc[1]);
	be32enc(&output[2], c + hc[2]);
	be32enc(&output[3], d + hc[3]);
	be32enc(&output[4], e + hc[4]);
	be32enc(&output[5], f + hc[5]);
	be32enc(&output[6], g + hc[6]);
	be32enc(&output[7], h + hc[7]);
}

extern "C" int scanhash_sha256d(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << 25);
	if (init[thr_id]) throughput = min(throughput, (max_nonce - first_nonce));

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x03;

	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		sha256d_init(thr_id);

		init[thr_id] = true;
	}

	uint32_t ms[8];
	sha256d_midstate(pdata, ms);

	do {
		// Hash with CUDA
		*hashes_done = pdata[19] - first_nonce + throughput;

		sha256d_hash_80(thr_id, throughput, pdata[19], ms, pdata[16], pdata[17], pdata[18], work->nonces);
		if (work->nonces[0] != UINT32_MAX)
		{
			uint32_t _ALIGN(64) vhash[8];

			sha256d_opt_hash(vhash, pdata, work->nonces[0], ms);
			if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
				if (work->nonces[1] != UINT32_MAX) {
					sha256d_opt_hash(vhash, pdata, work->nonces[1], ms);
					if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
						work->valid_nonces++;
						bn_set_target_ratio(work, vhash, 1);
					}
					pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
				} else {
					pdata[19] = work->nonces[0] + 1;
				}
				return work->valid_nonces;
			}
			else if (vhash[7] > ptarget[7]) {
				gpu_increment_reject(thr_id);
				if (!opt_quiet) {
					gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
				}
				pdata[19] = work->nonces[0] + 1;
				continue;
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
extern "C" void free_sha256d(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	sha256d_free(thr_id);

	init[thr_id] = false;

	cudaDeviceSynchronize();
}
