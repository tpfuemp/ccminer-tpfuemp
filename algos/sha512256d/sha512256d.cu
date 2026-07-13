// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * sha512256d host scanhash — sha256d-shaped skeleton, generic stratum path
 * (32-bit nonce at pdata[19], big-endian in the header). CPU reference is
 * the sph_sha512 core seeded with the SHA-512/256 IV, exactly like the
 * cpuminer-opt scalar scanhash_sha512256d; every GPU candidate is
 * revalidated here before submit.
 */

#include <miner.h>
#include <cuda_helper.h>

#include "sph/sph_sha2.h"

static bool init[MAX_GPUS] = { 0 };
extern void sha512256d_init(int thr_id);
extern void sha512256d_free(int thr_id);
extern void sha512256d_setBlock_80(const uint32_t *pdata);
extern void sha512256d_hash_80(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t targ_q3, uint32_t *resNonces);

static const uint64_t H512_256[8] = {
	0x22312194FC2BF72CULL, 0x9F555FA3C84C64C2ULL,
	0x2393B86B6F53B151ULL, 0x963877195940EABDULL,
	0x96283EE2A88EFFE3ULL, 0xBE5E1E2553863992ULL,
	0x2B0199FC2C85B8AAULL, 0x0EB72DDC81C52CA2ULL,
};

// CPU check: double SHA-512/256 of an 80-byte big-endian header, first
// 32 bytes of each close are the truncated digest (fulltest word order).
extern "C" void sha512256d_hash(void *output, const void *input)
{
	uint64_t _ALIGN(64) hash[8];
	sph_sha512_context ctx;

	memcpy(ctx.val, H512_256, sizeof(H512_256));
	ctx.count = 0;
	sph_sha512(&ctx, input, 80);
	sph_sha512_close(&ctx, hash);

	memcpy(ctx.val, H512_256, sizeof(H512_256));
	ctx.count = 0;
	sph_sha512(&ctx, hash, 32);
	sph_sha512_close(&ctx, hash);

	memcpy(output, hash, 32);
}

extern "C" int scanhash_sha512256d(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << 24);
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

		sha512256d_init(thr_id);

		init[thr_id] = true;
	}

	uint32_t _ALIGN(64) endiandata[20];
	for (int k = 0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	sha512256d_setBlock_80(pdata);
	const uint64_t targ_q3 = ((uint64_t*)ptarget)[3];

	do {
		// Hash with CUDA (GPU screens on the target's high qword; the host
		// fulltest below is authoritative)
		*hashes_done = pdata[19] - first_nonce + throughput;

		sha512256d_hash_80(thr_id, throughput, pdata[19], targ_q3, work->nonces);
		if (work->nonces[0] != UINT32_MAX)
		{
			uint32_t _ALIGN(64) vhash[8];

			be32enc(&endiandata[19], work->nonces[0]);
			sha512256d_hash(vhash, endiandata);
			if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
				// positive proof for kernel A/Bs: benchmark shares are otherwise
				// invisible (no submit, and the API ACC counter never ticks)
				if (opt_benchmark && opt_debug)
					gpulog(LOG_BLUE, thr_id, "benchmark candidate %08x validated on CPU", work->nonces[0]);
				if (work->nonces[1] != UINT32_MAX) {
					be32enc(&endiandata[19], work->nonces[1]);
					sha512256d_hash(vhash, endiandata);
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
extern "C" void free_sha512256d(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	sha512256d_free(thr_id);

	init[thr_id] = false;

	cudaDeviceSynchronize();
}
