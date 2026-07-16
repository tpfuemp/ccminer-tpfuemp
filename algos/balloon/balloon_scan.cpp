/*
* balloon algorithm - CUDA scan driver
*
*/
#include "miner.h"
#include <string.h>
#include <stdint.h>

#include <openssl/sha.h>

#include "balloon.h"
#include "cuda_helper.h"

// GPU entry points implemented in balloon/cuda_balloon.cu
extern void balloon_gpu_init(int thr_id);
extern void balloon_setBlock_80(int thr_id, void *pdata, const void *ptarget);
uint32_t balloon_cpu_hash(int thr_id, unsigned char *input, uint32_t threads,
	uint32_t startNounce, uint32_t *h_nounce, uint32_t max_nonce);

int scanhash_balloon(int thr_id, struct work *work, uint32_t max_nonce,
	unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;

	uint32_t _ALIGN(128) endiandata[20];
	uint32_t _ALIGN(64) vhash[8];
	uint32_t h_nounce[2] = { 0, 0 };

	const uint32_t Htarg = ptarget[7];
	const uint32_t first_nonce = pdata[19];
	uint32_t n = first_nonce;

	// 'batch' is the number of nonces scanned per kernel launch (one GPU thread
	// per nonce), so the host cursor advances by the same amount. This is the
	// main throughput tunable: larger batches keep the GPU saturated (balloon is
	// memory-latency bound, so more in-flight threads hide stalls) at the cost of
	// ~128 KB of device memory per resident nonce. Tunable at runtime via
	// -i / --intensity (a power of two); default 1<<14 = 16384. Rounded down to a
	// multiple of 64 (the kernel launches 64 threads/block).
	static THREAD volatile bool init = false;
	static THREAD uint32_t batch = 0;
	if (!init)
	{
		CUDA_SAFE_CALL(cudaSetDevice(device_map[thr_id]));
		if (opt_cudaschedule == -1) {
			cudaDeviceReset();
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		}
		balloon_gpu_init(thr_id);
		batch = cuda_default_throughput(thr_id, 1U << 14) & ~0x3fU;
		if (batch < 64) batch = 64;
		gpulog(LOG_INFO, thr_id, "intensity %.2f, %u nonces/launch",
			throughput2intensity(batch), batch);
		init = true;
	}

	for (int i = 0; i < 19; i++) {
		be32enc(&endiandata[i], pdata[i]);
	}

	// The pre-buffer depends only on the first SALT_LEN header bytes and is
	// nonce-independent, so it is refreshed once per work unit and then cached
	// on the GPU across the nonce batches below.
	reset_host_prebuf(thr_id);
	balloon_reset();

	// Upload the target (and padded header) to device constant memory.
	balloon_setBlock_80(thr_id, endiandata, ptarget);

	do {
		be32enc(&endiandata[19], n);

		uint32_t winning_nonce = balloon_cpu_hash(thr_id, (unsigned char *)endiandata,
			batch, n, h_nounce, max_nonce);

		if (work_restart[thr_id].restart)
			break;

		// Re-hash the GPU candidate on the CPU; this is authoritative and avoids
		// submitting GPU false positives. When the batch found nothing the GPU
		// returns its last nonce, which fails fulltest() here and we continue.
		be32enc(&endiandata[19], winning_nonce);
		balloon_128_orig((unsigned char *)endiandata, (unsigned char *)vhash);

		if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
			work->nonces[0] = winning_nonce;
			work_set_target_ratio(work, vhash);
			work->valid_nonces = 1;
			*hashes_done = winning_nonce - first_nonce + 1;
			pdata[19] = winning_nonce;
			return 1;
		}

		n += batch;
	} while (n < max_nonce && !work_restart[thr_id].restart);

	*hashes_done = n - first_nonce;
	pdata[19] = n;

	return 0;
}
