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
	uint32_t max_nonce);
extern "C" bool balloon_selftest(int thr_id);

int scanhash_balloon(int thr_id, struct work *work, uint32_t max_nonce,
	unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;

	uint32_t _ALIGN(128) endiandata[20];
	uint32_t _ALIGN(64) vhash[8];

	// benchmark-only: 0x0000ff still yields ~0 candidates at a kH/s rate
	if (opt_benchmark)
		ptarget[7] = 0x00ffff;

	const uint32_t Htarg = ptarget[7];
	const uint32_t first_nonce = pdata[19];
	uint32_t n = first_nonce;

	// Nonces per launch, one GPU thread each; the host cursor advances by the same
	// amount. Runtime-tunable with -i / --intensity, default 1<<14. Rounded down to
	// a multiple of 64 (the block size) so the grid covers the batch exactly.
	//
	// Do not raise this casually. The card is already saturated at the default, so a
	// larger batch adds no throughput, only launch duration: the kernel is not
	// chunked, so one launch runs batch/hashrate seconds (~0.43 s at -i 14 on an RTX
	// 3060, ~1.7 s at -i 16, and ~18% longer again once the card throttles). Past
	// ~2 s Windows resets a display-attached GPU; and since a launch cannot be
	// interrupted, that same figure is how long the miner keeps hashing a job the
	// pool has already replaced. The first launch is timed below and reported.
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
		// Consensus gate; fails closed inside balloon_selftest().
		balloon_selftest(thr_id);
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

	// See the batch-size note above.
	static THREAD bool launch_timed = false;

	do {
		be32enc(&endiandata[19], n);

		struct timeval tv_a, tv_b, tv_d;
		const bool time_it = (!launch_timed || opt_debug);
		if (time_it) gettimeofday(&tv_a, NULL);

		uint32_t winning_nonce = balloon_cpu_hash(thr_id, (unsigned char *)endiandata,
			batch, max_nonce);

		if (time_it) {
			gettimeofday(&tv_b, NULL);
			timeval_subtract(&tv_d, &tv_b, &tv_a);
			const double secs = (double)tv_d.tv_sec + (double)tv_d.tv_usec / 1e6;
			const bool first = !launch_timed;
			launch_timed = true;
			if (first && secs > 1.5) {
				gpulog(LOG_WARNING, thr_id,
					"a single launch takes %.2fs at intensity %.2f; it cannot be interrupted, "
					"so that is both the driver-reset window and the job-switch delay. "
					"Lower -i -- throughput does not improve above the default.",
					secs, throughput2intensity(batch));
			} else if (opt_debug) {
				gpulog(LOG_DEBUG, thr_id, "launch %.3fs at intensity %.2f (%u nonces)",
					secs, throughput2intensity(batch), batch);
			}
		}

		if (work_restart[thr_id].restart)
			break;

		// CPU re-hash is authoritative: a kernel fault costs a local reject, not a bad share
		if (winning_nonce != UINT32_MAX) {
			be32enc(&endiandata[19], winning_nonce);
			balloon_128_orig((unsigned char *)endiandata, (unsigned char *)vhash);

			if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
				work->nonces[0] = winning_nonce;
				work_set_target_ratio(work, vhash);
				work->valid_nonces = 1;
				*hashes_done = winning_nonce - first_nonce + 1;
				// Resume PAST the accepted nonce: without the +1 a re-entry on the
				// same work re-finds and re-submits it.
				pdata[19] = winning_nonce + 1;
				return 1;
			}
			else if (vhash[7] > Htarg) {
				gpu_increment_reject(thr_id);
				if (!opt_quiet) gpulog(LOG_WARNING, thr_id,
					"result for %08x does not validate on CPU!", winning_nonce);
			}
		}

		n += batch;
	} while (n < max_nonce && !work_restart[thr_id].restart);

	*hashes_done = n - first_nonce;
	pdata[19] = n;

	return 0;
}
