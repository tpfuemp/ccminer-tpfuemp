// SPDX-License-Identifier: GPL-3.0-or-later
//
// Verthash (Vertcoin VTC) ccminer bridge: scanhash_verthash + init/free.
//
// Standard Bitcoin-fork mining path: 80-byte header, 32-bit nonce at data[19],
// generic stratum/submit. The only new capability is a persistent ~1.19 GiB
// verthash.dat image resident in VRAM (loaded once per device from a single
// shared host copy). Every GPU candidate is re-verified on the host with the
// CPU oracle (verthash_hash_oracle) before submit, so a kernel bug can only ever
// cost a local reject, never a bad share.
//
// Provenance & design: algos/verthash/README.md; device kernels in
// cuda_verthash.cu; datafile management in verthash-data.cpp; CPU oracle in
// verthash-cpu.c. Ported from VerthashMiner (GPLv2) + cpuminer-opt (GPLv2).

#include <cuda_runtime.h>
#include <stdint.h>
#include <unistd.h>   // sleep() on Linux (Windows shim: compat/unistd.h + compat.h)

extern "C" {
#include "miner.h"
}
#include "cuda_helper.h"

extern "C" {
#include "algos/verthash/verthash-data.h"
#include "algos/verthash/verthash-cpu.h"
}

// cuda_verthash.cu launchers
extern "C" void verthash_cuda_set_header(const uint32_t header19[19]);
extern "C" void verthash_cuda_set_mdiv(uint32_t mdiv);
extern "C" void verthash_cuda_precompute(uint2 *d_kstates);
extern "C" void verthash_cuda_hash(uint2 *d_iohashes, const uint2 *d_kstates, const uint2 *d_memory,
                                   uint32_t in18, uint32_t firstNonce, uint32_t nonces,
                                   uint32_t *d_results, uint32_t target);

// opt_verthash_data (--verthash-data <path>) comes from miner.h; default
// "verthash.dat" in the cwd.

// -------- shared host datafile (loaded once for all GPUs) --------
static uint8_t  *s_dat      = NULL;
static size_t    s_dat_size = 0;
static uint32_t  s_mdiv     = 0;
static pthread_mutex_t s_dat_lock = PTHREAD_MUTEX_INITIALIZER;

// -------- per-device state --------
static bool       s_init[MAX_GPUS]       = { 0 };
static bool       s_selftested[MAX_GPUS] = { 0 };
static uint2     *d_memory[MAX_GPUS]     = { 0 };
static uint2     *d_iohashes[MAX_GPUS]   = { 0 };
static uint2     *d_kstates[MAX_GPUS]    = { 0 };
static uint32_t  *d_results[MAX_GPUS]    = { 0 };
static uint32_t   s_throughput[MAX_GPUS] = { 0 };

// Load + verify the host datafile exactly once, shared across GPU threads.
static bool ensure_host_datafile(int thr_id)
{
	bool ok = true;
	pthread_mutex_lock(&s_dat_lock);
	if (!s_dat) {
		const char *path = opt_verthash_data ? opt_verthash_data : "verthash.dat";
		gpulog(LOG_INFO, thr_id, "Verthash: loading data file '%s' (~1.2 GiB)...", path);
		if (verthash_data_load(path, &s_dat, &s_dat_size) != 0) {
			gpulog(LOG_ERR, thr_id, "Verthash: failed to load data file '%s'", path);
			gpulog(LOG_NOTICE, thr_id, "Verthash: set --verthash-data <path> to your verthash.dat");
			ok = false;
		} else {
			s_mdiv = verthash_data_mdiv(s_dat_size);
			if (verthash_data_verify(s_dat, s_dat_size))
				gpulog(LOG_INFO, thr_id, "Verthash: data file verified (%zu bytes, mdiv %u)", s_dat_size, s_mdiv);
			else
				gpulog(LOG_WARNING, thr_id, "Verthash: data file digest MISMATCH -- non-canonical file? (mdiv %u)", s_mdiv);
		}
	}
	pthread_mutex_unlock(&s_dat_lock);
	return ok;
}

// GPU vs CPU-oracle self-test on a fixed header (fail-closed) plus a negative
// test (corrupt input must diverge -- proves the check is not vacuous).
static bool verthash_selftest(int thr_id)
{
	const uint32_t N = 256;
	uint8_t header[80];
	for (int i = 0; i < 80; i++) header[i] = (uint8_t)(i * 7 + 1);
	uint32_t *hw = (uint32_t *) header;

	verthash_cuda_set_header(hw);
	verthash_cuda_set_mdiv(s_mdiv);
	verthash_cuda_precompute(d_kstates[thr_id]);
	cudaMemset(d_results[thr_id], 0, sizeof(uint32_t));
	verthash_cuda_hash(d_iohashes[thr_id], d_kstates[thr_id], d_memory[thr_id],
	                   hw[18], 0, N, d_results[thr_id], 0xffffffffu);
	if (cudaDeviceSynchronize() != cudaSuccess) return false;

	uint32_t *gpu = (uint32_t *) malloc(N * 32);
	if (!gpu) return false;
	cudaMemcpy(gpu, d_iohashes[thr_id], N * 32, cudaMemcpyDeviceToHost);

	int mism = 0;
	for (uint32_t n = 0; n < N; n++) {
		uint8_t ref[32];
		hw[19] = n;
		verthash_hash_oracle(s_dat, s_dat_size, header, ref);
		if (memcmp(ref, gpu + n * 8, 32) != 0) mism++;
	}

	// Negative control: a bit-flipped reference for nonce 0 must NOT match the
	// GPU hash -- proves the comparison above is not vacuous.
	uint8_t ref0[32];
	hw[19] = 0;
	verthash_hash_oracle(s_dat, s_dat_size, header, ref0);
	ref0[0] ^= 0x01;
	bool negative_ok = (memcmp(ref0, gpu, 32) != 0);
	free(gpu);

	if (mism == 0 && negative_ok)
		gpulog(LOG_INFO, thr_id, "Verthash self-test OK (GPU==CPU on %u nonces, negative control passed)", N);
	else
		gpulog(LOG_WARNING, thr_id, "Verthash self-test FAILED (%d/%u mismatch, neg=%d)", mism, N, (int) negative_ok);
	return mism == 0 && negative_ok;
}

extern "C" int scanhash_verthash(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[20];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	const int dev_id = device_map[thr_id];

	if (opt_benchmark)
		ptarget[7] = 0x00ff;

	if (!ensure_host_datafile(thr_id)) { *hashes_done = 0; sleep(1); return 0; }

	// throughput: memory-latency bound (4096 random reads/nonce). Round DOWN to a
	// multiple of 256 (exact grids; the IO kernel cannot early-return).
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << 15);
	throughput &= ~255u;
	if (throughput < 256) throughput = 256;

	if (!s_init[thr_id]) {
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		}
		cuda_get_arch(thr_id);

		// resident datafile + scratch buffers
		if (cudaMalloc(&d_memory[thr_id], s_dat_size) != cudaSuccess) {
			gpulog(LOG_ERR, thr_id, "Verthash: cudaMalloc %zu bytes for datafile failed", s_dat_size);
			return -1;
		}
		cudaMemcpy(d_memory[thr_id], s_dat, s_dat_size, cudaMemcpyHostToDevice);
		cudaMalloc(&d_iohashes[thr_id], (size_t) throughput * 4 * sizeof(uint2));
		cudaMalloc(&d_kstates[thr_id], 8 * 25 * sizeof(uint2));
		cudaMalloc(&d_results[thr_id], (size_t)(throughput + 1) * sizeof(uint32_t));
		s_throughput[thr_id] = throughput;
		CUDA_LOG_ERROR();

		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads",
		       throughput2intensity(throughput), throughput);
		s_init[thr_id] = true;
	}
	throughput = s_throughput[thr_id];  // fixed after alloc

	if (!s_selftested[thr_id]) {
		s_selftested[thr_id] = true;
		verthash_selftest(thr_id);
	}

	// Build the byteswapped header (matches the CPU oracle / cpuminer-opt
	// v128_bswap32_80). Words 0..18 are the fixed part; the nonce (raw counter)
	// is placed at word 19 by the kernel / by the reverify below.
	for (int k = 0; k < 19; k++)
		be32enc(&endiandata[k], pdata[k]);

	verthash_cuda_set_header(endiandata);
	verthash_cuda_set_mdiv(s_mdiv);
	verthash_cuda_precompute(d_kstates[thr_id]);

	const uint32_t Htarg = ptarget[7];

	do {
		cudaMemset(d_results[thr_id], 0, sizeof(uint32_t));
		verthash_cuda_hash(d_iohashes[thr_id], d_kstates[thr_id], d_memory[thr_id],
		                   endiandata[18], pdata[19], throughput, d_results[thr_id], Htarg);
		cudaDeviceSynchronize();

		uint32_t nres = 0;
		cudaMemcpy(&nres, d_results[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);
		if (nres > throughput) nres = throughput;

		if (nres && bench_algo < 0) {
			uint32_t *offs = (uint32_t *) malloc((size_t) nres * sizeof(uint32_t));
			cudaMemcpy(offs, d_results[thr_id] + 1, (size_t) nres * sizeof(uint32_t), cudaMemcpyDeviceToHost);

			work->valid_nonces = 0;
			for (uint32_t r = 0; r < nres && work->valid_nonces < 2; r++) {
				const uint32_t nonce = pdata[19] + offs[r];
				uint32_t _ALIGN(64) vhash[8];
				((uint32_t *) endiandata)[19] = nonce;   // raw nonce (verthash hashes it LE)
				verthash_hash_oracle(s_dat, s_dat_size, endiandata, vhash);
				if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
					work->nonces[work->valid_nonces] = nonce;
					bn_set_target_ratio(work, vhash, work->valid_nonces);
					work->valid_nonces++;
				}
			}
			free(offs);

			if (work->valid_nonces) {
				pdata[19] += throughput;
				if (pdata[19] > max_nonce) pdata[19] = max_nonce;
				*hashes_done = pdata[19] - first_nonce;
				return work->valid_nonces;
			}
			// all candidates were GPU false-positives (top word == target,
			// lower words failed) -- keep scanning.
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

extern "C" void free_verthash(int thr_id)
{
	if (!s_init[thr_id]) return;
	cudaDeviceSynchronize();
	if (d_memory[thr_id])   { cudaFree(d_memory[thr_id]);   d_memory[thr_id] = NULL; }
	if (d_iohashes[thr_id]) { cudaFree(d_iohashes[thr_id]); d_iohashes[thr_id] = NULL; }
	if (d_kstates[thr_id])  { cudaFree(d_kstates[thr_id]);  d_kstates[thr_id] = NULL; }
	if (d_results[thr_id])  { cudaFree(d_results[thr_id]);  d_results[thr_id] = NULL; }
	s_selftested[thr_id] = false;
	s_init[thr_id] = false;
}
