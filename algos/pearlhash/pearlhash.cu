// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * pearlhash: Pearl (PRL) noisy int8 GEMM proof of work, pool mode. See README.md.
 *
 * A = 0; B^T = 0 except its first 1024-byte chunk, which carries the trial (the
 * nonce). Every reported tile is re-verified on the CPU before its PlainProof is
 * built. The hashrate is in int8 MAC/s.
 */
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>	/* usleep (compat/unistd.h on Windows) */
#include <vector>

#include "miner.h"
#include "cuda_helper.h"
#include "cuda/selftest_gate.cuh"
#include "algos/pearlhash/pearl_device.cuh"
#include "algos/pearlhash/pearl_proof.h"

#define PEARL_M		8192
#define PEARL_N		8192
#define PEARL_K		2048
#define PEARL_R		128
/* hashes_done unit: 2^20 MAC, so one trial (2^37 MAC) is 2^17 units */
#define PEARL_TRIAL_UNITS	((uint32_t)(((uint64_t)PEARL_M * PEARL_N * PEARL_K) >> 20))
#define PEARL_CALL_SECONDS	1.0

/* ------------------------------------------------------------ proof stash */
/* struct work is copied by value to the submit thread, so it carries a stash id
 * in nonces[] instead of the ~45 KB proof. */
#define PEARL_STASH	64
static pthread_mutex_t stash_lock = PTHREAD_MUTEX_INITIALIZER;
static char *stash_proof[PEARL_STASH];
static uint32_t stash_id[PEARL_STASH];
static uint32_t stash_next = 1;

extern "C" uint32_t pearl_proof_put(char *b64)
{
	pthread_mutex_lock(&stash_lock);
	uint32_t id = stash_next++;
	if (!stash_next) stash_next = 1;
	const uint32_t s = id % PEARL_STASH;
	free(stash_proof[s]);
	stash_proof[s] = b64;
	stash_id[s] = id;
	pthread_mutex_unlock(&stash_lock);
	return id;
}

extern "C" char *pearl_proof_take(uint32_t id)
{
	char *p = NULL;
	pthread_mutex_lock(&stash_lock);
	const uint32_t s = id % PEARL_STASH;
	if (stash_id[s] == id && stash_proof[s]) {
		p = stash_proof[s];
		stash_proof[s] = NULL;
		stash_id[s] = 0;
	}
	pthread_mutex_unlock(&stash_lock);
	return p;
}

/* ------------------------------------------------------------ per-GPU state */
struct pearl_state {
	bool imma;
	const uint32_t *rows, *cols;
	pearl_config cfg;
	uint32_t *dU, *dP, *dApan, *dBpan, *dBbp, *dHits, *dChunk;
	uint32_t *hChunk;			/* pinned 1 KB staging for the nonce chunk */
	double t_host, t_gpu, t_hit;		/* -D timing: seconds since the last log */
	uint32_t t_trials;
	time_t t_last;
	uint8_t *A, *Bt;			/* host matrices: A = 0, B^T = 0 but chunk 0 */
	pearl_merkle ta, tb;
	bool have_job;
	uint8_t header[PEARL_HEADER_LEN];
	uint8_t job_key[32];
	uint64_t salt;
	uint32_t bench_trial;			/* --benchmark's own cursor, see scanhash */
};
static pearl_state st[MAX_GPUS];
static bool init_done[MAX_GPUS];
static bool sched_set[MAX_GPUS];

/* (trial, salt, gpu) as 6-bit digits, each byte a value in [-32, 31] */
static void pearl_nonce_chunk(uint8_t *chunk, uint32_t trial, uint64_t salt, int thr_id)
{
	memset(chunk, 0, BLAKE3_CHUNK_LEN);
	int p = 0;
	for (int i = 0; i < 6; i++) chunk[p++] = (uint8_t)(int8_t)((int)((trial >> (6 * i)) & 63) - 32);
	for (int i = 0; i < 11; i++) chunk[p++] = (uint8_t)(int8_t)((int)((salt >> (6 * i)) & 63) - 32);
	for (int i = 0; i < 2; i++) chunk[p++] = (uint8_t)(int8_t)((int)(((uint32_t)thr_id >> (6 * i)) & 63) - 32);
}

/* One trial on the GPU from given seeds: noise, prepass, kernel. dbg may be NULL. */
static void pearl_gpu_trial(pearl_state &s, uint32_t m, uint32_t n, const uint8_t a_seed[32],
	const uint8_t b_seed[32], const uint32_t bound[8], const uint32_t *base_b, uint32_t *dbg)
{
	const uint32_t k = s.cfg.k, r = s.cfg.r;
	pearl_key8 ka, kb;
	pearl_key_words(a_seed, ka.w);
	pearl_key_words(b_seed, kb.w);
	const size_t smem = PEARL_PREP_KW * 16 + (size_t)PEARL_BT * (r + 4);
	const dim3 ga(m / PEARL_BT, (k / 4 + PEARL_PREP_KW - 1) / PEARL_PREP_KW);
	const dim3 gb(n / PEARL_BT, (k / 4 + PEARL_PREP_KW - 1) / PEARL_PREP_KW);
	pearl_noise_uniform_kernel <<<(m * r / 32 + 255) / 256, 256>>> (ka, 'A', m * r / 32, s.dU);
	pearl_noise_perm_kernel <<<(k / 8 + 127) / 128, 128>>> (ka, 'A', k, r, s.dP);
	pearl_operand_panel_fast_kernel <<<ga, 256, smem>>> (NULL, (int8_t *)s.dU, s.dP, k, r, s.dApan);
	pearl_noise_uniform_kernel <<<(n * r / 32 + 255) / 256, 256>>> (kb, 'B', n * r / 32, s.dU);
	pearl_noise_perm_kernel <<<(k / 8 + 127) / 128, 128>>> (kb, 'B', k, r, s.dP);
	pearl_operand_panel_fast_kernel <<<gb, 256, smem>>> (base_b, (int8_t *)s.dU, s.dP, k, r, s.dBpan);

	pearl_gemm_params gp;
	gp.m = m; gp.n = n; gp.k = k; gp.r = r; gp.a_seed = ka;
	memcpy(gp.bound, bound, 32);
	cudaMemsetAsync(s.dHits, 0, (1 + 2 * PEARL_MAX_HITS) * 4);
	const uint32_t blocks = (m / PEARL_BT) * (n / PEARL_BT);
	if (s.imma)
		pearl_tiles_imma_kernel <<<blocks, 256>>> (gp, (const uint4 *)s.dApan, (const uint4 *)s.dBpan, dbg, s.dHits);
	else
		pearl_tiles_dp4a_kernel <<<blocks, 256>>> (gp, (const uint4 *)s.dApan, (const uint4 *)s.dBpan, dbg, s.dHits);
}

/* CPU re-verify of one tile; true if it meets the bound */
static bool pearl_cpu_tile(const pearl_state &s, const uint8_t *A, const uint8_t *Bt,
	const uint8_t a_seed[32], const uint8_t b_seed[32], uint32_t ro, uint32_t co,
	const uint32_t bound[8], uint32_t jp_out[16], uint32_t hash_out[8])
{
	const uint32_t k = s.cfg.k, r = s.cfg.r;
	uint32_t tr[8], tc[8], loc[8];
	for (int i = 0; i < 8; i++) { tr[i] = ro + s.rows[i]; tc[i] = co + s.cols[i]; loc[i] = i; }
	int8_t *na = (int8_t *)malloc(8 * (size_t)k), *nb = (int8_t *)malloc(8 * (size_t)k);
	pearl_noise_rows(a_seed, 'A', tr, 8, k, r, na);
	pearl_noise_rows(b_seed, 'B', tc, 8, k, r, nb);
	for (int i = 0; i < 8; i++)
		for (uint32_t l = 0; l < k; l++) {
			na[(size_t)i * k + l] = (int8_t)((int8_t)A[(size_t)tr[i] * k + l] + na[(size_t)i * k + l]);
			nb[(size_t)i * k + l] = (int8_t)((int8_t)Bt[(size_t)tc[i] * k + l] + nb[(size_t)i * k + l]);
		}
	uint32_t jp[16];
	uint8_t hb[32];
	pearl_tile_jackpot(na, nb, k, r, loc, 8, loc, 8, jp);
	pearl_jackpot_hash(jp, a_seed, hb);
	free(na); free(nb);
	if (jp_out) memcpy(jp_out, jp, 64);
	if (hash_out) pearl_key_words(hb, hash_out);
	return pearl_meets(hb, bound) != 0;
}

/* ------------------------------------------------------------ self-test */
/* Known answers from the Pearl Rust reference: seed.rs commitment_hash_pinned_vectors
 * (Salted), and a generated case (k 2048, rank 128) for noise and the ticket hash. */
static const char *KAT_SEED_B = "60ed9b73c5a9599b200b6cd563e7f0d5d9a67d2402d85fd4ef966c580080d0e5";
static const char *KAT_SEED_A = "301784168005ec833ab0aa60006f7fe7faaa95307d8c1fc6819b2ffdd717eccf";
static const char *KAT_C2_A_SEED = "1f3718ce3fb826e0e1a2588f7fa34c801475ee0cd70b160f719ac1b10ca5b2e7";
static const char *KAT_C2_B_SEED = "00ec501cf14fb4468b690f460c3ad90b94c53db61236e7427d4ce8009cd958fe";
static const char *KAT_C2_NOISE_A0 = "c7ff1edd13e50be217daf1ebfcd30a2201f7eb0cfe1df00b1a0710011b09f013f51af9cb2af428011d04090ed614d4e3";
static const char *KAT_C2_NOISE_A1 = "eafc21fdd0d0f11a28d105cefc0aea1a";
static const char *KAT_C2_NOISE_B0 = "0df9e0fa06f5091cecf51111f8ddcb0a0b081a15f41a08f10312f902f3f9dc15";
static const uint32_t KAT_C2_JP[16] = {
	0x0001954c, 0xfffed045, 0xfffff348, 0xfffcd5c1, 0xfffc114b, 0xfffd87a5, 0x0002578e, 0x0003fcbb,
	0xfffc98da, 0x0002bbe5, 0xfffe7f2f, 0x0003190e, 0x0001fdd6, 0xfffcc8ee, 0x0002c5d0, 0xfffe2e48 };
static const char *KAT_C2_JP_HASH = "02793b88ea575adbc98cbe26890ef31e9504445263afa3f6dc2c2fb6136e4247";

static bool hex_eq(const uint8_t *b, size_t n, const char *hex)
{
	char t[3];
	for (size_t i = 0; i < n; i++) {
		snprintf(t, 3, "%02x", b[i]);
		if (t[0] != hex[2 * i] || t[1] != hex[2 * i + 1]) return false;
	}
	return true;
}

static void hex_bytes(const char *hex, uint8_t *out, size_t n)
{
	for (size_t i = 0; i < n; i++) {
		char t[3] = { hex[2 * i], hex[2 * i + 1], 0 };
		out[i] = (uint8_t)strtoul(t, NULL, 16);
	}
}

static bool pearl_host_kats(int fault)
{
	bool ok = true;
	uint8_t jk[32], ha[32], hb[32], bs[32], as[32];
	memset(jk, 0x11, 32); memset(ha, 0xAA, 32); memset(hb, 0xBB, 32);
	pearl_seeds(jk, ha, hb, 192, 320, bs, as);
	if (fault == 1) as[0] ^= 1;
	ok &= hex_eq(bs, 32, KAT_SEED_B) && hex_eq(as, 32, KAT_SEED_A);

	uint8_t sa[32], sb[32];
	hex_bytes(KAT_C2_A_SEED, sa, 32);
	hex_bytes(KAT_C2_B_SEED, sb, 32);
	const uint32_t r01[2] = { 0, 1 };
	int8_t na[2 * PEARL_K], nb[PEARL_K];
	pearl_noise_rows(sa, 'A', r01, 2, PEARL_K, PEARL_R, na);
	pearl_noise_rows(sb, 'B', r01, 1, PEARL_K, PEARL_R, nb);
	ok &= hex_eq((uint8_t *)na, 48, KAT_C2_NOISE_A0) && hex_eq((uint8_t *)na + PEARL_K, 16, KAT_C2_NOISE_A1)
		&& hex_eq((uint8_t *)nb, 32, KAT_C2_NOISE_B0);

	uint8_t h[32];
	pearl_jackpot_hash(KAT_C2_JP, sa, h);
	ok &= hex_eq(h, 32, KAT_C2_JP_HASH);
	return ok;
}

/* GPU vs host on every tile of a 256 x 256 trial, then a changed seed must change the tiles */
static bool pearl_gpu_differential(int thr_id, pearl_state &s, int fault, bool *ran)
{
	const uint32_t m = 256, n = 256, k = s.cfg.k, r = s.cfg.r;
	const uint32_t ntile = (m / 8) * (n / 8);
	*ran = false;
	uint8_t *A = (uint8_t *)calloc((size_t)m * k, 1), *Bt = (uint8_t *)malloc((size_t)n * k);
	uint32_t lcg = 0x9e3779b9u;
	for (size_t i = 0; i < (size_t)n * k; i++) {
		lcg = lcg * 1664525u + 1013904223u;
		Bt[i] = (uint8_t)(int8_t)((int)(lcg >> 25) - 64);
	}
	uint8_t header[PEARL_HEADER_LEN];
	for (int i = 0; i < PEARL_HEADER_LEN; i++) header[i] = (uint8_t)(i * 37 + 11);
	uint8_t jk[32], ha[32], hb[32], bs[32], as[32];
	pearl_job_key(header, &s.cfg, jk);
	pearl_matrix_root(A, (size_t)m * k, jk, ha);
	pearl_matrix_root(Bt, (size_t)n * k, jk, hb);
	pearl_seeds(jk, ha, hb, m, n, bs, as);
	uint32_t bound[8] = { 0, 0, 0, 0, 0, 0, 0, 0x08000000u };	/* ~1/32 tickets */

	uint32_t *d_bt = NULL, *d_btp = NULL, *d_dbg = NULL;
	uint32_t *dbg = (uint32_t *)malloc((size_t)ntile * 96), *dbg2 = (uint32_t *)malloc((size_t)ntile * 96);
	bool ok = false;
	if (cudaMalloc(&d_bt, (size_t)n * k) != cudaSuccess || cudaMalloc(&d_btp, (size_t)n * k) != cudaSuccess ||
	    cudaMalloc(&d_dbg, (size_t)ntile * 96) != cudaSuccess) {
		cudaGetLastError();
		goto done;
	}
	cudaMemcpy(d_bt, Bt, (size_t)n * k, cudaMemcpyHostToDevice);
	pearl_to_panel_kernel <<<(n * k / 4 + 255) / 256, 256>>> ((const int8_t *)d_bt, n, k, d_btp);
	{
		uint8_t as_gpu[32];
		memcpy(as_gpu, as, 32);
		if (fault == 2) as_gpu[3] ^= 0x40;
		pearl_gpu_trial(s, m, n, as_gpu, bs, bound, d_btp, d_dbg);
	}
	if (cudaDeviceSynchronize() != cudaSuccess) { cudaGetLastError(); goto done; }
	uint32_t hits[1 + 2 * PEARL_MAX_HITS];
	cudaMemcpy(dbg, d_dbg, (size_t)ntile * 96, cudaMemcpyDeviceToHost);
	cudaMemcpy(hits, s.dHits, sizeof(hits), cudaMemcpyDeviceToHost);
	*ran = true;
	{
		int8_t *na = (int8_t *)malloc((size_t)m * k), *nb = (int8_t *)malloc((size_t)n * k);
		pearl_noise_full(as, 'A', m, k, r, na);
		pearl_noise_full(bs, 'B', n, k, r, nb);
		for (size_t i = 0; i < (size_t)n * k; i++) nb[i] = (int8_t)((int8_t)Bt[i] + nb[i]);
		std::vector<uint32_t> roff, coff;
		for (uint32_t o = 0; o < m; o++) if (pearl_offset_is_valid(&s.cfg.rows, o)) roff.push_back(o);
		for (uint32_t o = 0; o < n; o++) if (pearl_offset_is_valid(&s.cfg.cols, o)) coff.push_back(o);
		uint32_t bad = 0, ref_hits = 0;
		for (uint32_t t = 0; t < ntile; t++) {
			uint32_t tr[8], tc[8], jp[16], hw[8];
			uint8_t hh[32];
			for (int i = 0; i < 8; i++) { tr[i] = roff[t / (n / 8)] + s.rows[i]; tc[i] = coff[t % (n / 8)] + s.cols[i]; }
			pearl_tile_jackpot(na, nb, k, r, tr, 8, tc, 8, jp);
			pearl_jackpot_hash(jp, as, hh);
			pearl_key_words(hh, hw);
			ref_hits += pearl_meets(hh, bound);
			if (memcmp(jp, dbg + (size_t)t * 24, 64) || memcmp(hw, dbg + (size_t)t * 24 + 16, 32)) bad++;
		}
		free(na); free(nb);
		ok = bad == 0 && hits[0] == ref_hits && ref_hits > 0;
		if (!ok)
			gpulog(LOG_ERR, thr_id, "pearlhash differential: %u of %u tiles differ, hits %u vs %u",
				bad, ntile, hits[0], ref_hits);
	}
	if (ok) {
		uint8_t as_bad[32];
		memcpy(as_bad, as, 32);
		as_bad[17] ^= 1;
		pearl_gpu_trial(s, m, n, as_bad, bs, bound, d_btp, d_dbg);
		if (cudaDeviceSynchronize() != cudaSuccess) { cudaGetLastError(); ok = false; *ran = false; goto done; }
		cudaMemcpy(dbg2, d_dbg, (size_t)ntile * 96, cudaMemcpyDeviceToHost);
		if (!memcmp(dbg, dbg2, (size_t)ntile * 96)) {
			gpulog(LOG_ERR, thr_id, "pearlhash differential: negative control did not change the tiles");
			ok = false;
		}
	}
done:
	cudaFree(d_bt); cudaFree(d_btp); cudaFree(d_dbg);
	free(A); free(Bt); free(dbg); free(dbg2);
	return ok;
}

static bool pearl_selftest(int thr_id, pearl_state &s)
{
	const char *fenv = getenv("PEARLHASH_SELFTEST_FAULT");
	const int fault = fenv ? atoi(fenv) : 0;
	if (fault)
		gpulog(LOG_WARNING, thr_id, "pearlhash: SELF-TEST FAULT INJECTION ACTIVE (%d) -- for testing only", fault);
	const bool kat = pearl_host_kats(fault);
	bool ran = false;
	const bool diff = pearl_gpu_differential(thr_id, s, fault, &ran);
	if (!ran) selftest_cuda_fault();
	const bool passed = kat && diff;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "pearlhash self-test FAILED (host KATs %s, GPU differential %s)",
			kat ? "ok" : "FAIL", diff ? "ok" : (ran ? "FAIL" : "not run"));
	else
		gpulog(LOG_INFO, thr_id, "pearlhash self-test passed (seed/noise/ticket KATs, %s kernel == oracle on 1024 tiles, negative control)",
			s.imma ? "IMMA" : "dp4a");
	return selftest_gate(thr_id, "pearlhash", passed);
}

/* ------------------------------------------------------------ init / job */
static void pearl_blocking_sync(int thr_id)
{
	unsigned flags = 0;
	cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
	cudaGetLastError();
	if (cudaGetDeviceFlags(&flags) != cudaSuccess ||
	    (flags & cudaDeviceScheduleMask) != cudaDeviceScheduleBlockingSync)
		cuda_reset_device(thr_id, NULL);
	sched_set[thr_id] = true;
}

static void pearl_init(int thr_id)
{
	pearl_state &s = st[thr_id];
	const int dev = device_map[thr_id];
	cudaSetDevice(dev);
	if (opt_cudaschedule < 0) pearl_blocking_sync(thr_id);

	s.imma = device_sm[dev] >= 800;
	s.rows = s.imma ? PEARL_IMMA_ROWS : PEARL_DP4A_PATTERN;
	s.cols = s.imma ? PEARL_IMMA_COLS : PEARL_DP4A_PATTERN;
	s.cfg.k = PEARL_K;
	s.cfg.r = PEARL_R;
	pearl_pattern_from_list(s.rows, 8, &s.cfg.rows);
	pearl_pattern_from_list(s.cols, 8, &s.cfg.cols);

	CUDA_SAFE_CALL(cudaMalloc(&s.dU, (size_t)PEARL_M * PEARL_R));
	CUDA_SAFE_CALL(cudaMalloc(&s.dP, (size_t)PEARL_K * 4));
	CUDA_SAFE_CALL(cudaMalloc(&s.dApan, (size_t)PEARL_M * PEARL_K));
	CUDA_SAFE_CALL(cudaMalloc(&s.dBpan, (size_t)PEARL_N * PEARL_K));
	CUDA_SAFE_CALL(cudaMalloc(&s.dBbp, (size_t)PEARL_N * PEARL_K));
	CUDA_SAFE_CALL(cudaMalloc(&s.dHits, (1 + 2 * PEARL_MAX_HITS) * 4));
	CUDA_SAFE_CALL(cudaMemset(s.dBbp, 0, (size_t)PEARL_N * PEARL_K));
	CUDA_SAFE_CALL(cudaMalloc(&s.dChunk, BLAKE3_CHUNK_LEN));
	CUDA_SAFE_CALL(cudaMallocHost(&s.hChunk, BLAKE3_CHUNK_LEN));
	s.A = (uint8_t *)calloc((size_t)PEARL_M * PEARL_K, 1);
	s.Bt = (uint8_t *)calloc((size_t)PEARL_N * PEARL_K, 1);
	s.have_job = false;

	pearl_selftest(thr_id, s);
	gpulog(LOG_INFO, thr_id, "pearlhash: %s kernel, m=n=%u k=%u rank %u, %u tickets of %u MAC per trial",
		s.imma ? "IMMA (sm_80+)" : "dp4a", PEARL_M, PEARL_K, PEARL_R,
		(PEARL_M / 8) * (PEARL_N / 8), 64 * PEARL_K);
	init_done[thr_id] = true;
}

/* new header: job_key, both trees (B^T chunk 0 is patched per trial), fresh salt */
static void pearl_new_job(int thr_id, const uint8_t *header)
{
	pearl_state &s = st[thr_id];
	memcpy(s.header, header, PEARL_HEADER_LEN);
	pearl_job_key(header, &s.cfg, s.job_key);
	if (s.have_job) { pearl_merkle_free(&s.ta); pearl_merkle_free(&s.tb); }
	memset(s.Bt, 0, BLAKE3_CHUNK_LEN);
	pearl_merkle_build(&s.ta, s.A, (size_t)PEARL_M * PEARL_K, s.job_key);
	pearl_merkle_build(&s.tb, s.Bt, (size_t)PEARL_N * PEARL_K, s.job_key);
	s.salt = ((uint64_t)rand() << 40) ^ ((uint64_t)rand() << 20) ^ (uint64_t)rand() ^ ((uint64_t)time(NULL) << 24)
		^ ((uint64_t)thr_id << 58);
	s.have_job = true;
}

/* ------------------------------------------------------------ scanhash */
extern "C" int scanhash_pearlhash(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	pearl_state &s = st[thr_id];
	if (!init_done[thr_id]) pearl_init(thr_id);

	/* job refused by the stratum layer (unsupported cert_version): idle */
	if (!work->is_pearl && !opt_benchmark) {
		usleep(200 * 1000);
		*hashes_done = 0;
		return 0;
	}

	uint8_t header[PEARL_HEADER_LEN];
	memcpy(header, pdata, PEARL_HEADER_LEN);
	uint32_t bound[8];
	if (work->is_pearl) {
		if (!pearl_share_bound(work->pearl_target, &s.cfg, bound)) {
			static char logged[MAX_GPUS][65];
			if (strcmp(logged[thr_id], work->pearl_target)) {
				strcpy(logged[thr_id], work->pearl_target);
				gpulog(LOG_ERR, thr_id, "pearlhash: unusable share target %s", work->pearl_target);
			}
			*hashes_done = 0;
			return 0;
		}
	} else {
		/* --benchmark: about one share per 16 trials, written into the header so a
		 * dumped share is valid against its own nbits */
		const uint32_t nbits = 0x1c008000;
		memcpy(header + 72, &nbits, 4);
		pearl_bound(nbits, &s.cfg, bound);
	}
	/* --benchmark changes data[18] and resets the cursor on every call: ignore that
	 * word and keep a private cursor, or every call re-mines trial 0 */
	const bool bench = opt_benchmark;
	const size_t cmp_len = bench ? 72 : PEARL_HEADER_LEN;
	if (!s.have_job || memcmp(header, s.header, cmp_len)) {
		pearl_new_job(thr_id, header);
		s.bench_trial = 0;
	}
	uint32_t *cursor = bench ? &s.bench_trial : &pdata[19];

	work->valid_nonces = 0;
	uint32_t trials = 0;
	struct timeval t0, t1;
	gettimeofday(&t0, NULL);
	uint8_t chunk[BLAKE3_CHUNK_LEN];
	do {
		struct timeval ta, tb2, tc;
		gettimeofday(&ta, NULL);
		const uint32_t trial = *cursor;
		pearl_nonce_chunk(chunk, trial, s.salt, thr_id);
		memcpy(s.Bt, chunk, BLAKE3_CHUNK_LEN);
		pearl_merkle_update_leaf(&s.tb, 0, chunk);
		memcpy(s.hChunk, chunk, BLAKE3_CHUNK_LEN);
		cudaMemcpyAsync(s.dChunk, s.hChunk, BLAKE3_CHUNK_LEN, cudaMemcpyHostToDevice);
		pearl_patch_row0_kernel <<<1, BLAKE3_CHUNK_LEN / 4>>> (s.dChunk, s.dBbp);

		uint8_t bs[32], as[32];
		pearl_seeds(s.job_key, s.ta.root, s.tb.root, PEARL_M, PEARL_N, bs, as);
		pearl_gpu_trial(s, PEARL_M, PEARL_N, as, bs, bound, s.dBbp, NULL);
		gettimeofday(&tb2, NULL);
		uint32_t hits[1 + 2 * PEARL_MAX_HITS];
		CUDA_SAFE_CALL(cudaMemcpy(hits, s.dHits, sizeof(hits), cudaMemcpyDeviceToHost));
		gettimeofday(&tc, NULL);
		*cursor = trial + 1;
		trials++;
		s.t_host += (tb2.tv_sec - ta.tv_sec) + 1e-6 * (tb2.tv_usec - ta.tv_usec);
		s.t_gpu += (tc.tv_sec - tb2.tv_sec) + 1e-6 * (tc.tv_usec - tb2.tv_usec);
		s.t_trials++;

		const uint32_t nh = min(hits[0], (uint32_t)PEARL_MAX_HITS);
		for (uint32_t h = 0; h < nh && work->valid_nonces < MAX_NONCES; h++) {
			const uint32_t ro = hits[1 + 2 * h], co = hits[2 + 2 * h];
			if (!pearl_cpu_tile(s, s.A, s.Bt, as, bs, ro, co, bound, NULL, NULL)) {
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
					gpulog(LOG_WARNING, thr_id, "pearlhash: tile %u,%u of trial %08x does not validate on CPU!", ro, co, trial);
				continue;
			}
			uint32_t tr[8], tc[8];
			for (int i = 0; i < 8; i++) { tr[i] = ro + s.rows[i]; tc[i] = co + s.cols[i]; }
			pearl_buf b = { 0, 0, 0 };
			pearl_plain_proof(&b, PEARL_M, PEARL_N, PEARL_K, PEARL_R, &s.ta, s.A, tr, 8, &s.tb, s.Bt, tc, 8);
			/* PEARLHASH_DUMP=<dir>: write the first share as header76.hex + proof.bin */
			static bool dumped[MAX_GPUS];
			const char *dump = getenv("PEARLHASH_DUMP");
			if (dump && !dumped[thr_id]) {
				char path[512];
				snprintf(path, sizeof(path), "%s/proof.bin", dump);
				FILE *f = fopen(path, "wb");
				if (f) { fwrite(b.p, 1, b.n, f); fclose(f); }
				snprintf(path, sizeof(path), "%s/header76.hex", dump);
				f = fopen(path, "w");
				if (f) { for (int i = 0; i < PEARL_HEADER_LEN; i++) fprintf(f, "%02x", header[i]); fclose(f); }
				dumped[thr_id] = true;
				gpulog(LOG_NOTICE, thr_id, "pearlhash: share of trial %08x dumped to %s (%zu B)", trial, dump, b.n);
			}
			char *b64 = pearl_base64(b.p, b.n);
			free(b.p);
			if (opt_benchmark) {
				free(b64);
				work->nonces[work->valid_nonces] = trial;
			} else {
				work->nonces[work->valid_nonces] = pearl_proof_put(b64);
			}
			work->sharediff[work->valid_nonces] = work->targetdiff;
			work->valid_nonces++;
		}
		if (nh) {
			struct timeval td;
			gettimeofday(&td, NULL);
			s.t_hit += (td.tv_sec - tc.tv_sec) + 1e-6 * (td.tv_usec - tc.tv_usec);
		}
		if (opt_debug && time(NULL) - s.t_last >= 10 && s.t_trials) {
			gpulog(LOG_DEBUG, thr_id, "pearlhash: %u trials, per trial: host+launch %.3f ms, gpu wait %.3f ms, share path %.3f ms",
				s.t_trials, 1e3 * s.t_host / s.t_trials, 1e3 * s.t_gpu / s.t_trials, 1e3 * s.t_hit / s.t_trials);
			s.t_host = s.t_gpu = s.t_hit = 0; s.t_trials = 0; s.t_last = time(NULL);
		}
		if (work->valid_nonces) break;
		gettimeofday(&t1, NULL);
		if ((t1.tv_sec - t0.tv_sec) + 1e-6 * (t1.tv_usec - t0.tv_usec) >= PEARL_CALL_SECONDS) break;
	} while (!work_restart[thr_id].restart && (bench || pdata[19] < max_nonce));

	*hashes_done = (unsigned long)trials * PEARL_TRIAL_UNITS;
	return work->valid_nonces;
}

extern "C" void free_pearlhash(int thr_id)
{
	if (!init_done[thr_id]) return;
	pearl_state &s = st[thr_id];
	cudaSetDevice(device_map[thr_id]);
	cudaFree(s.dU); cudaFree(s.dP); cudaFree(s.dApan); cudaFree(s.dBpan); cudaFree(s.dBbp); cudaFree(s.dHits);
	cudaFree(s.dChunk); cudaFreeHost(s.hChunk);
	free(s.A); free(s.Bt);
	if (s.have_job) { pearl_merkle_free(&s.ta); pearl_merkle_free(&s.tb); }
	s.have_job = false;
	if (sched_set[thr_id]) {
		cudaSetDeviceFlags(cudaDeviceScheduleAuto);
		cudaGetLastError();
		sched_set[thr_id] = false;
	}
	init_done[thr_id] = false;
}
