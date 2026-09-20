/**
 * Flex algorithm (Kylacoin / Lyncoin "flex") -- GPU driver.
 *
 * 15 core rounds + 3 CryptoNight-v1 rounds + a closing SHA3-256, where the
 * order is derived from SHA3-512 of the WHOLE 80-byte header and therefore
 * changes with EVERY NONCE.  The chain itself lives in
 * algos/flex/cuda_flex_pipeline.cuh; this file is the miner-facing driver:
 * throughput sizing, the startup guards, the candidate screen and the host
 * re-verify.
 *
 * Consensus notes that are easy to get wrong -- all four are gated, see
 * algos/flex/flex.h and the project notes
 * - every keccak is SHA-3 (0x06), never legacy Keccak (0x01);
 * - the CN finalization is `state[0] & 2` over {blake, groestl, skein-512};
 * - the high 32 bytes of a CN result are NOT zeroed and ARE hashed by the
 * closing SHA3-256;
 * - the core pool is 14 wide, drops JH, and is REINDEXED.
 *
 * Difficulty: flex leaves opt_target_factor at 1.0 -- it does NOT share
 * ghostrider's 2^16 factor.
 *
 * Reference: cpuminer-opt algo/flex (pool-confirmed on Kylacoin).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include "algos/flex/cuda_flex_pipeline.cuh"
#include "algos/flex/cuda_flex_filter.cuh"

static flex_ctx ctx[MAX_GPUS];
static bool init_done[MAX_GPUS] = { 0 };

// ----------------------------------------------------------------------------
// Startup guard 1 -- per-kernel self-consistency at full throughput.
//
// Each stage must produce the same answer on repeated runs. Variation means an
// intra-kernel race; that is how ghostrider's shared-AES-table bug in
// shavite/echo was found, and flex drives the same stages through a different
// launch pattern (many small launches instead of a few large ones), which is
// exactly the regime where such a race changes character.
// ----------------------------------------------------------------------------
static int flex_guard_races(int thr_id, uint32_t throughput)
{
	uint8_t tv[80];
	for (int i = 0; i < 80; i++) tv[i] = (uint8_t)(i * 13 + 5);

	int races = 0;
	for (int a = 0; a < FLEX_CORE_ALGO_COUNT; a++) {
		uint8_t first[64], g[64];
		int v64 = 0;
		for (int rep = 0; rep < 8; rep++) {
			cudaMemcpy(ctx[thr_id].d_hash, tv, 64, cudaMemcpyHostToDevice);
			flex_stage_hash_64(a, thr_id, throughput, ctx[thr_id].d_hash, 0);
			cudaMemcpy(g, ctx[thr_id].d_hash, 64, cudaMemcpyDeviceToHost);
			if (rep == 0) memcpy(first, g, 64);
			else if (memcmp(g, first, 64)) v64++;
		}
		if (v64) {
			races++;
			gpulog(LOG_ERR, thr_id, "flex: RACE in core algo %d (hash64 varied %d/7)", a, v64);
		}
	}
	return races;
}

// ----------------------------------------------------------------------------
// Startup guard 2 -- the whole chain, GPU vs CPU, with a bisect.
//
// Runs the real 19-step pipeline on a small batch and compares every lane's
// digest to flex_hash(). On a mismatch it re-runs with the pipeline's step
// limit to report the FIRST diverging step plus that lane's derived order --
// without it a failure says only "one of 19 stages, on one of 14 algorithms".
// ----------------------------------------------------------------------------
static void flex_cpu_chain_upto(const uint32_t *hdr, int steps, uint8_t out[64])
{
	uint8_t coreOrder[FLEX_CORE_CHAIN_LEN], cnOrder[FLEX_CN_ALGO_COUNT];
	uint8_t h1[64] = { 0 }, h2[64] = { 0 };
	int step = 0, cn = 0;

	flex_derive_order(hdr, coreOrder, cnOrder, NULL);
	uint8_t *in = h2, *o = h1;

	if (step++ >= steps) { memcpy(out, in, 64); return; }
	flex_core_algo_cpu(coreOrder[0], hdr, o, 80);
	{ uint8_t *t = in; in = o; o = t; }

	for (int r = 1; r < FLEX_CORE_CHAIN_LEN; r++) {
		if (step++ >= steps) { memcpy(out, in, 64); return; }
		flex_core_algo_cpu(coreOrder[r], in, o, 64);
		{ uint8_t *t = in; in = o; o = t; }
		if (r == 4 || r == 9 || r == 14) {
			if (step++ >= steps) { memcpy(out, in, 64); return; }
			flex_cn_algo_cpu(cnOrder[cn++], in, o, 64);
			{ uint8_t *t = in; in = o; o = t; }
		}
	}
	if (step++ >= steps) { memcpy(out, in, 64); return; }
	memset(out, 0, 64);
	flex_sha3_256(out, in, 64);
}

/* the nonce filter: the same chain check over the FILTERED path -- the one scanhash now
 * takes. flex_guard_chain() drives the CONTIGUOUS path, so on its own it
 * leaves the shipped code ungated: how -a lyra2v2 stayed dead behind a green
 * gate. This also proves what the filter gate cannot -- that the round-0
 * GATHER hands each scattered lane its OWN 80-byte state (a chunk off-by-one
 * would mis-pair lanes with nonces and is invisible to a filter-only check).
 */
static int flex_guard_filtered(int thr_id, uint32_t saved_threads)
{
	uint32_t guard_lanes = 256u;
	{
		const char *ev = getenv("FLEX_GUARD_LANES");
		if (ev) {
			const unsigned long v = strtoul(ev, NULL, 0);
			if (v >= FLEX_CN_TPB) guard_lanes = (uint32_t)v;
		}
	}
	const uint32_t T = min(saved_threads, guard_lanes);
	flex_ctx *c = &ctx[thr_id];
	const uint32_t real = c->threads;
	c->threads = T;

	uint8_t  *h_gpu = (uint8_t*)malloc((size_t)64 * T);
	uint32_t *h_n   = (uint32_t*)malloc(sizeof(uint32_t) * T);
	int diffs = 0, checked = 0;
	uint32_t lcg = 0x2468aceu;

	for (int it = 0; it < 2 && !diffs; it++) {
		uint32_t pd[20], ed[20];
		for (int k = 0; k < 20; k++) { lcg = lcg * 1664525u + 1013904223u; pd[k] = lcg; }
		for (int k = 0; k < 20; k++) be32enc(&ed[k], pd[k]);

		/* Arm 1 uses a short span so the filter under-yields. At the normal span it
		 * over-yields and the count clamps to the capacity, which makes lanes equal
		 * stride and hides that class of fault. */
		const uint32_t span = (it == 1)
			? (T * FLEX_FILTER_SPAN_PER_LANE) / 4u
			: (T * FLEX_FILTER_SPAN_PER_LANE);
		uint32_t yield = 0;
		c->span = span;
		cudaMemset(c->d_count, 0, sizeof(uint32_t));
		flex_seed_setBlock_80(ed);
		flex_filter_cpu(span, pd[19], T, c->d_nonces, c->d_count);
		cudaMemcpy(&yield, c->d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
		if (yield > T) yield = T;
		yield = (yield / FLEX_CN_TPB) * FLEX_CN_TPB;
		if (yield == 0) continue;
		if (opt_debug)
			gpulog(LOG_DEBUG, thr_id, "flex: guard arm %d span %u -> yield %u of %u%s",
			       it, span, yield, T, (yield == T) ? " (CLAMPED)" : " (under-yield)");
		cudaMemcpy(h_n, c->d_nonces, sizeof(uint32_t) * yield, cudaMemcpyDeviceToHost);

		flex_pipeline_run(c, thr_id, ed, pd, pd[19], -1, c->d_nonces, yield);
		const uint32_t L = c->lanes;
		cudaMemcpy(h_gpu, c->d_hash, (size_t)64 * L, cudaMemcpyDeviceToHost);

		for (uint32_t t = 0; t < L; t++) {
			uint32_t lane[20];
			uint32_t _ALIGN(64) ref[8];
			memcpy(lane, ed, 80);
			be32enc(&lane[19], h_n[t]);
			flex_hash(ref, lane);
			checked++;
			if (memcmp(h_gpu + (size_t)t * 64, ref, 32) != 0) {
				diffs++;
				if (diffs == 1)
					gpulog(LOG_ERR, thr_id,
					       "flex: FILTERED chain DIFF at lane %u, nonce %08x", t, h_n[t]);
			}
		}
	}

	c->threads = real;
	free(h_gpu); free(h_n);
	if (checked == 0) {
		gpulog(LOG_ERR, thr_id, "flex: filtered guard VACUOUS -- the filter yielded nothing");
		return -1;
	}
	if (!diffs)
		gpulog(LOG_INFO, thr_id, "flex: GPU==CPU on the filtered chain (%d lanes)", checked);
	return diffs;
}

static int flex_guard_chain(int thr_id, uint32_t saved_threads)
{
	/* A small batch keeps startup fast, because this guard is about chain
	 * logic rather than throughput.
	 *
	 * But a SMALL batch cannot see a SCALE-dependent fault, and an earlier phase shipped
	 * exactly one: with variable-stride packing every digest was wrong above
	 * ~4096 lanes while 256 lanes passed cleanly. FLEX_GUARD_LANES overrides
	 * the width so the guard can be run at the real batch size -- do that
	 * after ANY change to the scratchpad layout or the lane count. */
	uint32_t guard_lanes = 256u;
	{
		const char *ev = getenv("FLEX_GUARD_LANES");
		if (ev) {
			const unsigned long v = strtoul(ev, NULL, 0);
			if (v >= FLEX_CN_TPB) guard_lanes = (uint32_t)v;
		}
	}
	const uint32_t T = min(saved_threads, guard_lanes);
	flex_ctx *c = &ctx[thr_id];
	/* Shrink the CAPACITY for the guard so the pipeline derives and runs only
	 * T lanes; flex_pipeline_run() resets c->lanes from it on every call. */
	const uint32_t real = c->threads;
	c->threads = T;

	uint8_t *h_gpu = (uint8_t*)malloc((size_t)64 * T);
	int diffs = 0;
	uint32_t lcg = 0x12345678u;

	for (int t = 0; t < 3 && !diffs; t++) {
		uint32_t pd[20], ed[20];
		for (int k = 0; k < 20; k++) { lcg = lcg * 1664525u + 1013904223u; pd[k] = lcg; }
		for (int k = 0; k < 20; k++) be32enc(&ed[k], pd[k]);
		/* RAW frame, like pdata[19] -- see flex_sha3_512_seed_80. Lane l is the
		 * raw nonce startNonce + l, whose header word is be32enc of that. */
		const uint32_t startNonce = pd[19];

		flex_pipeline_run(c, thr_id, ed, pd, startNonce, -1, NULL, 0);
		cudaMemcpy(h_gpu, c->d_hash, (size_t)64 * T, cudaMemcpyDeviceToHost);

		int firstbad = -1;
		for (uint32_t l = 0; l < T; l++) {
			uint32_t lane[20];
			uint8_t ref[32];
			memcpy(lane, ed, 80);
			be32enc(&lane[19], startNonce + l);
			flex_hash(ref, lane);
			if (memcmp(h_gpu + (size_t)l * 64, ref, 32) != 0) { firstbad = (int)l; break; }
		}
		if (firstbad < 0)
			continue;

		diffs++;
		uint32_t lane[20];
		memcpy(lane, ed, 80);
		be32enc(&lane[19], startNonce + (uint32_t)firstbad);
		for (int s = 1; s <= FLEX_PIPELINE_STEPS; s++) {
			uint8_t cref[64];
			flex_pipeline_run(c, thr_id, ed, pd, startNonce, s, NULL, 0);
			cudaMemcpy(h_gpu, c->d_hash, (size_t)64 * T, cudaMemcpyDeviceToHost);
			flex_cpu_chain_upto(lane, s, cref);
			const int n = (s == FLEX_PIPELINE_STEPS) ? 32 : 64;
			if (memcmp(h_gpu + (size_t)firstbad * 64, cref, n) != 0) {
				uint8_t co[FLEX_CORE_CHAIN_LEN], cv[FLEX_CN_ALGO_COUNT];
				flex_derive_order(lane, co, cv, NULL);
				gpulog(LOG_ERR, thr_id,
					"flex: GPU!=CPU, first divergence at step %d/%d (lane %d)",
					s, FLEX_PIPELINE_STEPS, firstbad);
				gpulog(LOG_ERR, thr_id,
					"flex: that lane's core order %d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d cn %d,%d,%d",
					co[0],co[1],co[2],co[3],co[4],co[5],co[6],co[7],co[8],co[9],
					co[10],co[11],co[12],co[13],co[14], cv[0],cv[1],cv[2]);
				break;
			}
		}
	}

	free(h_gpu);
	c->threads = real;
	return diffs;
}

// ----------------------------------------------------------------------------
// Startup guard 3 -- WIDTH INVARIANCE.
//
// Every lane's chain is a function of its own nonce alone, so lane L must
// produce the same digest whatever the batch width. Running one header at two
// widths and comparing the overlap therefore needs no CPU reference at all --
// and it catches the class guard 2 cannot: guard 2 runs a small batch, and a
// fault that only appears at scale passes it cleanly. an earlier phase shipped exactly such
// a fault (every digest wrong above ~6144 lanes, 256 lanes perfect).
//
// Reports the first step whose output stops being width-invariant, plus how
// many of the overlapping lanes moved -- the count distinguishes "one bucket
// is wrong" from "everything is wrong".
// ----------------------------------------------------------------------------
static int flex_guard_width(int thr_id, uint32_t wide)
{
	flex_ctx *c = &ctx[thr_id];
	const uint32_t narrow = 256;
	if (wide <= narrow)
		return 0;

	const uint32_t real = c->threads;
	uint32_t pd[20], ed[20];
	uint32_t lcg = 0x5bd1e995u;
	for (int k = 0; k < 20; k++) { lcg = lcg * 1664525u + 1013904223u; pd[k] = lcg; }
	for (int k = 0; k < 20; k++) be32enc(&ed[k], pd[k]);
	const uint32_t startNonce = pd[19];

	uint8_t *a = (uint8_t*)malloc((size_t)64 * narrow);
	uint8_t *b = (uint8_t*)malloc((size_t)64 * narrow);
	int bad_step = 0;

	for (int s = 1; s <= FLEX_PIPELINE_STEPS && !bad_step; s++) {
		c->threads = narrow;
		flex_pipeline_run(c, thr_id, ed, pd, startNonce, s, NULL, 0);
		cudaMemcpy(a, c->d_hash, (size_t)64 * narrow, cudaMemcpyDeviceToHost);

		c->threads = wide;
		flex_pipeline_run(c, thr_id, ed, pd, startNonce, s, NULL, 0);
		cudaMemcpy(b, c->d_hash, (size_t)64 * narrow, cudaMemcpyDeviceToHost);

		const int n = (s == FLEX_PIPELINE_STEPS) ? 32 : 64;
		uint32_t moved = 0, first = 0;
		for (uint32_t l = 0; l < narrow; l++)
			if (memcmp(a + (size_t)l * 64, b + (size_t)l * 64, n) != 0) {
				if (!moved) first = l;
				moved++;
			}
		if (moved) {
			bad_step = s;
			gpulog(LOG_ERR, thr_id,
				"flex: WIDTH-DEPENDENT at step %d/%d -- %u of %u overlapping lanes differ between "
				"a %u-lane and a %u-lane batch (first lane %u)",
				s, FLEX_PIPELINE_STEPS, moved, narrow, narrow, wide, first);
			/* Which lanes SURVIVED? If the survivors all share one algorithm,
			 * the per-bucket base offset is not reaching the stage; if they
			 * are scattered across algos, the fault is in the gather/scatter
			 * or the data, not the dispatch. */
			uint32_t okhist[FLEX_CORE_ALGO_COUNT] = { 0 };
			uint32_t badhist[FLEX_CORE_ALGO_COUNT] = { 0 };
			const uint8_t *row = c->h_core + (size_t)(s - 1) * c->threads;
			for (uint32_t l = 0; l < narrow; l++) {
				const uint8_t alg = row[l];
				if (alg >= FLEX_CORE_ALGO_COUNT) continue;
				if (memcmp(a + (size_t)l * 64, b + (size_t)l * 64, n) == 0) okhist[alg]++;
				else badhist[alg]++;
			}
			char ok[160] = { 0 }, bd[160] = { 0 };
			for (uint32_t k = 0; k < FLEX_CORE_ALGO_COUNT; k++) {
				char t1[16];
				snprintf(t1, sizeof t1, "%u ", okhist[k]);  strncat(ok, t1, sizeof(ok) - strlen(ok) - 1);
				snprintf(t1, sizeof t1, "%u ", badhist[k]); strncat(bd, t1, sizeof(bd) - strlen(bd) - 1);
			}
			gpulog(LOG_ERR, thr_id, "flex: per-algo MATCH counts (algo 0..13): %s", ok);
			gpulog(LOG_ERR, thr_id, "flex: per-algo DIFFER counts (algo 0..13): %s", bd);
		}
	}

	c->threads = real;
	free(a); free(b);
	if (!bad_step)
		gpulog(LOG_INFO, thr_id, "flex: width-invariant (%u vs %u lanes)", narrow, wide);
	return bad_step;
}

// ----------------------------------------------------------------------------
// Candidate screen audit (FLEX_VERIFY=1, =2 arms a negative control).
// The host only ever sees nonces the screen reported, so a screen that MISSES
// one produces no reject and no failed verify -- only slightly worse luck.
// ----------------------------------------------------------------------------
static uint32_t *flex_audit_buf[MAX_GPUS] = { 0 };
static uint32_t  flex_audit_cap[MAX_GPUS] = { 0 };

static bool flex_host_below(const uint32_t *h, const uint32_t *t)
{
	for (int i = 7; i >= 0; i--) {
		if (h[i] > t[i]) return false;
		if (h[i] < t[i]) return true;
	}
	return true;
}

static void flex_screen_audit(int thr_id, uint32_t throughput, uint32_t start_nonce,
	uint32_t *dh, const uint32_t *ptarget, uint32_t reported, int mode)
{
	if (flex_audit_cap[thr_id] < throughput) {
		free(flex_audit_buf[thr_id]);
		flex_audit_buf[thr_id] = (uint32_t*) malloc((size_t)throughput * 64);
		if (!flex_audit_buf[thr_id]) { flex_audit_cap[thr_id] = 0; return; }
		flex_audit_cap[thr_id] = throughput;
	}
	uint32_t *hb = flex_audit_buf[thr_id];
	cudaMemcpy(hb, dh, (size_t)throughput * 64, cudaMemcpyDeviceToHost);

	if (mode >= 2) hb[7] = 0;   /* negative control: lane 0 becomes unmissable */

	uint32_t hostcnt = 0, hostfirst = UINT32_MAX;
	bool sawreported = (reported == UINT32_MAX);
	for (uint32_t t = 0; t < throughput; t++) {
		if (flex_host_below(&hb[t * 16], ptarget)) {
			if (!hostcnt) hostfirst = start_nonce + t;
			if (start_nonce + t == reported) sawreported = true;
			hostcnt++;
		}
	}

	cuda_check_hash_suppl(thr_id, throughput, start_nonce, dh, 1);
	const uint32_t gpucnt = cuda_check_hash_count(thr_id);
	const char *ctl = (mode >= 2) ? "  [negative control armed - a MISS here is EXPECTED]" : "";

	if (gpucnt != hostcnt)
		gpulog(LOG_ERR, thr_id, "flex screen/host MISS: screen counted %u, host counted %u over [%08x,+%u) first=%08x%s",
			gpucnt, hostcnt, start_nonce, throughput, hostfirst, ctl);
	else if (!sawreported)
		gpulog(LOG_ERR, thr_id, "flex screen/host: screen reported %08x which the host does not find%s", reported, ctl);
	else if (opt_debug)
		gpulog(LOG_DEBUG, thr_id, "flex screen/host ok: %u candidates over [%08x,+%u)", hostcnt, start_nonce, throughput);
}

// ----------------------------------------------------------------------------
extern "C" int scanhash_flex(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	const int dev_id = device_map[thr_id];
	uint32_t _ALIGN(64) endiandata[20];

	/* Sized so the CPU re-verify actually fires at this algo's rates; the
	 * usual 0x00ff screen yields roughly one candidate every few hours. */
	if (opt_benchmark) {
		const char *env = getenv("FLEX_BENCH_TARGET");
		ptarget[7] = env ? (uint32_t) strtoul(env, NULL, 0) : 0x0003ffff;
	}

	if (!init_done[thr_id]) {
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		}

		{
			char det[FLEX_SELFTEST_DETAIL];
			if (flex_self_test(det, sizeof det))
				gpulog(LOG_INFO, thr_id, "flex: CPU self-test OK (%s)", det);
			else
				gpulog(LOG_ERR, thr_id, "flex: CPU SELF-TEST FAILED -- %s", det);
		}

		/* Lane count is sized from the MEAN per-lane CryptoNight cost, not the
		 * maximum. The variant is chosen per nonce, so a batch is a mix:
		 * reserving the largest variant's 2 MiB for every lane leaves ~5 of 6
		 * lanes sitting on memory they never touch, and costs ~2.7x the
		 * achievable batch. flex_pipeline_run() trims the tail on the rare
		 * batch whose mix exceeds the budget.
		 *
		 * The margin covers the sampling spread: the total is a sum of `lanes`
		 * independent draws with sd ~627 KiB, so at these lane counts the
		 * deviation is well under 1%. */
		size_t freeMem = 0, totalMem = 0;
		cudaMemGetInfo(&freeMem, &totalMem);
		const size_t reserve = 900ULL << 20;   /* stage buffers + check + driver */
		size_t budget = (freeMem > reserve) ? (freeMem - reserve) : (freeMem / 2);

		/* ~700 B/lane of ctx + chain state on top of the scratchpad. */
		const size_t per_lane = (size_t)(FLEX_CN_MEM_MEAN * 1.06) + 1024;
		uint32_t throughput = (uint32_t)(budget / per_lane);

		/* Cap the live batch below the VRAM maximum. A launch cannot be preempted,
		 * so a batch still running when a new job arrives was chained against a
		 * retired header and is wasted entirely. Measured on one card against one
		 * pool's job cadence, so treat it as unknown elsewhere; the batch duration,
		 * not the lane count, is what matters. An explicit -i overrides it. */
		const uint32_t lane_cap = 8192;
		if (gpus_intensity[thr_id] > 0) {
			if (gpus_intensity[thr_id] < throughput)
				throughput = gpus_intensity[thr_id];
		} else if (throughput > lane_cap) {
			throughput = lane_cap;
		}
		throughput = (throughput / 128) * 128;
		if (throughput < 128) throughput = 128;

		/* The scratchpad arena the packer lays groups into. */
		const size_t scratch = (size_t)throughput * (size_t)(FLEX_CN_MEM_MEAN * 1.06);

		gpulog(LOG_INFO, thr_id, "flex: %u lanes, %.0f MiB CryptoNight arena (variable stride, mean %u KiB/lane)",
			throughput, (double)scratch / (1024.0 * 1024.0), FLEX_CN_MEM_MEAN >> 10);

		flex_stage_init_all(thr_id, throughput);
		if (!flex_pipeline_init(&ctx[thr_id], thr_id, throughput, scratch)) {
			gpulog(LOG_ERR, thr_id, "flex: out of memory sizing %u lanes", throughput);
			return -1;
		}
		cuda_check_cpu_init(thr_id, throughput);

		gpulog(LOG_INFO, thr_id, "flex: running startup guards (kernels + full chain)...");
		if (!flex_guard_races(thr_id, throughput))
			gpulog(LOG_INFO, thr_id, "flex: core kernels race-free");
		if (!flex_guard_chain(thr_id, throughput))
			gpulog(LOG_INFO, thr_id, "flex: GPU==CPU on the full chain");
		flex_guard_filtered(thr_id, throughput);
		if (getenv("FLEX_GUARD_WIDTH"))
			flex_guard_width(thr_id, throughput);
		gpulog(LOG_INFO, thr_id, "flex: guards complete, starting mining");

		init_done[thr_id] = true;
	}

	flex_ctx *c = &ctx[thr_id];

	for (int k = 0; k < 19; k++)
		be32enc(&endiandata[k], pdata[k]);
	be32enc(&endiandata[19], pdata[19]);

	if (work_restart[thr_id].restart) {
		*hashes_done = 0;
		return 0;
	}

	cuda_check_cpu_setTarget(ptarget);

	/* ---- the nonce filter: scan a span, chain only the cheap-CN-triple nonces ----
	 *
	 * The filter keeps ~3.7% of nonces, so the span is ~31x the lane count.
	 * Every accepted nonce is ordinary work; the saving is that its three
	 * CryptoNight rounds total 4 iteration units against an unfiltered mean
	 * of 7.388. Consensus-legal, because any nonce is a valid nonce.
	 *
	 * The nonce pre-filter is DISABLED by default: it benchmarks well but fails
	 * the host re-verify in live mining, delivering less accepted work than the
	 * plain path. Set FLEX_NONCE_FILTER=1 to re-enable for debugging. */
	static int flex_use_filter = -1;
	if (flex_use_filter < 0) {
		const char *ev = getenv("FLEX_NONCE_FILTER");
		flex_use_filter = (ev && atoi(ev)) ? 1 : 0;
		if (flex_use_filter)
			gpulog(LOG_WARNING, thr_id,
			       "flex: nonce filter ENABLED -- known to fail the host re-verify; debugging only");
	}

	const uint32_t scan_base = pdata[19];
	const uint32_t span = flex_use_filter
		? (uint32_t)c->threads * FLEX_FILTER_SPAN_PER_LANE
		: c->threads;               /* disabled: one nonce per lane, no scan */
	uint32_t yield = 0;
	c->span = span;
	if (flex_use_filter) {
		cudaMemset(c->d_count, 0, sizeof(uint32_t));
		flex_filter_cpu(span, pdata[19], c->threads, c->d_nonces, c->d_count);
		cudaMemcpy(&yield, c->d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
	} else {
		yield = c->threads;
	}

	/* The kernel keeps counting past the buffer so this can legitimately
	 * exceed the capacity; clamp, and say so rather than silently believing
	 * the span was exactly right. */
	if (yield > c->threads) {
		if (opt_debug)
			applog(LOG_DEBUG, "flex: filter saturated, %u accepted for %u lanes",
			       yield, c->threads);
		yield = c->threads;
	}
	if (yield < FLEX_CN_TPB) {
		/* Nothing worth launching for. Still consume the span. */
		pdata[19] += span;
		if (pdata[19] > max_nonce || pdata[19] < first_nonce) pdata[19] = max_nonce;
		*hashes_done = 0;
		return 0;
	}
	yield = (yield / FLEX_CN_TPB) * FLEX_CN_TPB;
	if (flex_use_filter)
		cudaMemcpy(c->h_nonces, c->d_nonces, sizeof(uint32_t) * yield, cudaMemcpyDeviceToHost);

	flex_pipeline_run(c, thr_id, endiandata, pdata, pdata[19], -1,
	                  flex_use_filter ? c->d_nonces : NULL, flex_use_filter ? yield : 0);

	/* The pipeline may have trimmed the batch to fit this mix of CN
	 * variants, so the screen and the nonce cursor must both use the lanes
	 * ACTUALLY hashed. Screening  here would read past the end of
	 * the batch and report nonces that were never computed. */
	const uint32_t hashed = c->lanes;

	/* The lanes are SCATTERED in nonce space now, so the shared screen --
	 * which reports startNounce + thread -- must be given a base of 0. What it
	 * returns is then a LANE INDEX, and the nonce is h_nonces[lane]. Passing
	 * pdata[19] here would report nonces that were never hashed. */
	const uint32_t lane0 = cuda_check_hash(thr_id, hashed,
	                                       flex_use_filter ? 0 : pdata[19], c->d_hash);
	work->nonces[0] = (!flex_use_filter || lane0 == UINT32_MAX) ? lane0
	                : (lane0 < hashed ? c->h_nonces[lane0] : UINT32_MAX);

	static int flex_verify = -1;
	if (flex_verify < 0) {
		const char *ev = getenv("FLEX_VERIFY");
		flex_verify = ev ? atoi(ev) : 0;
	}
	/* The audit rebuilds each lane's nonce as start + lane, which is only
	 * true on the unfiltered path. Under the nonce filter the lanes are scattered, so it is
	 * reported as UNAVAILABLE rather than run against the wrong nonces --
	 * a silently-wrong instrument is worse than an absent one, and its silence
	 * would read as a pass. */
	if (flex_verify) {
		static bool warned = false;
		if (!warned) {
			gpulog(LOG_WARNING, thr_id,
			       "flex: FLEX_VERIFY screen audit is unavailable under the nonce filter "
			       "(scattered nonces); use the chain guard instead");
			warned = true;
		}
	}

	if (work->nonces[0] != UINT32_MAX) {
		uint32_t _ALIGN(64) vhash[8];
		be32enc(&endiandata[19], work->nonces[0]);
		flex_hash(vhash, endiandata);

		if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
			work->valid_nonces = 1;
			work_set_target_ratio(work, vhash);
			/* Lane index again, for the same reason as nonces[0]. */
			const uint32_t lane1 = cuda_check_hash_suppl(thr_id, hashed,
			                                       flex_use_filter ? 0 : pdata[19], c->d_hash, 1);
			work->nonces[1] = (!flex_use_filter || lane1 == UINT32_MAX) ? lane1
			                : (lane1 < hashed ? c->h_nonces[lane1] : UINT32_MAX);
			const uint32_t found = cuda_check_hash_count(thr_id);
			if (work->nonces[1] != UINT32_MAX) {
				be32enc(&endiandata[19], work->nonces[1]);
				flex_hash(vhash, endiandata);
				/* The screen compares the top word only, so guard the second
				 * nonce exactly like the first. */
				if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
					bn_set_target_ratio(work, vhash, 1);
					work->valid_nonces++;
				}
			}

			/* the nonce cursor under a filter: the cursor advances by the SCANNED SPAN,
			 * never by a found nonce. The old max(nonce)+1 form is wrong here in
			 * BOTH directions -- the accepted nonces are unordered, so the max is
			 * not the high-water mark of the scan, and resuming from it would
			 * re-scan most of a span we have already rejected while also skipping
			 * any accepted nonce that happened to sort below it. */
			uint32_t resume = flex_use_filter ? (scan_base + span) : (first_nonce + hashed);
			if (resume > max_nonce || resume < first_nonce)
				resume = max_nonce;
			pdata[19] = resume;

			/* CHAINED lanes, not the span. Counting the ~31x scanned nonces
			 * would inflate the displayed rate by that factor -- the rinhash
			 * defect. A scanned-and-rejected nonce produces no digest and so has
			 * no chance of being a share; it is not hashing. */
			*hashes_done = hashed;
			return work->valid_nonces;
		} else {
			gpu_increment_reject(thr_id);
			if (!opt_quiet)
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
		}
	}

	pdata[19] = flex_use_filter ? (scan_base + span) : (scan_base + hashed);
	if (pdata[19] > max_nonce || pdata[19] < first_nonce)
		pdata[19] = max_nonce;
	*hashes_done = hashed;      /* chained, not scanned -- see above */
	return 0;
}

extern "C" void free_flex(int thr_id)
{
	if (!init_done[thr_id])
		return;

	cudaDeviceSynchronize();

	flex_pipeline_free(&ctx[thr_id]);
	flex_stage_free_all(thr_id);
	cuda_check_cpu_free(thr_id);

	free(flex_audit_buf[thr_id]);
	flex_audit_buf[thr_id] = NULL;
	flex_audit_cap[thr_id] = 0;

	cudaDeviceSynchronize();
	init_done[thr_id] = false;
}
