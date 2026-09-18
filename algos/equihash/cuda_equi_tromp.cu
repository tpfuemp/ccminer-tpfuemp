// Equihash 144/5 GPU solver wrapper (tromp/equihash), ccminer integration.
//
// ===========================================================================
// FOR TECHNICAL STUDY ONLY. This integrates John Tromp's *reference* Equihash
// solver to demonstrate a correct, open, in-fork 144/5 mining path (validated
// against real chain data and accepted live). It is intentionally NOT
// performance-optimized: throughput is a fraction of dedicated Equihash miners
// and it is not intended for competitive/production mining.
// ===========================================================================
//
// This translation unit imports John Tromp's (n,k)-parameterized Equihash CUDA
// solver (MIT) and exposes a small C API (tromp144_init/solve/free). It is a
// SEPARATE solver from equi/cuda_equi.cu (djeZo's fast 200/9 path): the two are
// kept apart because tromp defines the same symbol names (`equi`, `digitK`,
// `digit_1..8`, blake helpers). To avoid link clashes the entire tromp code is
// pulled into `namespace tromp144` below.
//
// Build (see ccminer.vcxproj): -DWN=144 -DWK=5 -DRESTBITS=4  (NO -DXINTREE,
// NO -DUNROLL — the generic digitH/digitO/digitE/digitK path solves 144/5).
//
// Host BLAKE2b: instead of tromp's SSE blake2b.cpp (which #errors on MSVC for
// lack of __SSE2__ and would clash with sph/blake2b), we reuse the fork's
// eq_blake2b_* host impl. This is sound because equi/blake2/blake2.h and tromp's
// blake/blake2.h define a byte-identical blake2b_state/blake2b_param.

// ---- system + fork includes at GLOBAL scope --------------------------------
// (Included here so the guarded re-includes inside equi_tromp.h become no-ops
//  and never end up nested inside the namespace.)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "blake2/blake2.h"   // blake2b_state/param + eq_blake2b_* (extern "C")
#include "cuda_equi_tromp.h"

// ---- tromp solver, fully namespaced ----------------------------------------
namespace tromp144 {
#include "equi_tromp.h"          // params, proof/u32/uchar types, setheader, verify
#include "blake2b_tromp.cuh"     // __device__ blake2b_gpu_hash
#include "equi_miner_tromp.cuh"  // equi struct + kernels (digitH/O/E/K); main() #if 0'd

// Persistent solver context (mirrors the setup in tromp's original main()).
struct solver_ctx {
	equi   heq;        // host-side mirror; holds the device pointers below
	equi  *device_eq;
	u32   *heap0, *heap1;
	proof *sols;       // host readback buffer
	u32    nthreads, tpb;
	// Solutions the device found and the MAXSOLS cap threw away on the last
	// solve. `nsols` counts every candidate; only the first MAXSOLS are stored,
	// so the surplus is lost share value and nothing used to report it.
	u32    sols_dropped;
#ifdef EQ_DIAG
	u32   *diag_ns;    // NBUCKETS host buffer for the per-round fill readback
#endif
	solver_ctx(u32 n) : heq(n), device_eq(0), heap0(0), heap1(0), sols(0),
	                    nthreads(n), tpb(0), sols_dropped(0)
#ifdef EQ_DIAG
	                    , diag_ns(0)
#endif
	                    {}
};

#ifdef EQ_DIAG
// Per-round bucket-fill / discard census. Diagnostic build only: it inserts a
// blocking 4 MB D2H copy between kernel launches, so it must not ship enabled.
//
// No kernel-side drop counter is needed, deliberately. Every scatter does
//   xorslot = atomicAdd(&nslots[..][bucket], 1);  if (xorslot >= NSLOTS) continue;
// so the counter already records ATTEMPTS, not stores, and the discarded pairs
// are exactly sum(max(0, ns[b] - NSLOTS)). In-kernel counters would add an
// atomic to the drop path of kernels whose limiter is LSU issue rate.
//
// MUST be called after the producing kernel and BEFORE the consuming one:
// eq_take() zeroes each counter as it reads it.
static void diag_round(solver_ctx *c, u32 r) {
	if (!c->diag_ns)
		c->diag_ns = (u32 *)malloc(NBUCKETS * sizeof(u32));
	if (!c->diag_ns)
		return;
	checkCudaErrors(cudaMemcpy(c->diag_ns, c->heq.nslots[r & 1],
	                           NBUCKETS * sizeof(u32), cudaMemcpyDeviceToHost));
	unsigned long long attempts = 0, stored = 0;
	u32 overfull = 0, maxfill = 0;
	for (u32 b = 0; b < NBUCKETS; b++) {
		const u32 n = c->diag_ns[b];
		attempts += n;
		stored   += n < NSLOTS ? n : NSLOTS;
		if (n > NSLOTS) overfull++;
		if (n > maxfill) maxfill = n;
	}
	const unsigned long long dropped = attempts - stored;
	printf("eq-diag r%u: attempts %llu, stored %llu, DROPPED %llu (%.3f%%), "
	       "buckets overfull %u/%u (%.3f%%), max fill %u of %u, mean %.2f\n",
	       r, attempts, stored, dropped,
	       attempts ? 100.0 * (double)dropped / (double)attempts : 0.0,
	       overfull, NBUCKETS, 100.0 * (double)overfull / (double)NBUCKETS,
	       maxfill, NSLOTS, (double)attempts / (double)NBUCKETS);
	c->heq.showbsizes(r);   // histogram, if HIST/SPARK/LOGSPARK is also defined
}
#endif

static solver_ctx *ctx_init(u32 nthreads, u32 tpb) {
	if (!tpb) // default threads-per-block to roughly sqrt(nthreads)
		for (tpb = 1; tpb * tpb < nthreads; tpb *= 2) ;
	solver_ctx *c = new solver_ctx(nthreads);
	c->tpb = tpb;

	checkCudaErrors(cudaMalloc((void**)&c->heap0, sizeof(digit0)));
	checkCudaErrors(cudaMalloc((void**)&c->heap1, sizeof(digit1)));
	for (u32 r = 0; r < WK; r++)
		if ((r & 1) == 0)
			c->heq.hta.trees0[r/2] = (bucket0 *)(c->heap0 + r/2);
		else
			c->heq.hta.trees1[r/2] = (bucket1 *)(c->heap1 + r/2);

	checkCudaErrors(cudaMalloc((void**)&c->heq.nslots, 2 * NBUCKETS * sizeof(u32)));
	checkCudaErrors(cudaMalloc((void**)&c->heq.sols, MAXSOLS * sizeof(proof)));
	checkCudaErrors(cudaMalloc((void**)&c->device_eq, sizeof(equi)));
	c->sols = (proof *)malloc(MAXSOLS * sizeof(proof));
	return c;
}

static int ctx_solve(solver_ctx *c, const char *headernonce, const char *personal,
                     tromp144_emit_fn emit, void *ud) {
	const u32 nt = c->nthreads, tpb = c->tpb;

	c->heq.setheadernonce(headernonce, HEADERNONCELEN, personal);
	// Fresh bucket counts for this run.
	checkCudaErrors(cudaMemset((void*)c->heq.nslots, 0, 2 * NBUCKETS * sizeof(u32)));
	checkCudaErrors(cudaMemcpy(c->device_eq, &c->heq, sizeof(equi), cudaMemcpyHostToDevice));

	// heq holds the same device pointers as the uploaded struct; passing them as
	// kernel args keeps them in the global address space (see equi_miner_tromp.cuh).
	//
	// digitH uses a grid-stride loop, so its grid can exceed eq->nthreads;
	// at nt/tpb=64 blocks the GPU runs ~19% occupied (profiled 2026-07-02).
	digitH<<<8 * (nt/tpb), tpb>>>(c->device_eq, c->heq.hta, c->heq.nslots);
#ifdef EQ_DIAG
	diag_round(c, 0);
#endif
#if WN == 144 && WK == 5 && BUCKBITS == 20 && RESTBITS == 4 && !defined(XINTREE)
	// warp-per-bucket collision kernels (see equi_miner_tromp.cuh): one warp
	// stages one bucket coalesced and processes its pairs in parallel via
	// ballot masks. Grid swept: 128 blocks best (64: -5%, 256: -1%, 512: -2%);
	// 512 warps = 512 buckets in flight (~0.7MB staged, fits L2 easily).
	// digitOT/ET kept above as reference.
	digitWB<1><<<128, EQ_WB_TPB>>>(c->heq.hta, c->heq.nslots);
#ifdef EQ_DIAG
	diag_round(c, 1);
#endif
	digitWB<2><<<128, EQ_WB_TPB>>>(c->heq.hta, c->heq.nslots);
#ifdef EQ_DIAG
	diag_round(c, 2);
#endif
	digitWB<3><<<128, EQ_WB_TPB>>>(c->heq.hta, c->heq.nslots);
#ifdef EQ_DIAG
	diag_round(c, 3);
#endif
	digitWB<4><<<128, EQ_WB_TPB>>>(c->heq.hta, c->heq.nslots);
#ifdef EQ_DIAG
	diag_round(c, 4);
#endif
#else
	for (u32 r = 1; r < WK; r++)
		r & 1 ? digitO<<<nt/tpb, tpb>>>(c->device_eq, r)
		      : digitE<<<nt/tpb, tpb>>>(c->device_eq, r);
#endif
	digitK<<<nt/tpb, tpb>>>(c->device_eq);

	checkCudaErrors(cudaMemcpy(&c->heq, c->device_eq, sizeof(equi), cudaMemcpyDeviceToHost));
	const u32 maxsols = c->heq.nsols < MAXSOLS ? c->heq.nsols : MAXSOLS;
	// nsols counts every candidate the device produced; sols[] holds the first
	// MAXSOLS. Record the surplus rather than discarding it silently.
	c->sols_dropped = c->heq.nsols > MAXSOLS ? c->heq.nsols - MAXSOLS : 0;
	checkCudaErrors(cudaMemcpy(c->sols, c->heq.sols, maxsols * sizeof(proof),
	                           cudaMemcpyDeviceToHost));

	int found = 0;
	for (u32 s = 0; s < maxsols; s++) {
		if (duped(c->sols[s]))
			continue;
		if (emit)
			emit(ud, c->sols[s], PROOFSIZE);
		found++;
	}
	return found;
}

static void ctx_free(solver_ctx *c) {
	if (c->device_eq)  cudaFree(c->device_eq);
	if (c->heq.nslots) cudaFree(c->heq.nslots);
	if (c->heq.sols)   cudaFree(c->heq.sols);
	if (c->heap0)      cudaFree(c->heap0);
	if (c->heap1)      cudaFree(c->heap1);
	if (c->sols)       free(c->sols);
#ifdef EQ_DIAG
	if (c->diag_ns)    free(c->diag_ns);
#endif
	delete c;
}

} // namespace tromp144

// ---- C API -----------------------------------------------------------------
extern "C" void *tromp144_init(unsigned nthreads, unsigned tpb) {
	return (void *)tromp144::ctx_init(nthreads, tpb);
}

extern "C" int tromp144_solve(void *ctx, const char *headernonce, const char *personal,
                              tromp144_emit_fn emit, void *ud) {
	if (!ctx)
		return -1;
	return tromp144::ctx_solve((tromp144::solver_ctx *)ctx, headernonce, personal, emit, ud);
}

extern "C" unsigned tromp144_sols_dropped(void *ctx) {
	return ctx ? ((tromp144::solver_ctx *)ctx)->sols_dropped : 0;
}

extern "C" void tromp144_free(void *ctx) {
	if (ctx)
		tromp144::ctx_free((tromp144::solver_ctx *)ctx);
}

extern "C" int tromp144_verify(const char *headernonce, const char *personal,
                               const uint32_t *indices) {
	return tromp144::verify((tromp144::u32 *)indices, headernonce,
	                        HEADERNONCELEN, personal);
}
