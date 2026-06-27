/**
 * SHA256Dv (Veil) - double SHA-256 with a coin-specific Stratum protocol.
 *
 * The hash is ordinary SHA-256d over an 80-byte "stage2" buffer:
 *
 *   off 0  : version  (LE, 4)
 *   off 4  : midstate (BE, 32)  - supplied pre-hashed by the pool
 *   off 36 : merkle   (LE, 32)  - byte-reversed merkle_be from the pool
 *   off 68 : ntime    (LE, 4)
 *   off 72 : nonce_lo (LE, 4)   - the searched 32-bit word
 *   off 76 : nonce_hi (LE, 4)   - base high word of the 64-bit nonce
 *
 *   hash = SHA256( SHA256( stage2 ) )
 *
 * The first 64-byte SHA-256 block (version + midstate + merkle[0:28]) is fixed
 * for a job, so its midstate is precomputed on the host once per job; the GPU
 * kernel only runs the second block (which holds nonce_lo) plus the second
 * SHA-256. nonce_lo sits at word index 2 of the second block (byte 72), with
 * nonce_hi fixed at word index 3 (byte 76).
 *
 * Bespoke notify/submit live in util.cpp (stratum_notify) and ccminer.cpp
 * (submit path); see also algo/sha/sha256dv.c in cpuminer-opt.
 */

#include "miner.h"
#include "cuda_helper.h"

#include <stdint.h>
#include <string.h>

// ---------------------------------------------------------------------------
// SHA-256 building blocks (shared host + device constants)
// ---------------------------------------------------------------------------

static const uint32_t h_H256[8] = {
	0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
	0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

static const uint32_t h_K256[64] = {
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
	0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
	0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
	0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
	0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

__device__ __constant__ static uint32_t d_K256[64];

// Per-job constants uploaded via sha256dv_setBlock().
__device__ __constant__ static uint32_t d_midstate[8]; // SHA-256 midstate of stage2 block 1
__device__ __constant__ static uint32_t d_block2[4];    // w0=merkle tail, w1=ntime, [2] unused, w3=nonce_hi
__device__ __constant__ static uint32_t d_target[8];    // work->target (index 7 = most significant)

static uint32_t *d_resNonce[MAX_GPUS];
static bool      init_done[MAX_GPUS] = { 0 };

// A Veil job's identity lives in work->veil_* (midstate/merkle/ntime/nonce_hi),
// which miner_thread's data[]-based job-change check cannot see. As a result
// stratum_gen_work resets the nonce_lo cursor (work->data[19]) to 0 on every
// scantime regen of an *unchanged* job, which would rescan it from zero and
// resubmit duplicate shares. We therefore track the cursor ourselves, keyed on a
// signature of the pool-fixed inputs: resume where we left off, and stop once a
// job's 32-bit nonce_lo space is exhausted (wait for genuinely new work).
static uint32_t dv_cursor[MAX_GPUS]  = { 0 };
static uint32_t dv_sig[MAX_GPUS]     = { 0 };
static bool     dv_sig_set[MAX_GPUS] = { 0 };
static bool     dv_done[MAX_GPUS]    = { 0 };

// ---------------------------------------------------------------------------
// Host SHA-256 (for midstate precompute, CPU validation and self-test)
// ---------------------------------------------------------------------------

#define ROTR32(x,n) (((x) >> (n)) | ((x) << (32 - (n))))

static void host_sha256_transform(uint32_t state[8], const uint32_t in[16])
{
	uint32_t w[64];
	for (int i = 0; i < 16; i++) w[i] = in[i];
	for (int i = 16; i < 64; i++) {
		uint32_t s0 = ROTR32(w[i-15], 7) ^ ROTR32(w[i-15], 18) ^ (w[i-15] >> 3);
		uint32_t s1 = ROTR32(w[i-2], 17) ^ ROTR32(w[i-2], 19) ^ (w[i-2] >> 10);
		w[i] = w[i-16] + s0 + w[i-7] + s1;
	}
	uint32_t a=state[0],b=state[1],c=state[2],d=state[3],e=state[4],f=state[5],g=state[6],h=state[7];
	for (int i = 0; i < 64; i++) {
		uint32_t S1 = ROTR32(e,6) ^ ROTR32(e,11) ^ ROTR32(e,25);
		uint32_t ch = (e & f) ^ (~e & g);
		uint32_t t1 = h + S1 + ch + h_K256[i] + w[i];
		uint32_t S0 = ROTR32(a,2) ^ ROTR32(a,13) ^ ROTR32(a,22);
		uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
		uint32_t t2 = S0 + maj;
		h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
	}
	state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
	state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
}

// Double SHA-256 of an 80-byte stage2 buffer -> 8 LE words (index 7 = MSW,
// matching the target convention).
static void host_sha256dv(uint32_t hash_le[8], const uint8_t stage2[80])
{
	uint32_t blk[16], st[8];

	// First SHA-256 (two blocks).
	memcpy(st, h_H256, sizeof(st));
	for (int i = 0; i < 16; i++) blk[i] = be32dec(stage2 + i*4);
	host_sha256_transform(st, blk);
	for (int i = 0; i < 4;  i++) blk[i] = be32dec(stage2 + 64 + i*4);
	blk[4] = 0x80000000;
	for (int i = 5; i < 15; i++) blk[i] = 0;
	blk[15] = 640;
	host_sha256_transform(st, blk);

	// Second SHA-256 over the 32-byte first digest.
	uint32_t st2[8];
	memcpy(st2, h_H256, sizeof(st2));
	for (int i = 0; i < 8;  i++) blk[i] = st[i];
	blk[8] = 0x80000000;
	for (int i = 9; i < 15; i++) blk[i] = 0;
	blk[15] = 256;
	host_sha256_transform(st2, blk);

	for (int i = 0; i < 8; i++) hash_le[i] = st2[i];
}

// Build the 80-byte stage2 buffer from the work fields for a given nonce pair.
static void sha256dv_build_stage2(uint8_t out[80], const struct work *work,
                                  uint32_t nonce_lo, uint32_t nonce_hi)
{
	uint8_t *p = out;
	le32enc(p, work->data[0]); p += 4;                  // version (LE)
	memcpy(p, work->veil_midstate_be, 32); p += 32;     // midstate (BE)
	for (int i = 0; i < 32; i++) p[i] = work->veil_merkle_be[31 - i];
	p += 32;                                            // merkle (LE = reversed BE)
	le32enc(p, work->veil_ntime); p += 4;               // ntime (LE)
	le32enc(p, nonce_lo); p += 4;                       // nonce_lo (LE)
	le32enc(p, nonce_hi);                               // nonce_hi (LE)
}

static bool sha256dv_meets_target(const uint32_t *hash, const uint32_t *target)
{
	for (int i = 7; i >= 0; i--) {
		if (hash[i] > target[i]) return false;
		if (hash[i] < target[i]) return true;
	}
	return true;
}

// ---------------------------------------------------------------------------
// Device SHA-256
// ---------------------------------------------------------------------------

__device__ __forceinline__
static void dev_sha256_transform(uint32_t state[8], const uint32_t in[16])
{
	uint32_t w[64];
	#pragma unroll
	for (int i = 0; i < 16; i++) w[i] = in[i];
	#pragma unroll
	for (int i = 16; i < 64; i++) {
		uint32_t s0 = ROTR32(w[i-15], 7) ^ ROTR32(w[i-15], 18) ^ (w[i-15] >> 3);
		uint32_t s1 = ROTR32(w[i-2], 17) ^ ROTR32(w[i-2], 19) ^ (w[i-2] >> 10);
		w[i] = w[i-16] + s0 + w[i-7] + s1;
	}
	uint32_t a=state[0],b=state[1],c=state[2],d=state[3],e=state[4],f=state[5],g=state[6],h=state[7];
	#pragma unroll
	for (int i = 0; i < 64; i++) {
		uint32_t S1 = ROTR32(e,6) ^ ROTR32(e,11) ^ ROTR32(e,25);
		uint32_t ch = (e & f) ^ (~e & g);
		uint32_t t1 = h + S1 + ch + d_K256[i] + w[i];
		uint32_t S0 = ROTR32(a,2) ^ ROTR32(a,13) ^ ROTR32(a,22);
		uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
		uint32_t t2 = S0 + maj;
		h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
	}
	state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
	state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
}

__global__
void sha256dv_gpu_hash(uint32_t threads, uint32_t startNonce, uint32_t *resNonce)
{
	const uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < threads)
	{
		const uint32_t nonce_lo = startNonce + tid;

		// Second block of the first SHA-256 (continues from the job midstate).
		uint32_t in[16];
		in[0] = d_block2[0];                 // merkle tail
		in[1] = d_block2[1];                 // ntime
		in[2] = cuda_swab32(nonce_lo);       // nonce_lo message word
		in[3] = d_block2[3];                 // nonce_hi
		in[4] = 0x80000000;
		#pragma unroll
		for (int i = 5; i < 15; i++) in[i] = 0;
		in[15] = 640;

		uint32_t st[8];
		#pragma unroll
		for (int i = 0; i < 8; i++) st[i] = d_midstate[i];
		dev_sha256_transform(st, in);

		// Second SHA-256 over the 32-byte first digest.
		uint32_t in2[16];
		#pragma unroll
		for (int i = 0; i < 8; i++) in2[i] = st[i];
		in2[8] = 0x80000000;
		#pragma unroll
		for (int i = 9; i < 15; i++) in2[i] = 0;
		in2[15] = 256;

		uint32_t st2[8] = {
			0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
			0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
		};
		dev_sha256_transform(st2, in2);

		// MSW-first compare against target.
		bool ok = true;
		#pragma unroll
		for (int i = 7; i >= 0; i--) {
			if (st2[i] > d_target[i]) { ok = false; break; }
			if (st2[i] < d_target[i]) break;
		}
		if (ok)
			resNonce[0] = nonce_lo;
	}
}

__host__
static void sha256dv_setBlock(const uint32_t midstate[8], const uint32_t block2[4],
                              const uint32_t target[8])
{
	cudaMemcpyToSymbol(d_midstate, midstate, 8 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_block2,   block2,   4 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_target,   target,   8 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
}

// ---------------------------------------------------------------------------
// Startup self-test: GPU kernel must reproduce the host SHA-256d on a fixed
// input (the host path uses an independent transform; per-share validation in
// scanhash re-checks every candidate against it as well).
// ---------------------------------------------------------------------------
static bool sha256dv_self_test(int thr_id)
{
	struct work w;
	memset(&w, 0, sizeof(w));
	w.data[0] = 0x20000000;
	for (int i = 0; i < 32; i++) { w.veil_midstate_be[i] = (uint8_t)(i + 1); w.veil_merkle_be[i] = (uint8_t)(0xa0 + i); }
	w.veil_ntime = 0x66000000;
	const uint32_t nonce_hi = 0x12345678, nonce_lo = 0x0000abcd;

	uint8_t stage2[80];
	uint32_t exp[8], ms[8], blk[16], block2[4];

	sha256dv_build_stage2(stage2, &w, nonce_lo, nonce_hi);
	host_sha256dv(exp, stage2);

	// midstate of block 1, plus the block-2 fixed words.
	memcpy(ms, h_H256, sizeof(ms));
	for (int i = 0; i < 16; i++) blk[i] = be32dec(stage2 + i*4);
	host_sha256_transform(ms, blk);
	block2[0] = be32dec(stage2 + 64);
	block2[1] = be32dec(stage2 + 68);
	block2[2] = 0;
	block2[3] = be32dec(stage2 + 76);

	// Use the expected host digest as the target: the kernel reports the test
	// nonce only if its GPU-computed hash is <= exp. Since the host hash equals
	// exp exactly, a correct kernel reports it; a kernel that computes a larger
	// digest fails the test (got stays UINT32_MAX).
	sha256dv_setBlock(ms, block2, exp);

	cudaMemset(d_resNonce[thr_id], 0xff, sizeof(uint32_t));
	sha256dv_gpu_hash <<< 1, 1 >>> (1, nonce_lo, d_resNonce[thr_id]);
	uint32_t got = UINT32_MAX;
	cudaMemcpy(&got, d_resNonce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);

	(void)nonce_hi;
	return got == nonce_lo;
}

// ---------------------------------------------------------------------------
// scanhash
// ---------------------------------------------------------------------------

extern "C" int scanhash_sha256dv(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const int dev_id = device_map[thr_id];

	// --benchmark has no Stratum job; synthesize a deterministic one so the
	// kernel (and startup self-test) can run and report a hashrate.
	if (opt_benchmark && !work->veil_sha256dv) {
		work->veil_sha256dv = true;
		work->data[0] = 0x20000000;
		for (int i = 0; i < 32; i++) {
			work->veil_midstate_be[i] = (uint8_t)(i + 1);
			work->veil_merkle_be[i]   = (uint8_t)(0xa0 + i);
		}
		work->veil_ntime    = 0x66000000;
		work->veil_nonce_hi = 0;
	}

	if (!work->veil_sha256dv) { *hashes_done = 0; return 0; }

	// Live Veil jobs manage their own nonce_lo cursor (see dv_cursor note); only
	// --benchmark uses the host-supplied pdata[19]/max_nonce window directly.
	const bool dv_track = !opt_benchmark;

	uint32_t throughput = cuda_default_throughput(thr_id, 1U << 23);
	if (init_done[thr_id] && !dv_track) throughput = min(throughput, max_nonce - pdata[19]);

	if (opt_benchmark)
		ptarget[7] = 0x000000ff;

	if (!init_done[thr_id]) {
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads",
			throughput2intensity(throughput), throughput);

		cudaMemcpyToSymbol(d_K256, h_K256, sizeof(h_K256), 0, cudaMemcpyHostToDevice);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_resNonce[thr_id], 2 * sizeof(uint32_t)), 0);

		if (!sha256dv_self_test(thr_id))
			gpulog(LOG_WARNING, thr_id, "SHA256Dv self-test FAILED (GPU/CPU mismatch)");

		init_done[thr_id] = true;
	}

	// Per-job constants: stage2 block-1 midstate + block-2 fixed words.
	uint8_t  stage2[80];
	uint32_t ms[8], blk[16], block2[4];
	const uint32_t nonce_hi = work->veil_nonce_hi;

	sha256dv_build_stage2(stage2, work, pdata[19], nonce_hi);
	memcpy(ms, h_H256, sizeof(ms));
	for (int i = 0; i < 16; i++) blk[i] = be32dec(stage2 + i*4);
	host_sha256_transform(ms, blk);
	block2[0] = be32dec(stage2 + 64);
	block2[1] = be32dec(stage2 + 68);
	block2[2] = 0;
	block2[3] = be32dec(stage2 + 76);

	sha256dv_setBlock(ms, block2, ptarget);

	// Resolve the nonce_lo cursor for this job. Signature = FNV-1a over the
	// pool-fixed inputs (midstate, nonce_hi, ntime, target). A changed signature
	// is a genuinely new job: restart from 0. An unchanged signature resumes the
	// saved cursor, ignoring any spurious reset of pdata[19] by stratum_gen_work.
	const uint64_t scan_end = dv_track ? 0x100000000ull : (uint64_t)max_nonce;
	if (dv_track) {
		uint32_t sig = 2166136261u;
		for (int i = 0; i < 32; i++) { sig ^= work->veil_midstate_be[i]; sig *= 16777619u; }
		sig ^= nonce_hi;           sig *= 16777619u;
		sig ^= work->veil_ntime;   sig *= 16777619u;
		for (int i = 0; i < 8; i++) { sig ^= ptarget[i]; sig *= 16777619u; }

		if (!dv_sig_set[thr_id] || sig != dv_sig[thr_id]) {
			dv_sig[thr_id] = sig; dv_sig_set[thr_id] = true;
			dv_cursor[thr_id] = 0; dv_done[thr_id] = false;
		}
		if (dv_done[thr_id]) { *hashes_done = 0; return 0; } // exhausted: await new work
		pdata[19] = dv_cursor[thr_id];
	}

	const uint32_t first_nonce = pdata[19];

	do {
		// Clamp the final window so the cursor never wraps past the 2^32 nonce_lo
		// space (dv_track) or the host max_nonce (benchmark).
		uint32_t span = throughput;
		if ((uint64_t)pdata[19] + span > scan_end)
			span = (uint32_t)(scan_end - pdata[19]);

		cudaMemset(d_resNonce[thr_id], 0xff, sizeof(uint32_t));

		const uint32_t tpb = 256;
		dim3 grid((span + tpb - 1) / tpb);
		dim3 block(tpb);
		sha256dv_gpu_hash <<< grid, block >>> (span, pdata[19], d_resNonce[thr_id]);

		uint32_t resNonce = UINT32_MAX;
		cudaMemcpy(&resNonce, d_resNonce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);

		*hashes_done = pdata[19] - first_nonce + span;

		if (resNonce != UINT32_MAX) {
			uint32_t hash_le[8];
			sha256dv_build_stage2(stage2, work, resNonce, nonce_hi);
			host_sha256dv(hash_le, stage2);

			if (sha256dv_meets_target(hash_le, ptarget)) {
				work->veil_nonce_lo = resNonce;
				work->veil_nonce_hi = nonce_hi;
				work->nonces[0] = resNonce;
				work->valid_nonces = 1;
				memcpy(work->target, ptarget, sizeof(work->target));
				for (int i = 0; i < 8; i++) ((uint32_t*)work->extra)[i] = hash_le[i];
				pdata[19] = resNonce + 1;
				if (dv_track) {
					dv_cursor[thr_id] = pdata[19];
					if (resNonce == 0xFFFFFFFFu) dv_done[thr_id] = true; // last nonce_lo
				}
				return 1;
			} else {
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
					gpulog(LOG_WARNING, thr_id, "result for nonce_lo %08x does not validate on CPU!", resNonce);
			}
		}

		if ((uint64_t)pdata[19] + span >= scan_end) {
			if (dv_track) {
				dv_cursor[thr_id] = pdata[19] + span;   // for hashrate accounting
				dv_done[thr_id]   = true;               // job exhausted: await new work
			} else {
				pdata[19] = (uint32_t)scan_end;
			}
			break;
		}
		pdata[19] += span;
		if (dv_track) dv_cursor[thr_id] = pdata[19];

	} while (!work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

extern "C" void free_sha256dv(int thr_id)
{
	if (!init_done[thr_id])
		return;

	cudaDeviceSynchronize();
	cudaFree(d_resNonce[thr_id]);
	init_done[thr_id]  = false;
	dv_sig_set[thr_id] = false;
	dv_done[thr_id]    = false;
	dv_cursor[thr_id]  = 0;
	cudaDeviceSynchronize();
}
