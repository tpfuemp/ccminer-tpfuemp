/* yespower 1.0 -- the whole hash, as one inlinable device function.
 *
 * Transcribed from sph/yespower_ref.c:yespower_ref() + smix().  The KAT kernel
 * and the mining kernel both inline THIS body, so they cannot drift apart --
 * the same discipline sha512256d's differential uses.
 *
 * One instance = 4 cooperating threads (PWXgather lanes).  X, B and V are plain
 * memory; S is the 96 KiB box array, expected in shared memory.
 *
 * The S-box fill is smix1 with r=1, N=Sbytes/128=768 and V aliased ONTO S
 * itself, using Salsa20 rather than pwxform (the boxes do not exist yet).  That
 * fill writes all 768 x 128 B, so S is fully initialised before the first
 * gather; it needs no separate clear.
 */

#ifndef YESPOWER_HASH_CUH
#define YESPOWER_HASH_CUH

#include "yespower_smix.cuh"
#include "yespower_head_tail.cuh"

/* Head/tail primitive. yespower 1.0 uses SHA-256; yespower-b2b (mined as
 * `-a power2b` and `-a yespower-b2b`) uses BLAKE2b for the SAME structure --
 * smix/pwxform/salsa are byte-identical between them. */
enum { YP_HEAD_SHA256 = 0, YP_HEAD_B2B = 1 };

#define YP_SBOX_UINT4   6144u    /* 96 KiB / 16 B, three boxes of 2048 */
#define YP_SFILL_N       768u    /* Sbytes / 128 */

/* hdr : 20 words in SHA-256 input order (big-endian valued); hdr[19] ignored
 * w19 : the last header word, passed separately so a mining kernel can hold the
 *       other 19 in __constant__ and vary only the nonce
 * out : 8 words, the digest as LITTLE-endian words of the byte stream, which is
 *       what fulltest()/the target compare expect.
 */

/* WIDTH is capped at one warp (32) and the cap is load-bearing: every barrier
 * in here is a `__syncwarp`, which orders one warp and nothing more.  Above 32
 * the code would need `__syncthreads()`.  The static_assert enforces it. */
template<uint32_t R, int PLACE = PWX_S_SHARED, int HEAD = YP_HEAD_SHA256, int WIDTH = 4>
__device__ __forceinline__ void yespower_hash_1_0(const uint32_t *hdr, const uint32_t w19,
                                                  const uint32_t N,
                                                  uint4 *S, uint32_t *B, uint32_t *X,
                                                  uint32_t *V, uint32_t out[8],
                                                  const int j, const unsigned mask,
 /* NO DEFAULTS. `tid` defaulting to 0 would make every
 * thread walk the SAME V indices at WIDTH == 4 -- lanes
 * 1..3 would never write their quarter, silently, and the
 * default would hide it from every caller. Required, so
 * the compiler names anyone who forgets. */
                                                  const int tid, const unsigned wmask)
{
	static_assert(WIDTH >= 4 && WIDTH <= 32 && (WIDTH & (WIDTH - 1)) == 0,
	              "yespower: WIDTH must be a power of two in [4,32] -- one warp; "
	              "above 32 __syncwarp cannot order the block and __syncthreads is required");
	uint32_t key[8], saved[8];
	uint8_t  b2b_init[32], b2b_saved[32];

	/* The head and the tail are 4-lane code (`j`, `mask`). With WIDTH > 4 the
 * spare threads must NOT run them: `j = tid & 3` would give threads 4..7 the
 * same lanes as 0..3 and they would race writing B. They only widen the V
 * traffic inside smix; everything else stays exactly 4 lanes wide. */
	if (tid < PWX_GATHER) {
	if (HEAD == YP_HEAD_B2B) {
		/* -- head: init = BLAKE2b(header); B = PBKDF2-BLAKE2b(init, pers, 1, 128r) --
		* No swab anywhere: (uint8_t *)B already IS the byte stream. */
		yp_b2b_init_hash(hdr, w19, b2b_init);
		yp_b2b_fill_B<R>(b2b_init, B, j, mask);
#pragma unroll
		for (int i = 0; i < 32; i++) b2b_saved[i] = ((const uint8_t *)B)[i];
	} else {
	/* -- head: sha256 = SHA256(header); B = PBKDF2(sha256, pers, 1, 128r) -- */
	yp_sha256_80(hdr, w19, key);
	yp_pbkdf2_fill_B<R>(key, B, j, mask);

	/* sha256 <- B[0..7].  The reference copies WORDS out of B and later feeds
	 * them to HMAC as BYTES, so this is where the LE word convention of B meets
	 * the BE word convention of SHA-256. */
#pragma unroll
	for (int i = 0; i < 8; i++) saved[i] = cuda_swab32(B[i]);
	}
	} /* tid < PWX_GATHER */
	YP_WSYNC(wmask); /* B is now built; every thread may read it in smix */

	/* -- smix -- */
	uint32_t b0 = 0, b1 = 2048u, b2 = 4096u, w4 = 0;

	uint32_t Nloop_all = (N + 2u) / 3u;               /* 1/3, round up   */
	uint32_t Nloop_rw  = Nloop_all;
	Nloop_all++; Nloop_all &= ~(uint32_t)1;           /* round up to even */
	Nloop_rw++;  Nloop_rw  &= ~(uint32_t)1;           /* 1.0: also up     */

	yp_smix1<1, true , PLACE, WIDTH>(B, YP_SFILL_N, (uint32_t *)S, X, S, b0, b1, b2, w4, j, mask, tid, wmask);
	yp_smix1<R, false, PLACE, WIDTH>(B, N, V, X, S, b0, b1, b2, w4, j, mask, tid, wmask);
	yp_smix2<R, PLACE, WIDTH>(B, N, Nloop_rw,              V, X, S, b0, b1, b2, w4, j, mask, tid, wmask);
	yp_smix2<R, PLACE, WIDTH>(B, N, Nloop_all - Nloop_rw,  V, X, S, b0, b1, b2, w4, j, mask, tid, wmask);

	/* -- tail: HMAC(key = last 64 B of B, msg = the saved 32 B) --
 *
 * Test-only ablation: -DYP_ABLATE_TAIL replaces the whole tail with a copy
 * out of B, bounding what truncating the final stage could ever save.  Keeps
 * B live so smix is not dead-code-eliminated.  Neither build system defines
 * it, and it produces a wrong digest by construction. */
#ifdef YP_ABLATE_TAIL
	YP_WSYNC(wmask);
	if (tid >= PWX_GATHER) return;
#pragma unroll
	for (int i = 0; i < 8; i++) out[i] = B[i];
	return;
#endif
	YP_WSYNC(wmask); /* smix's last write to B must be visible to lane 0 */
	if (tid >= PWX_GATHER) return; /* spare threads are done */
	if (HEAD == YP_HEAD_B2B) {
		/* The 32 output bytes written into (uint8_t *)out ARE the little-endian
		* words fulltest wants, so unlike the SHA-256 twin there is no swab. */
		yp_b2b_tail<R>(B, b2b_saved, out);
	} else {
	yp_hmac_tail<R>(B, saved, out);
#pragma unroll
	for (int i = 0; i < 8; i++) out[i] = cuda_swab32(out[i]);
}
}

#endif /* YESPOWER_HASH_CUH */
