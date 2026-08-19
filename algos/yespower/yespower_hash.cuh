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

#define YP_SBOX_UINT4   6144u    /* 96 KiB / 16 B, three boxes of 2048 */
#define YP_SFILL_N       768u    /* Sbytes / 128 */

/* hdr : 20 words in SHA-256 input order (big-endian valued); hdr[19] ignored
 * w19 : the last header word, passed separately so a mining kernel can hold the
 *       other 19 in __constant__ and vary only the nonce
 * out : 8 words, the digest as LITTLE-endian words of the byte stream, which is
 *       what fulltest()/the target compare expect.
 */
template<uint32_t R>
__device__ __forceinline__ void yespower_hash_1_0(const uint32_t *hdr, const uint32_t w19,
                                                  const uint32_t N,
                                                  uint4 *S, uint32_t *B, uint32_t *X,
                                                  uint32_t *V, uint32_t out[8],
                                                  const int j, const unsigned mask)
{
	uint32_t key[8], saved[8];

	/* -- head: sha256 = SHA256(header); B = PBKDF2(sha256, pers, 1, 128r) -- */
	yp_sha256_80(hdr, w19, key);
	yp_pbkdf2_fill_B<R>(key, B, j, mask);

	/* sha256 <- B[0..7].  The reference copies WORDS out of B and later feeds
	 * them to HMAC as BYTES, so this is where the LE word convention of B meets
	 * the BE word convention of SHA-256. */
#pragma unroll
	for (int i = 0; i < 8; i++) saved[i] = cuda_swab32(B[i]);

	/* -- smix -- */
	uint32_t b0 = 0, b1 = 2048u, b2 = 4096u, w4 = 0;

	uint32_t Nloop_all = (N + 2u) / 3u;               /* 1/3, round up   */
	uint32_t Nloop_rw  = Nloop_all;
	Nloop_all++; Nloop_all &= ~(uint32_t)1;           /* round up to even */
	Nloop_rw++;  Nloop_rw  &= ~(uint32_t)1;           /* 1.0: also up     */

	yp_smix1<1, true >(B, YP_SFILL_N, (uint32_t *)S, X, S, b0, b1, b2, w4, j, mask);
	yp_smix1<R, false>(B, N, V, X, S, b0, b1, b2, w4, j, mask);
	yp_smix2<R>(B, N, Nloop_rw,              V, X, S, b0, b1, b2, w4, j, mask);
	yp_smix2<R>(B, N, Nloop_all - Nloop_rw,  V, X, S, b0, b1, b2, w4, j, mask);

	/* -- tail: HMAC(key = last 64 B of B, msg = the saved 32 B) -- */
	yp_hmac_tail<R>(B, saved, out);
#pragma unroll
	for (int i = 0; i < 8; i++) out[i] = cuda_swab32(out[i]);
}

#endif /* YESPOWER_HASH_CUH */
