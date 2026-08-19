/* yespower 1.0 head and tail -- the SHA-256 either side of SMix.
 *
 * Transcribed from sph/yespower_ref.c:yespower_ref(), version != YESPOWER_0_5:
 *
 *   sha256 = SHA256(src, srclen)                 <- the 80-byte header
 *   salt   = pers (or empty)                     <- 1.0 REPLACES the salt here
 *   B      = PBKDF2-SHA256(sha256, salt, 1, 128r)
 *   sha256 = B[0..7]                             <- first 32 bytes, as words
 *   smix(B)
 *   dst    = HMAC-SHA256(key = last 64 B of B, msg = sha256)
 *
 * WORD ORDER.  Two conventions meet here and the reference switches between
 * them silently, by casting a byte buffer to uint32_t*:
 *   - SHA-256 state/input words are BIG-endian valued (the standard).
 *   - B, X and V are LITTLE-endian words of the byte stream, because that is
 *     what Salsa20 and the SIMD shuffle in smix operate on.
 * So a bswap appears at exactly the two points where the reference reinterprets
 * a digest as host words, and nowhere else.  Both are marked below.
 *
 * NO LOCAL ARRAYS.  The message blocks are built word-by-word from
 * __constant__ memory instead of a uint8_t scratch buffer, so this contributes
 * a 0-byte stack frame to the fused kernel.  The only array is
 * the 16-word block sha256_transform_full() needs, which stays in registers
 * because every index into it is a compile-time constant after unrolling.
 */

#ifndef YESPOWER_HEAD_TAIL_CUH
#define YESPOWER_HEAD_TAIL_CUH

#include "cuda/sha256_device.cuh"

/* The personalisation string, uploaded once per job.  80 bytes covers every
 * known variant; the longest in the wild is cpupower's 73-byte string, which is
 * also the only one that spans more than one SHA-256 block. */
#define YP_PERS_MAX 80
__constant__ uint8_t  c_yp_pers[YP_PERS_MAX];
__constant__ uint32_t c_yp_perslen;

__device__ __forceinline__ void yp_sha256_init(uint32_t st[8])
{
#pragma unroll
	for (int i = 0; i < 8; i++) st[i] = c_sha256_H[i];
}

/* --------------------------------------------------------------------------
 * Head: SHA-256 of the 80-byte header.
 *
 * `hdr` is 20 words already in SHA-256 input order (big-endian valued), which
 * is what the host's be32enc(endiandata) pass produces.  `w19` supplies the
 * last word separately so the mining kernel can keep the other 19 in
 * __constant__ and vary only the nonce -- copying all 20 into a local array
 * would cost an 80-byte stack frame.  hdr[19] is ignored.
 * ------------------------------------------------------------------------ */
__device__ __forceinline__ void yp_sha256_80(const uint32_t *hdr, uint32_t w19,
                                             uint32_t out[8])
{
	uint32_t in[16];

	yp_sha256_init(out);

#pragma unroll
	for (int i = 0; i < 16; i++) in[i] = hdr[i];
	sha256_transform_full(in, out, c_sha256_K);

#pragma unroll
	for (int i = 0; i < 3; i++) in[i] = hdr[16 + i];
	in[3] = w19;
	in[4] = 0x80000000u;
#pragma unroll
	for (int i = 5; i < 15; i++) in[i] = 0;
	in[15] = 80u * 8u;                        /* 640 bits */
	sha256_transform_full(in, out, c_sha256_K);
}

/* --------------------------------------------------------------------------
 * PBKDF2-SHA256 with c == 1.
 *
 * Every output block is an INDEPENDENT HMAC, which is what lets the four lanes
 * split the work round-robin with no communication at all:
 *
 *   T_i = HMAC(passwd, salt || BE32(i)),  i = 1 .. dkLen/32
 *
 * The two pad midstates depend only on the password, so they are computed once
 * per instance and reused for every block.
 * ------------------------------------------------------------------------ */

/* Word `widx` of (pers || BE32(counter) || 0x80 || 0...), big-endian valued.
 * Reading past the message yields the padding, so the caller never special-cases
 * the boundary. The length word is written by the caller. */
__device__ __forceinline__ uint32_t yp_salt_msg_word(uint32_t widx, uint32_t counter)
{
	const uint32_t plen = c_yp_perslen;
	const uint32_t bo = widx * 4u;
	uint32_t w = 0;
#pragma unroll
	for (int b = 0; b < 4; b++) {
		const uint32_t o = bo + (uint32_t)b;
		uint32_t v;
		if (o < plen)             v = c_yp_pers[o];
		else if (o < plen + 4u)   v = (counter >> (8u * (3u - (o - plen)))) & 0xffu;
		else if (o == plen + 4u)  v = 0x80u;
		else                      v = 0;
		w = (w << 8) | v;
	}
	return w;
}

/* ist/ost <- the two HMAC pad midstates for a 32-byte key.
 * `key` is 8 big-endian-valued words (i.e. a SHA-256 digest as produced above),
 * so it is already in SHA-256 input order and needs no swap. */
__device__ __forceinline__ void yp_hmac_pads_key32(const uint32_t key[8],
                                                   uint32_t ist[8], uint32_t ost[8])
{
	uint32_t in[16];

	yp_sha256_init(ist);
#pragma unroll
	for (int i = 0; i < 8; i++)  in[i] = key[i] ^ 0x36363636u;
#pragma unroll
	for (int i = 8; i < 16; i++) in[i] = 0x36363636u;
	sha256_transform_full(in, ist, c_sha256_K);

	yp_sha256_init(ost);
#pragma unroll
	for (int i = 0; i < 8; i++)  in[i] = key[i] ^ 0x5c5c5c5cu;
#pragma unroll
	for (int i = 8; i < 16; i++) in[i] = 0x5c5c5c5cu;
	sha256_transform_full(in, ost, c_sha256_K);
}

/* One PBKDF2 output block: T_counter, 8 big-endian-valued words. */
__device__ __forceinline__ void yp_pbkdf2_block(const uint32_t ist[8], const uint32_t ost[8],
                                                uint32_t counter, uint32_t out[8])
{
	uint32_t in[16];
	uint32_t st[8];

	/* inner = SHA256(K^ipad || salt || BE32(counter)) -- the ipad block is
	 * already absorbed into ist, so the message here is salt||counter and its
	 * length includes that leading block. */
	const uint32_t mlen = c_yp_perslen + 4u;      /* bytes after the pad block */
	const uint32_t nblk = (mlen + 9u + 63u) / 64u;

#pragma unroll
	for (int i = 0; i < 8; i++) st[i] = ist[i];

	for (uint32_t b = 0; b < nblk; b++) {
#pragma unroll
		for (int i = 0; i < 16; i++) in[i] = yp_salt_msg_word(b * 16u + (uint32_t)i, counter);
		if (b == nblk - 1u) {
			in[14] = 0;
			in[15] = (mlen + 64u) * 8u;
		}
		sha256_transform_full(in, st, c_sha256_K);
	}

	/* outer = SHA256(K^opad || inner) */
#pragma unroll
	for (int i = 0; i < 8; i++) in[i] = st[i];
	in[8] = 0x80000000u;
#pragma unroll
	for (int i = 9; i < 15; i++) in[i] = 0;
	in[15] = (64u + 32u) * 8u;
#pragma unroll
	for (int i = 0; i < 8; i++) out[i] = ost[i];
	sha256_transform_full(in, out, c_sha256_K);
}

/* B <- PBKDF2(sha256(header), pers, 1, 128*R), the four lanes taking every
 * fourth output block.  B is in the LITTLE-endian word convention smix wants,
 * hence the bswap on the way out. */
template<uint32_t R>
__device__ __forceinline__ void yp_pbkdf2_fill_B(const uint32_t key[8], uint32_t *B,
                                                 const int j, const unsigned mask)
{
	uint32_t ist[8], ost[8], t[8];
	yp_hmac_pads_key32(key, ist, ost);

	const uint32_t nblocks = 4u * R;          /* 128*R bytes / 32 bytes */
	for (uint32_t i = (uint32_t)j; i < nblocks; i += 4u) {
		yp_pbkdf2_block(ist, ost, i + 1u, t);
#pragma unroll
		for (int k = 0; k < 8; k++) B[i * 8u + (uint32_t)k] = cuda_swab32(t[k]);
	}
	__syncwarp(mask);
}

/* --------------------------------------------------------------------------
 * Tail: HMAC-SHA256 keyed by the LAST 64 bytes of B, over the 32 bytes saved
 * from the FIRST 32 bytes of B before smix ran.
 *
 * The key is exactly one block, so there is no key hashing -- but it does come
 * from B, so it needs the bswap back into SHA-256 word order.
 * ------------------------------------------------------------------------ */
template<uint32_t R>
__device__ __forceinline__ void yp_hmac_tail(const uint32_t *B, const uint32_t saved[8],
                                             uint32_t out[8])
{
	uint32_t in[16], st[8];
	const uint32_t *K = B + 32u * R - 16u;    /* last 64 bytes */

	/* inner = SHA256((K^ipad) || saved) */
	yp_sha256_init(st);
#pragma unroll
	for (int i = 0; i < 16; i++) in[i] = cuda_swab32(K[i]) ^ 0x36363636u;
	sha256_transform_full(in, st, c_sha256_K);

#pragma unroll
	for (int i = 0; i < 8; i++) in[i] = saved[i];
	in[8] = 0x80000000u;
#pragma unroll
	for (int i = 9; i < 15; i++) in[i] = 0;
	in[15] = (64u + 32u) * 8u;
	sha256_transform_full(in, st, c_sha256_K);

	/* out = SHA256((K^opad) || inner) */
	yp_sha256_init(out);
#pragma unroll
	for (int i = 0; i < 16; i++) in[i] = cuda_swab32(K[i]) ^ 0x5c5c5c5cu;
	sha256_transform_full(in, out, c_sha256_K);

#pragma unroll
	for (int i = 0; i < 8; i++) in[i] = st[i];
	in[8] = 0x80000000u;
#pragma unroll
	for (int i = 9; i < 15; i++) in[i] = 0;
	in[15] = (64u + 32u) * 8u;
	sha256_transform_full(in, out, c_sha256_K);
}

#endif /* YESPOWER_HEAD_TAIL_CUH */
