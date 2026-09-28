/*
 * Groestl-512 shared device library: two hashes per thread, bitsliced in registers.
 *
 * State s[row][bit], 64 words: bit p = 2*col + h of s[i][b] is bit b of byte (row i, col) of
 * hash h, so ShiftBytes is one rotate per word. ~250 registers per thread plus a 64-word shared
 * stash (st[k * stride]); launch with __launch_bounds__(128, 2) or similar.
 * load64 / load80 -> compress -> store; digest_to_msg64 chains a second hash in the planes.
 */

#ifndef CUDA_GROESTL512_X2_DEVICE_CUH
#define CUDA_GROESTL512_X2_DEVICE_CUH

#include <cuda_helper.h>

#define G512X2_SWAP(x, y, m, n) do { const uint32_t t_ = ((x) ^ ((y) << (n))) & (m); (x) ^= t_; (y) ^= t_ >> (n); } while (0)

/* 8 words x 4 bytes -> 8 bit planes: swaps index bits (word k) <-> (bit b). An involution. */
__device__ __forceinline__
void groestl512_x2_transpose(uint32_t *d)
{
	G512X2_SWAP(d[0], d[1], 0xaaaaaaaau, 1); G512X2_SWAP(d[2], d[3], 0xaaaaaaaau, 1);
	G512X2_SWAP(d[4], d[5], 0xaaaaaaaau, 1); G512X2_SWAP(d[6], d[7], 0xaaaaaaaau, 1);
	G512X2_SWAP(d[0], d[2], 0xccccccccu, 2); G512X2_SWAP(d[1], d[3], 0xccccccccu, 2);
	G512X2_SWAP(d[4], d[6], 0xccccccccu, 2); G512X2_SWAP(d[5], d[7], 0xccccccccu, 2);
	G512X2_SWAP(d[0], d[4], 0xf0f0f0f0u, 4); G512X2_SWAP(d[1], d[5], 0xf0f0f0f0u, 4);
	G512X2_SWAP(d[2], d[6], 0xf0f0f0f0u, 4); G512X2_SWAP(d[3], d[7], 0xf0f0f0f0u, 4);
}

/* GF(2^8) doubling, poly 0x11b, on 8 bit planes */
__device__ __forceinline__
void groestl512_x2_mul2(const uint32_t *x, uint32_t *o)
{
	o[0] = x[7]; o[1] = x[0] ^ x[7]; o[2] = x[1]; o[3] = x[2] ^ x[7];
	o[4] = x[3] ^ x[7]; o[5] = x[4]; o[6] = x[5]; o[7] = x[6];
}

/* AES S-box, Boyar-Peralta depth-16 circuit (113 XOR/XNOR + 32 AND), planes q[b] = bit b */
__device__ __forceinline__
void groestl512_x2_sbox(uint32_t *q)
{
	const uint32_t x0 = q[7], x1 = q[6], x2 = q[5], x3 = q[4], x4 = q[3], x5 = q[2], x6 = q[1], x7 = q[0];
	uint32_t y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21;
	uint32_t t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23;
	uint32_t t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t34, t35, t36, t37, t38, t39, t40, t41, t42, t43, t44, t45;
	uint32_t t46, t47, t48, t49, t50, t51, t52, t53, t54, t55, t56, t57, t58, t59, t60, t61, t62, t63, t64, t65, t66, t67;
	uint32_t z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12, z13, z14, z15, z16, z17;
	uint32_t s0, s1, s2, s3, s4, s5, s6, s7;
	y14 = x3 ^ x5; y13 = x0 ^ x6; y9 = x0 ^ x3; y8 = x0 ^ x5; t0 = x1 ^ x2; y1 = t0 ^ x7; y4 = y1 ^ x3;
	y12 = y13 ^ y14; y2 = y1 ^ x0; y5 = y1 ^ x6; y3 = y5 ^ y8; t1 = x4 ^ y12; y15 = t1 ^ x5; y20 = t1 ^ x1;
	y6 = y15 ^ x7; y10 = y15 ^ t0; y11 = y20 ^ y9; y7 = x7 ^ y11; y17 = y10 ^ y11; y19 = y10 ^ y8;
	y16 = t0 ^ y11; y21 = y13 ^ y16; y18 = x0 ^ y16;
	t2 = y12 & y15; t3 = y3 & y6; t4 = t3 ^ t2; t5 = y4 & x7; t6 = t5 ^ t2; t7 = y13 & y16; t8 = y5 & y1;
	t9 = t8 ^ t7; t10 = y2 & y7; t11 = t10 ^ t7; t12 = y9 & y11; t13 = y14 & y17; t14 = t13 ^ t12;
	t15 = y8 & y10; t16 = t15 ^ t12; t17 = t4 ^ t14; t18 = t6 ^ t16; t19 = t9 ^ t14; t20 = t11 ^ t16;
	t21 = t17 ^ y20; t22 = t18 ^ y19; t23 = t19 ^ y21; t24 = t20 ^ y18; t25 = t21 ^ t22; t26 = t21 & t23;
	t27 = t24 ^ t26; t28 = t25 & t27; t29 = t28 ^ t22; t30 = t23 ^ t24; t31 = t22 ^ t26; t32 = t31 & t30;
	t33 = t32 ^ t24; t34 = t23 ^ t33; t35 = t27 ^ t33; t36 = t24 & t35; t37 = t36 ^ t34; t38 = t27 ^ t36;
	t39 = t29 & t38; t40 = t25 ^ t39; t41 = t40 ^ t37; t42 = t29 ^ t33; t43 = t29 ^ t40; t44 = t33 ^ t37;
	t45 = t42 ^ t41;
	z0 = t44 & y15; z1 = t37 & y6; z2 = t33 & x7; z3 = t43 & y16; z4 = t40 & y1; z5 = t29 & y7;
	z6 = t42 & y11; z7 = t45 & y17; z8 = t41 & y10; z9 = t44 & y12; z10 = t37 & y3; z11 = t33 & y4;
	z12 = t43 & y13; z13 = t40 & y5; z14 = t29 & y2; z15 = t42 & y9; z16 = t45 & y14; z17 = t41 & y8;
	t46 = z15 ^ z16; t47 = z10 ^ z11; t48 = z5 ^ z13; t49 = z9 ^ z10; t50 = z2 ^ z12; t51 = z2 ^ z5;
	t52 = z7 ^ z8; t53 = z0 ^ z3; t54 = z6 ^ z7; t55 = z16 ^ z17; t56 = z12 ^ t48; t57 = t50 ^ t53;
	t58 = z4 ^ t46; t59 = z3 ^ t54; t60 = t46 ^ t57; t61 = z14 ^ t57; t62 = t52 ^ t58; t63 = t49 ^ t58;
	t64 = z4 ^ t59; t65 = t61 ^ t62; t66 = z1 ^ t63; s0 = t59 ^ t63; s6 = t56 ^ ~t62; s7 = t48 ^ ~t60;
	t67 = t64 ^ t65; s3 = t53 ^ t66; s4 = t51 ^ t66; s5 = t47 ^ t65; s1 = t64 ^ ~s3; s2 = t55 ^ ~t67;
	q[7] = s0; q[6] = s1; q[5] = s2; q[4] = s3; q[3] = s4; q[2] = s5; q[1] = s6; q[0] = s7;
}

__device__ __forceinline__
void groestl512_x2_round(uint32_t (&s)[8][8], const uint32_t r, const bool q, const uint32_t (&rot)[8])
{
	if (q) {
		#pragma unroll
		for (int i = 0; i < 8; i++)
			#pragma unroll
			for (int b = 0; b < 8; b++) s[i][b] = ~s[i][b];
		#pragma unroll
		for (int b = 0; b < 4; b++) s[7][b] ^= 0u - ((r >> b) & 1u);
		s[7][4] ^= 0xCCCCCCCCu; s[7][5] ^= 0xF0F0F0F0u; s[7][6] ^= 0xFF00FF00u; s[7][7] ^= 0xFFFF0000u;
	} else {
		#pragma unroll
		for (int b = 0; b < 4; b++) s[0][b] ^= 0u - ((r >> b) & 1u);
		s[0][4] ^= 0xCCCCCCCCu; s[0][5] ^= 0xF0F0F0F0u; s[0][6] ^= 0xFF00FF00u; s[0][7] ^= 0xFFFF0000u;
	}
	/* ShiftBytes before SubBytes (they commute) */
	#pragma unroll
	for (int i = 0; i < 8; i++)
		#pragma unroll
		for (int b = 0; b < 8; b++) s[i][b] = __funnelshift_r(s[i][b], s[i][b], rot[i]);
	#pragma unroll
	for (int i = 0; i < 8; i++) groestl512_x2_sbox(s[i]);
	/* MixBytes: t_i = a_i+a_{i+1}; x_i = t_i+t_{i+3}; y_i = t_i+t_{i+2}+a_{i+6};
	   w_i = 2x_i + y_{i+4}; out_i = 2w_{i+3} + y_{i+4}  (== circ(2,2,3,4,5,3,5,7)) */
	uint32_t x[8][8], y[8][8];
	#pragma unroll
	for (int b = 0; b < 8; b++) {
		uint32_t t[8];
		#pragma unroll
		for (int i = 0; i < 8; i++) t[i] = s[i][b] ^ s[(i + 1) & 7][b];
		#pragma unroll
		for (int i = 0; i < 8; i++) {
			x[i][b] = t[i] ^ t[(i + 3) & 7];
			y[i][b] = t[i] ^ t[(i + 2) & 7] ^ s[(i + 6) & 7][b];
		}
	}
	#pragma unroll
	for (int j = 0; j < 8; j++) {
		uint32_t m[8]; groestl512_x2_mul2(x[j], m);
		#pragma unroll
		for (int b = 0; b < 8; b++) x[j][b] = m[b] ^ y[(j + 4) & 7][b];
	}
	#pragma unroll
	for (int i = 0; i < 8; i++) {
		uint32_t m[8]; groestl512_x2_mul2(x[(i + 3) & 7], m);
		#pragma unroll
		for (int b = 0; b < 8; b++) s[i][b] = m[b] ^ y[(i + 4) & 7][b];
	}
}

__device__ __forceinline__
void groestl512_x2_perm(uint32_t (&s)[8][8], const bool q)
{
	/* ShiftBytes sigma, P: 0 1 2 3 4 5 6 11, Q: 1 3 5 11 0 2 4 6 (columns) -> rotate 2*sigma bits */
	uint32_t rot[8];
	rot[0] = q ? 2 : 0;  rot[1] = q ? 6 : 2;  rot[2] = q ? 10 : 4; rot[3] = q ? 22 : 6;
	rot[4] = q ? 0 : 8;  rot[5] = q ? 4 : 10; rot[6] = q ? 8 : 12; rot[7] = q ? 12 : 22;
	#pragma unroll 1
	for (uint32_t r = 0; r < 14; r++) groestl512_x2_round(s, r, q, rot);
}

/* ---- message loaders: planes s[8][8] of the padded 1024-bit block, both hashes ---- */

/* two 64-byte messages (16 LE words each): row r, W[k] byte n = byte(row r, col 4n + (k>>1))
 * of hash k&1; cols 8..15 are padding (0x80 at col 8 row 0, block count 1 at col 15 row 7) */
__device__ __forceinline__
void groestl512_x2_load64(const uint32_t (&in)[2][16], uint32_t (&s)[8][8])
{
	#pragma unroll
	for (int r = 0; r < 8; r++) {
		#pragma unroll
		for (int k = 0; k < 8; k++) {
			const int h = k & 1, cc = k >> 1, rw = r >> 2, rb = r & 3;
			s[r][k] = __byte_perm(in[h][2 * cc + rw], in[h][8 + 2 * cc + rw], 0x4400 + rb + ((4 + rb) << 4)) & 0xFFFFu;
		}
		if (r == 0) { s[0][0] |= 0x00800000u; s[0][1] |= 0x00800000u; }      /* col 8 row 0 = 0x80 */
		if (r == 7) { s[7][6] |= 0x01000000u; s[7][7] |= 0x01000000u; }      /* col 15 row 7 = 0x01 */
		groestl512_x2_transpose(s[r]);
	}
}

/* two 80-byte headers (20 LE words each, typically equal but for word 19 = the nonce):
 * cols 0..9 from the header, 0x80 at col 10 row 0, block count 1 at col 15 row 7 */
__device__ __forceinline__
void groestl512_x2_load80(const uint32_t (&m)[2][20], uint32_t (&s)[8][8])
{
	#pragma unroll
	for (int r = 0; r < 8; r++) {
		#pragma unroll
		for (int k = 0; k < 8; k++) {
			const int h = k & 1, cc = k >> 1, rw = r >> 2, rb = r & 3;
			uint32_t v = __byte_perm(m[h][2 * cc + rw], m[h][8 + 2 * cc + rw], 0x4400 + rb + ((4 + rb) << 4)) & 0xFFFFu;
			if (cc < 2)                                                        /* cols 8, 9 */
				v |= __byte_perm(m[h][16 + 2 * cc + rw], 0, 0x4044 + (rb << 8)) & 0x00FF0000u;
			s[r][k] = v;
		}
		if (r == 0) { s[0][4] |= 0x00800000u; s[0][5] |= 0x00800000u; }      /* col 10 row 0 = 0x80 */
		if (r == 7) { s[7][6] |= 0x01000000u; s[7][7] |= 0x01000000u; }      /* col 15 row 7 = 0x01 */
		groestl512_x2_transpose(s[r]);
	}
}

/* one compression from IV, in place: s = P(H) ^ H with H = P(m ^ IV) ^ Q(m) ^ IV (digest =
 * cols 8..15). stride = blockDim.x for a conflict-free [64][blockDim] shared stash. */
__device__ __forceinline__
void groestl512_x2_compress(uint32_t (&s)[8][8], uint32_t *st, const uint32_t stride)
{
	/* 3 permutations sharing one body: P(m ^ IV), Q(m), P(H) */
	#define ST(i, b) st[((i) * 8 + (b)) * stride]
	#pragma unroll 1
	for (int p = 0; p < 3; p++) {
		if (p == 0) {
			#pragma unroll
			for (int i = 0; i < 8; i++)
				#pragma unroll
				for (int b = 0; b < 8; b++) ST(i, b) = s[i][b];
			s[6][1] ^= 0xC0000000u;                                           /* IV: col 15 row 6 = 0x02 */
		} else if (p == 1) {
			#pragma unroll
			for (int i = 0; i < 8; i++)
				#pragma unroll
				for (int b = 0; b < 8; b++) { const uint32_t v = ST(i, b); ST(i, b) = s[i][b]; s[i][b] = v; }
		} else {
			#pragma unroll
			for (int i = 0; i < 8; i++)
				#pragma unroll
				for (int b = 0; b < 8; b++) { s[i][b] ^= ST(i, b); ST(i, b) = s[i][b] ^ ((i == 6 && b == 1) ? 0xC0000000u : 0u); }
			s[6][1] ^= 0xC0000000u;
		}
		groestl512_x2_perm(s, p == 1);
	}
	#pragma unroll
	for (int i = 0; i < 8; i++)
		#pragma unroll
		for (int b = 0; b < 8; b++) s[i][b] ^= ST(i, b);
	#undef ST
}

/* the 64-byte digest (cols 8..15) as the padded message block of a chained Groestl-512:
 * cols move 8 -> 0 (a 16-bit rotate of every plane), then the load64 padding bits */
__device__ __forceinline__
void groestl512_x2_digest_to_msg64(uint32_t (&s)[8][8])
{
	#pragma unroll
	for (int i = 0; i < 8; i++)
		#pragma unroll
		for (int b = 0; b < 8; b++) s[i][b] >>= 16;
	s[0][7] |= 0x00030000u;                                                   /* col 8 row 0 = 0x80 */
	s[7][0] |= 0xC0000000u;                                                   /* col 15 row 7 = 0x01 */
}

/* digest word w of hash h: col c = 8 + (w>>1), rows 4(w&1)+j -> W_row[2(c&3)+h] byte (c>>2).
 * Transposes s in place (s is consumed). */
__device__ __forceinline__
void groestl512_x2_store(uint32_t (&s)[8][8], uint32_t (&out)[2][16])
{
	#pragma unroll
	for (int i = 0; i < 8; i++) groestl512_x2_transpose(s[i]);
	#pragma unroll
	for (int h = 0; h < 2; h++)
		#pragma unroll
		for (int w = 0; w < 16; w++) {
			const int c = 8 + (w >> 1), k = 2 * (c & 3) + h, n = c >> 2, r0 = 4 * (w & 1);
			const uint32_t sel = n + ((4 + n) << 4);                              /* byte n of a -> 0, of b -> 1 */
			const uint32_t lo = __byte_perm(s[r0][k], s[r0 + 1][k], sel);
			const uint32_t hi = __byte_perm(s[r0 + 2][k], s[r0 + 3][k], sel);
			out[h][w] = __byte_perm(lo, hi, 0x5410);
		}
}

/* Groestl-512 of two 64-byte messages in[0], in[1], digests written back into in[] */
__device__ __forceinline__
void groestl512_x2_hash_64(uint32_t (&in)[2][16], uint32_t *st, const uint32_t stride)
{
	uint32_t s[8][8];
	groestl512_x2_load64(in, s);
	groestl512_x2_compress(s, st, stride);
	groestl512_x2_store(s, in);
}

#undef G512X2_SWAP

#endif /* CUDA_GROESTL512_X2_DEVICE_CUH */
