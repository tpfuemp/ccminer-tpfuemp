/*
 * bc3hash fixed-base secp256k1 multiplication: k*P for the four mix512 bases
 * H, J, K, L (mix.cpp CurveTable).
 *
 * Table per base: W-bit windows (W = 16 or 20), entry (j, i) = (i * 2^(Wj)) * P
 * as big-endian X||Y (64 bytes), entry 0 unused. Built on the device by
 * bc3_ec_table_kernel from the window bases 2^(Wj)*P (derived on the host with
 * libsecp). The table builder and the single-point bc3_curve_mul use
 * curvehash's libsecp256k1 transcription; the scan path (bc3_curve_mul4) uses
 * the 8 x 32-bit field below.
 *
 * Semantics (mix.cpp CurveTable::operator()): k >= n is reduced, k == 0 mod n
 * gives (INVALID_POINT_x, INVALID_POINT_y). Raw 16-bit windows of any k < 2^256
 * already sum to (k mod n)*P, and k mod n == 0 only for k = 0 and k = n, where
 * the sum is infinity, so "infinity => sentinel" covers both.
 */
#pragma once
#include "bc3_uint_device.cuh"
#include "bc3_constants.cuh"
#include "bc3_exp_device.cuh"
#include "../../curvehash/cuda/secp256k1_group_device.cuh"

#define BC3_EC_WBITS 16
#define BC3_EC_WINDOWS (256 / BC3_EC_WBITS)
#define BC3_EC_ENTRIES (1u << BC3_EC_WBITS)
#define BC3_EC_ENTRY 64
#define BC3_EC_TABLE_BYTES ((size_t)BC3_EC_WINDOWS * BC3_EC_ENTRIES * BC3_EC_ENTRY)

/* W-bit window layout: ceil(256/W) windows, window j covers bits [jW, jW+W) and
 * starts at entry j * 2^W; the last one may be shorter. */
template <int W>
struct bc3_ecw {
	static const int windows = (256 + W - 1) / W;
	static const int last_bits = 256 - W * (windows - 1);
	__host__ __device__ static size_t entries() { return ((size_t)(windows - 1) << W) + ((size_t)1 << last_bits); }
	__host__ __device__ static size_t bytes() { return entries() * BC3_EC_ENTRY; }
};

/* Normalized field element -> bc3u256 (the integer value, limbs LS first). */
__device__ __forceinline__ bc3u256 bc3_fe_to_u256(const secp256k1_fe* a)
{
	const uint32_t* n = a->n;
	bc3u256 r;
	r.w[0] = n[0] | n[1] << 26;
	r.w[1] = n[1] >> 6 | n[2] << 20;
	r.w[2] = n[2] >> 12 | n[3] << 14;
	r.w[3] = n[3] >> 18 | n[4] << 8;
	r.w[4] = n[4] >> 24 | n[5] << 2 | n[6] << 28;
	r.w[5] = n[6] >> 4 | n[7] << 22;
	r.w[6] = n[7] >> 10 | n[8] << 16;
	r.w[7] = n[8] >> 16 | n[9] << 10;
	return r;
}

/* Jacobian sum of the 16 window entries, before the affine conversion. */
__device__ __forceinline__ void bc3_ec_accumulate(secp256k1_gej* r, const uint8_t* __restrict__ table,
                                                  const bc3u256& k)
{
	secp256k1_ge add;
	add.infinity = 0;
	secp256k1_gej_set_infinity(r);
#pragma unroll 1
	for (int j = 0; j < BC3_EC_WINDOWS; j++) {
		const uint32_t bits = (k.w[j >> 1] >> (16 * (j & 1))) & 0xffff;
		if (bits == 0) continue; /* entry 0 is infinity: additive no-op */
		const uint8_t* p = table + ((size_t)j * BC3_EC_ENTRIES + bits) * BC3_EC_ENTRY;
		secp256k1_fe_set_b32(&add.x, p);
		secp256k1_fe_set_b32(&add.y, p + 32);
		secp256k1_gej_add_ge_var(r, r, &add, NULL);
	}
}

/* CurveTable::operator(): affine (x, y) of k*P, or the sentinel. */
__device__ void bc3_curve_mul(bc3u256& x, bc3u256& y, const uint8_t* __restrict__ table, const bc3u256& k)
{
	secp256k1_gej rj;
	secp256k1_ge ra;
	bc3_ec_accumulate(&rj, table, k);
	secp256k1_ge_set_gej_var(&ra, &rj);
	if (ra.infinity) {
#pragma unroll
		for (int i = 0; i < 8; i++) {
			x.w[i] = BC3_INVALID_X[i];
			y.w[i] = BC3_INVALID_Y[i];
		}
		return;
	}
	secp256k1_fe_normalize_var(&ra.x);
	secp256k1_fe_normalize_var(&ra.y);
	x = bc3_fe_to_u256(&ra.x);
	y = bc3_fe_to_u256(&ra.y);
}

/* ---- secp256k1 field on 8 x 32-bit limbs ------------------------------------
 * p = 2^256 - C, C = 2^32 + 977, on the mix's products (bc3_mul_full /
 * bc3_sqr_full). Values are lazy (x or x + p); results leaving this file are
 * made canonical. */
#define BC3_SECP_C0 977u

/* + C to r where m is all ones (a carry out of 2^256 folded back); returns the
 * carry out of that addition as a mask */
__device__ __forceinline__ uint32_t bc3_fp_addc(bc3u256& r, uint32_t m)
{
	uint64_t c = (uint64_t)r.w[0] + (BC3_SECP_C0 & m);
	r.w[0] = (uint32_t)c;
	c = (c >> 32) + r.w[1] + (1u & m);
	r.w[1] = (uint32_t)c;
#pragma unroll
	for (int i = 2; i < 8; i++) {
		c = (c >> 32) + r.w[i];
		r.w[i] = (uint32_t)c;
	}
	return 0u - (uint32_t)(c >> 32);
}

/* - C from r where m is all ones (a borrow below 0 folded back, i.e. + p);
 * returns the borrow out of that subtraction as a mask */
__device__ __forceinline__ uint32_t bc3_fp_subc(bc3u256& r, uint32_t m)
{
	int64_t c = (int64_t)r.w[0] - (int64_t)(BC3_SECP_C0 & m);
	r.w[0] = (uint32_t)c;
	c = (c >> 32) + (int64_t)r.w[1] - (int64_t)(1u & m);
	r.w[1] = (uint32_t)c;
#pragma unroll
	for (int i = 2; i < 8; i++) {
		c = (c >> 32) + (int64_t)r.w[i];
		r.w[i] = (uint32_t)c;
	}
	return 0u - (uint32_t)(c < 0);
}

/* t (16 words) mod p, to 8 words, not canonical. t = lo + hi*977 + hi*2^32. */
__device__ __forceinline__ bc3u256 bc3_fp_reduce(const uint32_t (&t)[16])
{
	bc3u256 r;
	uint64_t c = 0;
#pragma unroll
	for (int i = 0; i < 8; i++) {
		c += (uint64_t)t[i] + (uint64_t)t[8 + i] * BC3_SECP_C0 + (i ? t[7 + i] : 0u);
		r.w[i] = (uint32_t)c;
		c >>= 32;
	}
	const uint64_t w = c + t[15]; /* < 2^44 */
	const uint64_t f = w * BC3_SECP_C0;
	c = (uint64_t)r.w[0] + (uint32_t)f;
	r.w[0] = (uint32_t)c;
	c = (c >> 32) + r.w[1] + (f >> 32) + (uint32_t)w;
	r.w[1] = (uint32_t)c;
	c = (c >> 32) + r.w[2] + (w >> 32);
	r.w[2] = (uint32_t)c;
#pragma unroll
	for (int i = 3; i < 8; i++) {
		c = (c >> 32) + r.w[i];
		r.w[i] = (uint32_t)c;
	}
	/* wrapped past 2^256: r is now below 2^77, so + C cannot wrap again */
	bc3_fp_addc(r, 0u - (uint32_t)(c >> 32));
	return r;
}

__device__ __forceinline__ bc3u256 bc3_fp_mul(const bc3u256& a, const bc3u256& b)
{
	uint32_t t[16];
	bc3_mul_full<8>(t, a.w, b.w);
	return bc3_fp_reduce(t);
}
__device__ __forceinline__ bc3u256 bc3_fp_sqr(const bc3u256& a)
{
	uint32_t t[16];
	bc3_sqr_full<8>(t, a.w);
	return bc3_fp_reduce(t);
}

/* a + b; a carry out of 2^256 is folded back as + C, which itself carries out
 * (once at most) when a + b > 2^257 - C */
__device__ __forceinline__ bc3u256 bc3_fp_add(const bc3u256& a, const bc3u256& b)
{
	bc3u256 r;
	uint64_t c = 0;
#pragma unroll
	for (int i = 0; i < 8; i++) {
		c = (c >> 32) + a.w[i] + b.w[i];
		r.w[i] = (uint32_t)c;
	}
	const uint32_t m = bc3_fp_addc(r, 0u - (uint32_t)(c >> 32));
	bc3_fp_addc(r, m);
	return r;
}

/* a - b; a borrow is folded back as + p (- C), which itself borrows (once at
 * most) when b - a > p */
__device__ __forceinline__ bc3u256 bc3_fp_sub(const bc3u256& a, const bc3u256& b)
{
	bc3u256 r;
	int64_t c = 0;
#pragma unroll
	for (int i = 0; i < 8; i++) {
		c = (c >> 32) + (int64_t)a.w[i] - (int64_t)b.w[i];
		r.w[i] = (uint32_t)c;
	}
	const uint32_t m = bc3_fp_subc(r, 0u - (uint32_t)(c < 0));
	bc3_fp_subc(r, m);
	return r;
}

/* canonical: x >= p exactly when x + C carries out of 2^256 */
__device__ __forceinline__ bc3u256 bc3_fp_canon(const bc3u256& a)
{
	bc3u256 s = a;
	return bc3_fp_addc(s, 0xffffffffu) ? s : a;
}

/* lazy x == 0 (mod p): x is 0 or p */
__device__ __forceinline__ bool bc3_fp_is0(const bc3u256& a)
{
	uint32_t z = 0, q = (a.w[0] ^ 0xfffffc2fu) | (a.w[1] ^ 0xfffffffeu);
#pragma unroll
	for (int i = 0; i < 8; i++) z |= a.w[i];
#pragma unroll
	for (int i = 2; i < 8; i++) q |= ~a.w[i];
	return z == 0 || q == 0;
}

struct bc3_jac {
	bc3u256 x, y, z; /* affine (x / z^2, y / z^3) */
	bool inf;
};

/* r = 2r (a = 0); reached only when an addend equals the running sum */
#ifdef BC3_RT_DIGITS
__device__ __noinline__ void bc3_jac_double(bc3_jac& r)
#else
__device__ __forceinline__ void bc3_jac_double(bc3_jac& r)
#endif
{
	const bc3u256 a = bc3_fp_sqr(r.x), b = bc3_fp_sqr(r.y), c = bc3_fp_sqr(b);
	bc3u256 d = bc3_fp_sub(bc3_fp_sub(bc3_fp_sqr(bc3_fp_add(r.x, b)), a), c);
	d = bc3_fp_add(d, d);
	const bc3u256 e = bc3_fp_add(bc3_fp_add(a, a), a);
	const bc3u256 x3 = bc3_fp_sub(bc3_fp_sqr(e), bc3_fp_add(d, d));
	bc3u256 c8 = bc3_fp_add(c, c);
	c8 = bc3_fp_add(c8, c8);
	c8 = bc3_fp_add(c8, c8);
	const bc3u256 y3 = bc3_fp_sub(bc3_fp_mul(e, bc3_fp_sub(d, x3)), c8);
	const bc3u256 yz = bc3_fp_mul(r.y, r.z);
	r.z = bc3_fp_add(yz, yz);
	r.x = x3;
	r.y = y3;
}

/* r += (x2, y2) affine, with the degenerate cases of libsecp's gej_add_ge_var */
__device__ __forceinline__ void bc3_jac_add_affine(bc3_jac& r, const bc3u256& x2, const bc3u256& y2)
{
	if (r.inf) {
		r.x = x2;
		r.y = y2;
		r.z = bc3_small<8>(1);
		r.inf = false;
		return;
	}
	const bc3u256 z12 = bc3_fp_sqr(r.z);
	const bc3u256 u2 = bc3_fp_mul(x2, z12);
	const bc3u256 s2 = bc3_fp_mul(bc3_fp_mul(y2, r.z), z12);
	const bc3u256 h = bc3_fp_sub(u2, r.x);
	const bc3u256 i = bc3_fp_sub(s2, r.y);
	if (bc3_fp_is0(h)) {
		if (bc3_fp_is0(i)) bc3_jac_double(r);
		else r.inf = true;
		return;
	}
	const bc3u256 h2 = bc3_fp_sqr(h);
	const bc3u256 h3 = bc3_fp_mul(h, h2);
	const bc3u256 t = bc3_fp_mul(r.x, h2);
	const bc3u256 x3 = bc3_fp_sub(bc3_fp_sub(bc3_fp_sqr(i), h3), bc3_fp_add(t, t));
	r.y = bc3_fp_sub(bc3_fp_mul(i, bc3_fp_sub(t, x3)), bc3_fp_mul(r.y, h3));
	r.z = bc3_fp_mul(r.z, h);
	r.x = x3;
}

/* table entry (big-endian X || Y, 64 B aligned) as two limb vectors */
__device__ __forceinline__ void bc3_ec_load(bc3u256& x, bc3u256& y, const uint8_t* __restrict__ p)
{
	const uint4* q = (const uint4*)p;
	const uint4 a = __ldg(q), b = __ldg(q + 1), c = __ldg(q + 2), d = __ldg(q + 3);
	const uint32_t xs[8] = { a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w };
	const uint32_t ys[8] = { c.x, c.y, c.z, c.w, d.x, d.y, d.z, d.w };
#pragma unroll
	for (int k = 0; k < 8; k++) {
		x.w[7 - k] = __byte_perm(xs[k], 0, 0x0123);
		y.w[7 - k] = __byte_perm(ys[k], 0, 0x0123);
	}
}

/* Jacobian sum of the W-bit window entries of k, before the affine step */
template <int W>
__device__ __forceinline__ void bc3_jac_accumulate(bc3_jac& r, const uint8_t* __restrict__ table, const bc3u256& k)
{
	r.inf = true;
	r.x = r.y = r.z = bc3_small<8>(0);
	bc3u256 kk = k;
#pragma unroll 1
	for (int j = 0; j < bc3_ecw<W>::windows; j++) {
#ifdef BC3_RT_DIGITS
		const int pos = j * W, wd = pos >> 5;
		uint64_t v = k.w[wd];
		if (wd + 1 < 8) v |= (uint64_t)k.w[wd + 1] << 32;
		const uint32_t bits = (uint32_t)(v >> (pos & 31)) & ((1u << W) - 1);
#else
		/* digits from a running shifted copy (see bc3_fixed_pow256_pair) */
		const uint32_t bits = kk.w[0] & ((1u << W) - 1);
		kk = bc3_shr<W>(kk);
#endif
		if (bits == 0) continue; /* entry 0 is infinity: additive no-op */
		bc3u256 x2, y2;
		bc3_ec_load(x2, y2, table + (((size_t)j << W) + bits) * BC3_EC_ENTRY);
		bc3_jac_add_affine(r, x2, y2);
	}
}

/* canonical affine (x, y) of r given 1/z; infinity gives the sentinel */
__device__ __forceinline__ void bc3_jac_affine(bc3u256& x, bc3u256& y, const bc3_jac& r, const bc3u256& zi)
{
	if (r.inf) {
#pragma unroll
		for (int i = 0; i < 8; i++) {
			x.w[i] = BC3_INVALID_X[i];
			y.w[i] = BC3_INVALID_Y[i];
		}
		return;
	}
	const bc3u256 z2 = bc3_fp_sqr(zi), z3 = bc3_fp_mul(z2, zi);
	x = bc3_fp_canon(bc3_fp_mul(r.x, z2));
	y = bc3_fp_canon(bc3_fp_mul(r.y, z3));
}

/* Four bc3_curve_mul at once, one per base, sharing one inversion (Montgomery's
 * trick). An infinite sum takes z = 1 in the product so it does not zero the
 * others. out = x0 y0 .. x3 y3. */
template <int W = BC3_EC_WBITS>
__device__ void bc3_curve_mul4(bc3u256 (&out)[8], const uint8_t* const (&tab)[4], const bc3u256& k0,
                               const bc3u256& k1, const bc3u256& k2, const bc3u256& k3)
{
	bc3_jac r[4];
	bc3_jac_accumulate<W>(r[0], tab[0], k0);
	bc3_jac_accumulate<W>(r[1], tab[1], k1);
	bc3_jac_accumulate<W>(r[2], tab[2], k2);
	bc3_jac_accumulate<W>(r[3], tab[3], k3);
#pragma unroll
	for (int i = 0; i < 4; i++)
		if (r[i].inf) r[i].z = bc3_small<8>(1);
	const bc3u256 p01 = bc3_fp_mul(r[0].z, r[1].z);
	const bc3u256 p012 = bc3_fp_mul(p01, r[2].z);
	bc3u256 inv = bc3_modinv<BC3_MOD_SECP>(bc3_fp_canon(bc3_fp_mul(p012, r[3].z))); /* 1/(z0 z1 z2 z3) */
	bc3_jac_affine(out[6], out[7], r[3], bc3_fp_mul(inv, p012));
	inv = bc3_fp_mul(inv, r[3].z); /* 1/(z0 z1 z2) */
	bc3_jac_affine(out[4], out[5], r[2], bc3_fp_mul(inv, p01));
	inv = bc3_fp_mul(inv, r[2].z); /* 1/(z0 z1) */
	bc3_jac_affine(out[2], out[3], r[1], bc3_fp_mul(inv, r[0].z));
	bc3_jac_affine(out[0], out[1], r[0], bc3_fp_mul(inv, r[1].z));
}

/* Table builder: one thread per (window j, entry i >= 1) of one base; entry =
 * i * B_j by MSB-first double-and-add, then affine. wbase: the window bases
 * B_j = 2^(Wj)*P as big-endian X||Y. Builds entries [first, end). */
template <int W = BC3_EC_WBITS>
__global__ void bc3_ec_table_kernel(uint8_t* __restrict__ table, const uint8_t* __restrict__ wbase,
                                    uint32_t first = 0, uint32_t end = 0xffffffffu)
{
	const uint32_t t = first + blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= end || t >= bc3_ecw<W>::entries()) return;
	const uint32_t j = t >> W, i = t & ((1u << W) - 1);
	uint8_t* out = table + (size_t)t * BC3_EC_ENTRY;
	if (i == 0) { /* never read; zeroed */
		for (int b = 0; b < BC3_EC_ENTRY; b++) out[b] = 0;
		return;
	}
	secp256k1_ge base;
	secp256k1_fe_set_b32(&base.x, wbase + 64 * j);
	secp256k1_fe_set_b32(&base.y, wbase + 64 * j + 32);
	base.infinity = 0;
	secp256k1_gej r;
	secp256k1_gej_set_infinity(&r);
	for (int b = W - 1; b >= 0; b--) {
		secp256k1_gej_double_var(&r, &r, NULL);
		if ((i >> b) & 1) secp256k1_gej_add_ge_var(&r, &r, &base, NULL);
	}
	secp256k1_ge a;
	secp256k1_ge_set_gej_var(&a, &r);
	secp256k1_fe_normalize_var(&a.x);
	secp256k1_fe_normalize_var(&a.y);
	secp256k1_fe_get_b32(out, &a.x);
	secp256k1_fe_get_b32(out + 32, &a.y);
}
