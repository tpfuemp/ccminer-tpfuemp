/*
 * bc3hash device big integers: bc3u<N> = N 32-bit limbs, least significant first
 * (so its memory image equals the coin's UInt<N/2> on a little-endian host).
 * Transcribes ref/uint.hpp: wrapping +, -, bitwise ops, ror/rol/shr, the
 * per-width SHA-2-style mixers, and Mod<L,C> (pseudo-Mersenne 2^(32N) - C).
 *
 * Note: + WRAPS mod 2^(32N), as in the coin's UInt; only bc3_mod_* are modular.
 */
#pragma once
#include <stdint.h>

template <int N>
struct bc3u {
	uint32_t w[N];
};
typedef bc3u<2> bc3u64;
typedef bc3u<4> bc3u128;
typedef bc3u<8> bc3u256;

template <int N>
__device__ __forceinline__ bc3u<N> bc3_small(uint32_t x)
{
	bc3u<N> r;
#pragma unroll
	for (int i = 0; i < N; i++) r.w[i] = 0;
	r.w[0] = x;
	return r;
}

/* Zero-extend / truncate between widths (UInt<L>(const UInt<M>&)). */
template <int N, int M>
__device__ __forceinline__ bc3u<N> bc3_resize(const bc3u<M>& a)
{
	bc3u<N> r;
#pragma unroll
	for (int i = 0; i < N; i++) r.w[i] = i < M ? a.w[i] : 0;
	return r;
}

template <int N>
__device__ __forceinline__ bc3u<N> operator+(const bc3u<N>& a, const bc3u<N>& b)
{
	bc3u<N> r;
	uint64_t c = 0;
#pragma unroll
	for (int i = 0; i < N; i++) {
		c = (uint64_t)a.w[i] + b.w[i] + (c >> 32);
		r.w[i] = (uint32_t)c;
	}
	return r;
}

template <int N>
__device__ __forceinline__ bc3u<N> operator-(const bc3u<N>& a, const bc3u<N>& b)
{
	bc3u<N> r;
	uint32_t borrow = 0;
#pragma unroll
	for (int i = 0; i < N; i++) {
		const uint64_t d = (uint64_t)a.w[i] - b.w[i] - borrow;
		r.w[i] = (uint32_t)d;
		borrow = (uint32_t)(d >> 63);
	}
	return r;
}

#define BC3_BITOP(op)                                                               \
	template <int N>                                                                \
	__device__ __forceinline__ bc3u<N> operator op(const bc3u<N>& a, const bc3u<N>& b) \
	{                                                                               \
		bc3u<N> r;                                                                  \
		_Pragma("unroll") for (int i = 0; i < N; i++) r.w[i] = a.w[i] op b.w[i];    \
		return r;                                                                   \
	}
BC3_BITOP(^)
BC3_BITOP(&)
BC3_BITOP(|)
#undef BC3_BITOP

template <int N>
__device__ __forceinline__ bc3u<N> operator~(const bc3u<N>& a)
{
	bc3u<N> r;
#pragma unroll
	for (int i = 0; i < N; i++) r.w[i] = ~a.w[i];
	return r;
}

template <int N>
__device__ __forceinline__ bool bc3_is_zero(const bc3u<N>& a)
{
	uint32_t r = 0;
#pragma unroll
	for (int i = 0; i < N; i++) r |= a.w[i];
	return r == 0;
}

/* Rotate right by R over the full 32N-bit value (uint.hpp ror<R>). */
template <int R, int N>
__device__ __forceinline__ bc3u<N> bc3_ror(const bc3u<N>& a)
{
	constexpr int q = (R / 32) % N, s = R % 32;
	bc3u<N> r;
#pragma unroll
	for (int i = 0; i < N; i++) {
		const uint32_t lo = a.w[(i + q) % N], hi = a.w[(i + q + 1) % N];
		r.w[i] = s ? __funnelshift_r(lo, hi, s) : lo;
	}
	return r;
}

template <int R, int N>
__device__ __forceinline__ bc3u<N> bc3_rol(const bc3u<N>& a)
{
	return bc3_ror<32 * N - R % (32 * N)>(a);
}

/* Logical shift right by R (uint.hpp shr<R>). */
template <int R, int N>
__device__ __forceinline__ bc3u<N> bc3_shr(const bc3u<N>& a)
{
	constexpr int q = R / 32, s = R % 32;
	bc3u<N> r;
#pragma unroll
	for (int i = 0; i < N; i++) {
		const uint32_t lo = i + q < N ? a.w[i + q] : 0, hi = i + q + 1 < N ? a.w[i + q + 1] : 0;
		r.w[i] = s ? __funnelshift_r(lo, hi, s) : lo;
	}
	return r;
}

template <int N>
__device__ __forceinline__ bc3u<N> bc3_ch(const bc3u<N>& x, const bc3u<N>& y, const bc3u<N>& z)
{
	return z ^ (x & (y ^ z));
}

template <int N>
__device__ __forceinline__ bc3u<N> bc3_maj(const bc3u<N>& x, const bc3u<N>& y, const bc3u<N>& z)
{
	return (x & y) | (z & (x | y));
}

/* Per-width rotation constants (uint.hpp, L = N/2 64-bit limbs). */
template <int N> struct bc3_rot;
template <> struct bc3_rot<2> {
	enum { S0a = 28, S0b = 34, S0c = 39, S1a = 14, S1b = 18, S1c = 41,
	       s0a = 1, s0b = 8, s0s = 7, s1a = 19, s1b = 61, s1s = 6,
	       beta = 51, gr = 30, gl = 38, gs = 12 };
};
template <> struct bc3_rot<4> {
	enum { S0a = 57, S0b = 63, S0c = 92, S1a = 35, S1b = 44, S1c = 51,
	       s0a = 42, s0b = 83, s0s = 20, s1a = 24, s1b = 119, s1s = 18,
	       beta = 54, gr = 48, gl = 99, gs = 22 };
};
template <> struct bc3_rot<8> {
	enum { S0a = 113, S0b = 146, S0c = 171, S1a = 88, S1b = 97, S1c = 201,
	       s0a = 131, s0b = 57, s0s = 56, s1a = 53, s1b = 247, s1s = 26,
	       beta = 81, gr = 58, gl = 183, gs = 32 };
};

template <int N>
__device__ __forceinline__ bc3u<N> bc3_Sig0(const bc3u<N>& x)
{
	typedef bc3_rot<N> K;
	return bc3_ror<K::S0a>(x) ^ bc3_ror<K::S0b>(x) ^ bc3_ror<K::S0c>(x);
}
template <int N>
__device__ __forceinline__ bc3u<N> bc3_Sig1(const bc3u<N>& x)
{
	typedef bc3_rot<N> K;
	return bc3_ror<K::S1a>(x) ^ bc3_ror<K::S1b>(x) ^ bc3_ror<K::S1c>(x);
}
template <int N>
__device__ __forceinline__ bc3u<N> bc3_sig0(const bc3u<N>& x)
{
	typedef bc3_rot<N> K;
	return bc3_ror<K::s0a>(x) ^ bc3_ror<K::s0b>(x) ^ bc3_shr<K::s0s>(x);
}
template <int N>
__device__ __forceinline__ bc3u<N> bc3_sig1(const bc3u<N>& x)
{
	typedef bc3_rot<N> K;
	return bc3_ror<K::s1a>(x) ^ bc3_ror<K::s1b>(x) ^ bc3_shr<K::s1s>(x);
}
template <int N>
__device__ __forceinline__ bc3u<N> bc3_beta(const bc3u<N>& x)
{
	return x ^ bc3_Sig0(x) ^ bc3_Sig1(x) ^ bc3_sig0(x) ^ bc3_sig1(x) ^ bc3_rol<bc3_rot<N>::beta>(x);
}
template <int N>
__device__ __forceinline__ bc3u<N> bc3_gamma(const bc3u<N>& x, const bc3u<N>& y)
{
	typedef bc3_rot<N> K;
	return x ^ bc3_beta(y) ^ bc3_ror<K::gr>(y) ^ bc3_rol<K::gl>(bc3_sig1(x)) ^ bc3_shr<K::gs>(y);
}

/* UInt::load / store: BIG-endian bytes, byte 0 is the top of the value. */
template <int N>
__device__ __forceinline__ bc3u<N> bc3_load_be(const uint8_t* p)
{
	bc3u<N> r;
#pragma unroll
	for (int k = 0; k < N; k++)
		r.w[N - 1 - k] = ((uint32_t)p[4 * k] << 24) | ((uint32_t)p[4 * k + 1] << 16) |
		                 ((uint32_t)p[4 * k + 2] << 8) | p[4 * k + 3];
	return r;
}
template <int N>
__device__ __forceinline__ void bc3_store_be(uint8_t* p, const bc3u<N>& a)
{
#pragma unroll
	for (int k = 0; k < N; k++) {
		const uint32_t v = a.w[N - 1 - k];
		p[4 * k] = (uint8_t)(v >> 24);
		p[4 * k + 1] = (uint8_t)(v >> 16);
		p[4 * k + 2] = (uint8_t)(v >> 8);
		p[4 * k + 3] = (uint8_t)v;
	}
}

/* ---- Mod<L, C>: modulus 2^(32N) - C, C < 2^16 ------------------------------ */

/* a >= m  =>  a - m. Since m = 2^(32N) - C, a >= m iff a + C carries out. */
template <int N, uint32_t C>
__device__ __forceinline__ bc3u<N> bc3_mod_normalize(const bc3u<N>& a)
{
	bc3u<N> s;
	uint64_t c = C;
#pragma unroll
	for (int i = 0; i < N; i++) {
		c += a.w[i];
		s.w[i] = (uint32_t)c;
		c >>= 32;
	}
	return c ? s : a;
}

/* Mod::reduce of a 2N-limb product: two carry folds by C, then normalize.
 * Accepts any a, b < 2^(32N); the result is canonical (< m), so it equals the
 * coin's bit for bit. */
template <int N, uint32_t C>
__device__ __forceinline__ bc3u<N> bc3_mod_mul(const bc3u<N>& a, const bc3u<N>& b)
{
	uint32_t t[2 * N];
#pragma unroll
	for (int i = 0; i < 2 * N; i++) t[i] = 0;
#pragma unroll
	for (int i = 0; i < N; i++) {
		uint64_t c = 0;
#pragma unroll
		for (int j = 0; j < N; j++) {
			c = (uint64_t)a.w[i] * b.w[j] + t[i + j] + (c >> 32);
			t[i + j] = (uint32_t)c;
		}
		t[i + N] = (uint32_t)(c >> 32);
	}
	bc3u<N> r;
	uint64_t c = 0;
#pragma unroll
	for (int i = 0; i < N; i++) {
		c = (uint64_t)t[N + i] * C + t[i] + (c >> 32);
		r.w[i] = (uint32_t)c;
	}
	c = (c >> 32) * C; /* first carry <= C, so this is < 2^32 */
#pragma unroll
	for (int i = 0; i < N; i++) {
		c += r.w[i];
		r.w[i] = (uint32_t)c;
		c >>= 32;
	}
	if (c) { /* the second fold overflowed: adding C cannot overflow again */
		c = C;
#pragma unroll
		for (int i = 0; i < N; i++) {
			c += r.w[i];
			r.w[i] = (uint32_t)c;
			c >>= 32;
		}
	}
	return bc3_mod_normalize<N, C>(r);
}

template <int N, uint32_t C>
__device__ __forceinline__ bc3u<N> bc3_mod_sqr(const bc3u<N>& a)
{
	return bc3_mod_mul<N, C>(a, a);
}

/* The six moduli of mix512 (uint.hpp: M64/M128/M256 for the fixed exponents and
 * M*::mul, F64/F128/F256 for the fixed-base tables and inverse256). */
#define BC3_M64 2, 59u
#define BC3_M128 4, 159u
#define BC3_M256 8, 189u
#define BC3_F64 2, 1469u
#define BC3_F128 4, 15449u
#define BC3_F256 8, 36113u
