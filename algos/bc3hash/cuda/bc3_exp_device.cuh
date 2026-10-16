/*
 * bc3hash fixed exponents, fixed-base powers and inverse256 on device.
 *
 * FixedExponent (uint.hpp): sliding-window chain whose schedule is a constant of
 * the algorithm (generated into bc3_constants.cuh). The schedule is read from
 * __constant__ memory (warp-uniform); the odd-power table is indexed at run time.
 * .pair evaluates two independent chains for ILP, like the coin.
 *
 * FixedPower (uint.hpp): g^e = prod_i table[i][byte_i(e)], tables built on the
 * host by the oracle and uploaded (global memory, read through __ldg).
 *
 * inverse256 = a^(p-2) mod F256 (window 5); gives 0 for a == 0 mod p, as the
 * coin requires.
 */
#pragma once
#include "bc3_uint_device.cuh"
#include "bc3_constants.cuh"

#define BC3_SCHED(NAME) BC3_##NAME##_WINDOW, BC3_##NAME##_COUNT, BC3_##NAME##_TAIL, BC3_##NAME##_SQ, BC3_##NAME##_IX

template <int N, uint32_t C, int W>
__device__ __forceinline__ void bc3_odd_table(bc3u<N> (&odd)[1 << (W - 1)], const bc3u<N>& a)
{
	odd[0] = bc3_mod_normalize<N, C>(a);
	const bc3u<N> a2 = bc3_mod_sqr<N, C>(odd[0]);
	for (int i = 1; i < (1 << (W - 1)); i++)
		odd[i] = bc3_mod_mul<N, C>(odd[i - 1], a2);
}

template <int N, uint32_t C, int W>
__device__ void bc3_fixed_exp(bc3u<N>& r, const bc3u<N>& a, int count, int tail,
                              const uint8_t* sq, const uint8_t* ix)
{
	bc3u<N> odd[1 << (W - 1)];
	bc3_odd_table<N, C, W>(odd, a);
	r = odd[ix[0]];
	for (int i = 1; i < count; i++) {
		for (int j = 0; j < sq[i]; j++)
			r = bc3_mod_sqr<N, C>(r);
		r = bc3_mod_mul<N, C>(r, odd[ix[i]]);
	}
	for (int i = 0; i < tail; i++)
		r = bc3_mod_sqr<N, C>(r);
}

template <int N, uint32_t C, int W>
__device__ void bc3_fixed_exp_pair(bc3u<N>& r, bc3u<N>& s, const bc3u<N>& a, const bc3u<N>& b,
                                   int count, int tail, const uint8_t* sq, const uint8_t* ix)
{
	bc3u<N> odda[1 << (W - 1)], oddb[1 << (W - 1)];
	bc3_odd_table<N, C, W>(odda, a);
	bc3_odd_table<N, C, W>(oddb, b);
	r = odda[ix[0]];
	s = oddb[ix[0]];
	for (int i = 1; i < count; i++) {
		for (int j = 0; j < sq[i]; j++) {
			r = bc3_mod_sqr<N, C>(r);
			s = bc3_mod_sqr<N, C>(s);
		}
		r = bc3_mod_mul<N, C>(r, odda[ix[i]]);
		s = bc3_mod_mul<N, C>(s, oddb[ix[i]]);
	}
	for (int i = 0; i < tail; i++) {
		r = bc3_mod_sqr<N, C>(r);
		s = bc3_mod_sqr<N, C>(s);
	}
}

/* The three exponents mix512 uses (with the coin's windows) and the inverse. */
__device__ __forceinline__ void bc3_exp64_pair(bc3u64& r, bc3u64& s, const bc3u64& a, const bc3u64& b)
{
	bc3_fixed_exp_pair<BC3_M64, BC3_EXP64_WINDOW>(r, s, a, b, BC3_EXP64_COUNT, BC3_EXP64_TAIL, BC3_EXP64_SQ, BC3_EXP64_IX);
}
__device__ __forceinline__ void bc3_exp128_pair(bc3u128& r, bc3u128& s, const bc3u128& a, const bc3u128& b)
{
	bc3_fixed_exp_pair<BC3_M128, BC3_EXP128_WINDOW>(r, s, a, b, BC3_EXP128_COUNT, BC3_EXP128_TAIL, BC3_EXP128_SQ, BC3_EXP128_IX);
}
__device__ __forceinline__ void bc3_exp256_pair(bc3u256& r, bc3u256& s, const bc3u256& a, const bc3u256& b)
{
	bc3_fixed_exp_pair<BC3_M256, BC3_EXP256_WINDOW>(r, s, a, b, BC3_EXP256_COUNT, BC3_EXP256_TAIL, BC3_EXP256_SQ, BC3_EXP256_IX);
}
__device__ __forceinline__ bc3u256 bc3_inverse256(const bc3u256& a)
{
	bc3u256 r;
	bc3_fixed_exp<BC3_F256, BC3_INV256_WINDOW>(r, a, BC3_INV256_COUNT, BC3_INV256_TAIL, BC3_INV256_SQ, BC3_INV256_IX);
	return r;
}

/* ---- FixedPower: table[i][256] of bc3u<N>, i over the 4N exponent bytes ---- */

template <int N>
__device__ __forceinline__ bc3u<N> bc3_ldg(const bc3u<N>* p)
{
	bc3u<N> r;
	const uint4* q = (const uint4*)p;
#pragma unroll
	for (int k = 0; k < N / 4; k++) {
		const uint4 v = __ldg(q + k);
		r.w[4 * k] = v.x;
		r.w[4 * k + 1] = v.y;
		r.w[4 * k + 2] = v.z;
		r.w[4 * k + 3] = v.w;
	}
	if (N == 2) {
		const uint2 v = __ldg((const uint2*)p);
		r.w[0] = v.x;
		r.w[1] = v.y;
	}
	return r;
}

template <int N>
__device__ __forceinline__ uint32_t bc3_byte(const bc3u<N>& e, int i)
{
	return (e.w[i >> 2] >> (8 * (i & 3))) & 255;
}

template <int N, uint32_t C>
__device__ void bc3_fixed_pow_pair(bc3u<N>& r, bc3u<N>& s, const bc3u<N>* __restrict__ table,
                                   const bc3u<N>& a, const bc3u<N>& b)
{
	r = bc3_ldg(table + bc3_byte(a, 0));
	s = bc3_ldg(table + bc3_byte(b, 0));
	for (int i = 1; i < 4 * N; i++) {
		r = bc3_mod_mul<N, C>(r, bc3_ldg(table + 256 * i + bc3_byte(a, i)));
		s = bc3_mod_mul<N, C>(s, bc3_ldg(table + 256 * i + bc3_byte(b, i)));
	}
}

template <int N, uint32_t C>
__device__ bc3u<N> bc3_fixed_pow(const bc3u<N>* __restrict__ table, const bc3u<N>& a)
{
	bc3u<N> r = bc3_ldg(table + bc3_byte(a, 0));
	for (int i = 1; i < 4 * N; i++)
		r = bc3_mod_mul<N, C>(r, bc3_ldg(table + 256 * i + bc3_byte(a, i)));
	return r;
}
