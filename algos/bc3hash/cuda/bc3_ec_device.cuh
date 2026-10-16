/*
 * bc3hash fixed-base secp256k1 multiplication: k*P for the four mix512 bases
 * H, J, K, L (mix.cpp CurveTable). The field/group code is curvehash's
 * verbatim libsecp256k1 transcription, used unmodified.
 *
 * Table: 16-bit windows, the same layout as curvehash's ecmult_gen table:
 * per base 16 windows x 65536 entries x 64 bytes (big-endian X||Y), entry
 * (j, i) = (i * 2^(16j)) * P, entry 0 unused = 64 MB per base. Built ON THE
 * DEVICE by bc3_ec_table_kernel from the 16 window bases 2^(16j)*P (the host
 * derives those with libsecp, 16 doublings per window).
 *
 * Semantics (mix.cpp CurveTable::operator()): k >= n is reduced, k == 0 mod n
 * gives (INVALID_POINT_x, INVALID_POINT_y). Raw 16-bit windows of any k < 2^256
 * already sum to (k mod n)*P, and k mod n == 0 only for k = 0 and k = n, where
 * the sum is infinity, so "infinity => sentinel" covers both.
 */
#pragma once
#include "bc3_uint_device.cuh"
#include "bc3_constants.cuh"
#include "../../curvehash/cuda/secp256k1_group_device.cuh"

#define BC3_EC_WBITS 16
#define BC3_EC_WINDOWS (256 / BC3_EC_WBITS)
#define BC3_EC_ENTRIES (1u << BC3_EC_WBITS)
#define BC3_EC_ENTRY 64
#define BC3_EC_TABLE_BYTES ((size_t)BC3_EC_WINDOWS * BC3_EC_ENTRIES * BC3_EC_ENTRY)

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

/* Table builder: one thread per (window j, entry i >= 1) of one base; entry =
 * i * B_j by MSB-first double-and-add over the 16 bits of i, then affine.
 * wbase: 16 affine window bases B_j = 2^(16j)*P as big-endian X||Y (64 B each). */
__global__ void bc3_ec_table_kernel(uint8_t* __restrict__ table, const uint8_t* __restrict__ wbase)
{
	const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= BC3_EC_WINDOWS * BC3_EC_ENTRIES) return;
	const uint32_t j = t >> BC3_EC_WBITS, i = t & (BC3_EC_ENTRIES - 1);
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
	for (int b = BC3_EC_WBITS - 1; b >= 0; b--) {
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
