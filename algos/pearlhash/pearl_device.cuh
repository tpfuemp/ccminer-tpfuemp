// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * pearlhash device kernels: noise, operand prepass, and the GEMM + jackpot kernels
 * (dp4a for sm_61+, IMMA for sm_80+). The arithmetic shared with the host reference
 * uses the same B3_HD bodies from pearl_ref.h. The naive tile kernel is the
 * reference for the fast ones in the self-test.
 */
#ifndef PEARL_DEVICE_CUH
#define PEARL_DEVICE_CUH

#include <stdint.h>
#include "algos/pearlhash/pearl_ref.h"

#define PEARL_MAX_HITS		32	/* hit slots per trial; count may exceed it */

typedef struct { uint32_t w[8]; } pearl_key8;

/* label 'A' or 'B'; nblk = rows * r / 32 */
__global__ void pearl_noise_uniform_kernel(pearl_key8 key, char label, uint32_t nblk, uint32_t *U)
{
	uint32_t blk = blockIdx.x * blockDim.x + threadIdx.x;
	if (blk >= nblk) return;
	uint32_t h[8];
	pearl_random_hash(blk, label, key.w, 0, h);
	uint32_t out[8];
	#pragma unroll
	for (int i = 0; i < 8; i++) {
		uint32_t v = 0;
		#pragma unroll
		for (int b = 0; b < 4; b++)
			v |= (uint32_t)(uint8_t)pearl_uniform_byte(h[i], b) << (8 * b);
		out[i] = v;
	}
	uint4 *dst = (uint4 *)(U + (size_t)blk * 8);
	dst[0] = make_uint4(out[0], out[1], out[2], out[3]);
	dst[1] = make_uint4(out[4], out[5], out[6], out[7]);
}

/* P[l] = (p0 | p1 << 16); r <= 1024 so both fit 16 bits */
__global__ void pearl_noise_perm_kernel(pearl_key8 key, char label, uint32_t k, uint32_t r, uint32_t *P)
{
	uint32_t blk = blockIdx.x * blockDim.x + threadIdx.x;
	if (blk * 8 >= k) return;
	uint32_t h[8];
	pearl_random_hash(blk, label, key.w, 1, h);
	#pragma unroll
	for (int j = 0; j < 8; j++) {
		uint32_t l = blk * 8 + j, f, s;
		pearl_perm_pair(h[j], r, &f, &s);
		if (l < k) P[l] = f | (s << 16);
	}
}

/* out[i][l] = base[i][l] + U[i][p0[l]] - U[i][p1[l]], one thread per 4 bytes; base NULL = zero */
__global__ void pearl_operand_kernel(const int8_t *base, const int8_t *U, const uint32_t *P,
	uint32_t rows, uint32_t k, uint32_t r, int8_t *out)
{
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t words = k >> 2;
	if (idx >= rows * words) return;
	uint32_t i = idx / words, l0 = (idx % words) * 4;
	const int8_t *u = U + (size_t)i * r;
	uint32_t res = 0;
	#pragma unroll
	for (int b = 0; b < 4; b++) {
		uint32_t pp = P[l0 + b];
		int v = (int)u[pp & 0xffff] - (int)u[pp >> 16];
		if (base) v += base[(size_t)i * k + l0 + b];
		res |= (uint32_t)(uint8_t)(int8_t)v << (8 * b);
	}
	((uint32_t *)out)[idx] = res;
}

typedef struct {
	uint32_t k, r, h, w;
	uint32_t n_row_tiles, n_col_tiles;
	pearl_key8 a_seed;
	uint32_t bound[8];
} pearl_tile_params;

/* One thread per hash tile (reference order t = ri * n_col_tiles + ci).
 * dbg (optional): jackpot[16] + hash[8] per tile; hits[0] = count, then tile ids. */
__global__ void pearl_tiles_naive_kernel(pearl_tile_params p, const int8_t *An, const int8_t *Bn,
	const uint32_t *rows, const uint32_t *cols, const uint32_t *row_off, const uint32_t *col_off,
	uint32_t *dbg, uint32_t *hits)
{
	uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= p.n_row_tiles * p.n_col_tiles) return;
	uint32_t ro = row_off[t / p.n_col_tiles], co = col_off[t % p.n_col_tiles];

	int32_t acc[PEARL_MAX_PATTERN];
	for (uint32_t e = 0; e < p.h * p.w; e++) acc[e] = 0;
	uint32_t jp[16];
	#pragma unroll
	for (int i = 0; i < 16; i++) jp[i] = 0;

	uint32_t dot = p.k - p.k % p.r;
	for (uint32_t c = 0; c * p.r < dot; c++) {
		uint32_t x = 0;
		for (uint32_t u = 0; u < p.h; u++) {
			const int *pa = (const int *)(An + (size_t)(ro + rows[u]) * p.k + c * p.r);
			for (uint32_t v = 0; v < p.w; v++) {
				const int *pb = (const int *)(Bn + (size_t)(co + cols[v]) * p.k + c * p.r);
				int s = acc[u * p.w + v];
				for (uint32_t q = 0; q < p.r / 4; q++)
					s = __dp4a(pa[q], pb[q], s);
				acc[u * p.w + v] = s;
				x ^= (uint32_t)s;
			}
		}
		pearl_fold(jp, c, x);
	}

	uint32_t hw[8];
	blake3_keyed_block(p.a_seed.w, jp, hw);
	if (dbg) {
		uint32_t *d = dbg + (size_t)t * 24;
		for (int i = 0; i < 16; i++) d[i] = jp[i];
		for (int i = 0; i < 8; i++) d[16 + i] = hw[i];
	}
	if (pearl_meets_words(hw, p.bound)) {
		uint32_t slot = atomicAdd(&hits[0], 1);
		if (slot < PEARL_MAX_HITS) hits[1 + slot] = t;
	}
}

/* ===========================================================================
 * dp4a kernel. Block tile 128 x 128, 256 threads, each an 8 x 8 register block:
 * rows {4ty..4ty+3, 64+4ty..64+4ty+3}, cols likewise. That block is one hash tile
 * of PEARL_DP4A_PATTERN, so the jackpot fold is register-local. Operands are in
 * panel layout, panel[(rb * KW + kw) * 128 + row], KW = k / 4. The jackpot is a
 * rotating 16-word window (w[0] is the next word updated), so indices are static.
 */
#define PEARL_BT		128	/* block tile, rows and cols */
#define PEARL_SUB		16	/* k-words per pipeline step */
#define PEARL_GROUP_M		8	/* grouped rasterization for L2 reuse */

/* Panels: blocks of PEARL_SUB k-words x 128 rows (8 KB, one pipeline step);
 * pearl_panel_slot gives the word of (kw, row) inside its block.
 * PEARL_PERM_ID: [kw][row] (dp4a). IMMA layouts: one 16-byte quad holds one thread's
 * fragment registers and the quad index is the lane, so a fragment is one LDS.128.
 * With sl = kw / 8, t = kw & 3, kh = (kw >> 2) & 1 inside the block:
 *   A quad (sl, wm, i, lane)   = rows {g, g+8} + 16i of 64-row block wm, k-words {t, t+4}
 *   B64 quad (sl, wn, jp, lane) = cols g + 8j, j = 2jp + {0, 1}, of 64-col block wn */
#define PEARL_PERM_ID		0
#define PEARL_PERM_IMMA_A	1
#define PEARL_PERM_IMMA_B64	3

__host__ __device__ __forceinline__ uint32_t pearl_panel_slot(uint32_t perm, uint32_t kw, uint32_t row)
{
	const uint32_t ks = kw & (PEARL_SUB - 1), sl = ks >> 3, t = ks & 3, kh = (ks >> 2) & 1;
	if (perm == PEARL_PERM_IMMA_A) {
		const uint32_t wm = row >> 6, g = row & 7, h = (row >> 3) & 1, i = (row >> 4) & 3;
		return ((((sl * 2 + wm) * 4 + i) * 32 + g * 4 + t) << 2) + h + 2 * kh;
	}
	if (perm == PEARL_PERM_IMMA_B64) {
		const uint32_t wn = row >> 6, g = row & 7, j = (row >> 3) & 7;
		return ((((sl * 2 + wn) * 4 + (j >> 1)) * 32 + g * 4 + t) << 2) + kh + 2 * (j & 1);
	}
	return ks * PEARL_BT + row;
}

/* inverse of pearl_panel_slot for the IMMA layouts: slot (< 2048) -> k-word in block, row */
__host__ __device__ __forceinline__ void pearl_panel_word(uint32_t perm, uint32_t slot, uint32_t *ks, uint32_t *row)
{
	const uint32_t e = slot & 3, q = slot >> 2, g = (q >> 2) & 7, t = q & 3, rest = q >> 5;
	const uint32_t sub = rest & 3, w = (rest >> 2) & 1, sl = rest >> 3;
	if (perm == PEARL_PERM_IMMA_A) {	/* e = h + 2 kh, sub = i */
		*ks = sl * 8 + (e >> 1) * 4 + t;
		*row = w * 64 + sub * 16 + (e & 1) * 8 + g;
	} else {				/* B64: e = kh + 2 (j & 1), sub = j >> 1 */
		*ks = sl * 8 + (e & 1) * 4 + t;
		*row = w * 64 + (sub * 2 + (e >> 1)) * 8 + g;
	}
}

/* word index of (row block rb, k-word kw, row within block) in a panel of KW k-words */
__host__ __device__ __forceinline__ size_t pearl_panel_index(uint32_t perm, uint32_t rb, uint32_t KW, uint32_t kw, uint32_t row)
{
	if (perm == PEARL_PERM_ID) return ((size_t)rb * KW + kw) * PEARL_BT + row;
	return ((size_t)rb * KW + (kw & ~(uint32_t)(PEARL_SUB - 1))) * PEARL_BT + pearl_panel_slot(perm, kw, row);
}

/* panel layout of X' = X + E_U[p0] - E_U[p1]; rows must be a multiple of 128 */
template <uint32_t perm>
__global__ void pearl_operand_panel_t(const int8_t *base, const int8_t *U, const uint32_t *P,
	uint32_t rows, uint32_t k, uint32_t r, uint32_t *panel)
{
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t KW = k >> 2;
	if (idx >= rows * KW) return;
	uint32_t rr = idx % PEARL_BT, rest = idx / PEARL_BT;
	uint32_t kw = rest % KW, rb = rest / KW;
	uint32_t i = rb * PEARL_BT + rr, l0 = kw * 4;
	const int8_t *u = U + (size_t)i * r;
	uint32_t res = 0;
	#pragma unroll
	for (int b = 0; b < 4; b++) {
		uint32_t pp = P[l0 + b];
		int v = (int)u[pp & 0xffff] - (int)u[pp >> 16];
		if (base) v += base[(size_t)i * k + l0 + b];
		res |= (uint32_t)(uint8_t)(int8_t)v << (8 * b);
	}
	panel[perm == PEARL_PERM_ID ? idx : pearl_panel_index(perm, rb, KW, kw, rr)] = res;
}

/* Panel prepass: one block = one 128-row panel x PEARL_PREP_KW k-words. U rows are
 * staged in smem with a 4-byte pad (bank-conflict free); base_panel may be NULL.
 * Needs r <= 256. */
#define PEARL_PREP_KW		64

template <uint32_t perm>
__global__ void __launch_bounds__(256)
pearl_operand_panel_fast_t(const uint32_t *base_panel, const int8_t *U, const uint32_t *P,
	uint32_t k, uint32_t r, uint32_t *panel)
{
	extern __shared__ uint32_t prep_smem[];
	uint32_t *Ps = prep_smem;				/* PEARL_PREP_KW * 4 pairs */
	int8_t *Us = (int8_t *)(prep_smem + PEARL_PREP_KW * 4);	/* 128 rows, stride r + 4 */
	const uint32_t KW = k >> 2, rb = blockIdx.x, kw0 = blockIdx.y * PEARL_PREP_KW;
	const uint32_t ustride = r + 4;

	for (uint32_t i = threadIdx.x; i < PEARL_PREP_KW * 4; i += blockDim.x)
		Ps[i] = (kw0 * 4 + i < k) ? P[kw0 * 4 + i] : 0;
	const uint32_t rwords = r >> 2;
	const uint32_t *Ug = (const uint32_t *)(U + (size_t)rb * PEARL_BT * r);
	for (uint32_t i = threadIdx.x; i < PEARL_BT * rwords; i += blockDim.x) {
		uint32_t row = i / rwords, w = i % rwords;
		*(uint32_t *)(Us + row * ustride + w * 4) = Ug[i];
	}
	__syncthreads();

	if (perm != PEARL_PERM_ID) {
		/* one 16-byte output quad per step: consecutive threads write consecutive quads */
		const uint32_t kwn = min((uint32_t)PEARL_PREP_KW, KW - kw0), blk_q = PEARL_SUB * PEARL_BT / 4;
		const size_t o0 = ((size_t)rb * KW + kw0) * PEARL_BT;
		for (uint32_t qi = threadIdx.x; qi < kwn * PEARL_BT / 4; qi += blockDim.x) {
			const uint32_t blk = qi / blk_q, s0 = (qi % blk_q) * 4;
			const size_t o = o0 + (size_t)blk * PEARL_SUB * PEARL_BT + s0;
			const uint4 bq = base_panel ? *(const uint4 *)(base_panel + o) : make_uint4(0, 0, 0, 0);
			const uint32_t bw[4] = { bq.x, bq.y, bq.z, bq.w };
			uint32_t ow[4];
			#pragma unroll
			for (int e = 0; e < 4; e++) {
				uint32_t ks, wrow;
				pearl_panel_word(perm, s0 + e, &ks, &wrow);
				const uint32_t kk = blk * PEARL_SUB + ks;
				const int8_t *uu = Us + wrow * ustride;
				uint32_t out = 0;
				#pragma unroll
				for (int b = 0; b < 4; b++) {
					const uint32_t pp = Ps[kk * 4 + b];
					const int v = (int)(int8_t)(bw[e] >> (8 * b)) + (int)uu[pp & 0xffff] - (int)uu[pp >> 16];
					out |= (uint32_t)(uint8_t)(int8_t)v << (8 * b);
				}
				ow[e] = out;
			}
			*(uint4 *)(panel + o) = make_uint4(ow[0], ow[1], ow[2], ow[3]);
		}
		return;
	}
	const uint32_t row = threadIdx.x % PEARL_BT;
	const int8_t *u = Us + row * ustride;
	for (uint32_t kk = threadIdx.x / PEARL_BT; kk < PEARL_PREP_KW && kw0 + kk < KW; kk += blockDim.x / PEARL_BT) {
		const uint32_t kw = kw0 + kk;
		const size_t o = pearl_panel_index(perm, rb, KW, kw, row);
		uint32_t res = base_panel ? base_panel[o] : 0;
		uint32_t out = 0;
		#pragma unroll
		for (int b = 0; b < 4; b++) {
			const uint32_t pp = Ps[kk * 4 + b];
			const int v = (int)(int8_t)(res >> (8 * b)) + (int)u[pp & 0xffff] - (int)u[pp >> 16];
			out |= (uint32_t)(uint8_t)(int8_t)v << (8 * b);
		}
		panel[o] = out;
	}
}

/* write the first 1024 bytes of B^T row 0 into the base panel (stride 128 words) */
template <uint32_t perm>
__global__ void pearl_patch_row0_t(const uint32_t *chunk, uint32_t *panel)
{
	panel[pearl_panel_index(perm, 0, 0, threadIdx.x, 0)] = chunk[threadIdx.x];
}

/* add the first 1024 bytes of B^T row 0 into a noised panel, per byte mod 256: with a base
 * that is zero elsewhere this equals building the panel from that base */
template <uint32_t perm>
__global__ void pearl_add_row0_t(const uint32_t *chunk, uint32_t *panel)
{
	uint32_t *w = panel + pearl_panel_index(perm, 0, 0, threadIdx.x, 0);
	*w = __vadd4(*w, chunk[threadIdx.x]);
}

/* the same add, chunk passed by value (1 KB of kernel parameters); also clears the hit count */
struct pearl_chunk1k { uint32_t w[256]; };
template <uint32_t perm>
__global__ void pearl_add_row0_v_t(const pearl_chunk1k chunk, uint32_t *panel, uint32_t *hits)
{
	uint32_t *w = panel + pearl_panel_index(perm, 0, 0, threadIdx.x, 0);
	*w = __vadd4(*w, chunk.w[threadIdx.x]);
	if (threadIdx.x == 0) hits[0] = 0;
}

/* base (rows x k, row-major int8) -> panel layout, once per job */
template <uint32_t perm>
__global__ void pearl_to_panel_t(const int8_t *base, uint32_t rows, uint32_t k, uint32_t *panel)
{
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t KW = k >> 2;
	if (idx >= rows * KW) return;
	uint32_t rr = idx % PEARL_BT, rest = idx / PEARL_BT;
	uint32_t kw = rest % KW, rb = rest / KW;
	panel[perm == PEARL_PERM_ID ? idx : pearl_panel_index(perm, rb, KW, kw, rr)] = *(const uint32_t *)(base + (size_t)(rb * PEARL_BT + rr) * k + kw * 4);
}

/* identity-layout (dp4a) instances under their original names */
#define pearl_operand_panel_kernel	pearl_operand_panel_t<PEARL_PERM_ID>
#define pearl_operand_panel_fast_kernel	pearl_operand_panel_fast_t<PEARL_PERM_ID>
#define pearl_patch_row0_kernel		pearl_patch_row0_t<PEARL_PERM_ID>
#define pearl_add_row0_kernel		pearl_add_row0_t<PEARL_PERM_ID>
#define pearl_add_row0_v_kernel		pearl_add_row0_v_t<PEARL_PERM_ID>
#define pearl_to_panel_kernel		pearl_to_panel_t<PEARL_PERM_ID>

typedef struct {
	uint32_t m, n, k, r;		/* m, n multiples of 128; k multiple of 64; r multiple of 64 */
	pearl_key8 a_seed;
	uint32_t bound[8];
} pearl_gemm_params;

/* hits[0] = count; hits[1 + 2s] = tile row offset, hits[2 + 2s] = tile col offset.
 * dbg (optional): 24 words per tile in reference tile order (ri * n/8 + ci).
 * WIN: the 16-word jackpot window in registers (shifted every chunk, word 0 next) or in a
 * local array indexed by chunk & 15. */
#define PEARL_WIN_REG		0
#define PEARL_WIN_LOCAL		1

template <int WIN>
__global__ void __launch_bounds__(256, 2)
pearl_tiles_dp4a_t(pearl_gemm_params p, const uint4 *Ap, const uint4 *Bp, uint32_t *dbg, uint32_t *hits)
{
	__shared__ uint4 As[2][PEARL_SUB][PEARL_BT / 4];
	__shared__ uint4 Bs[2][PEARL_SUB][PEARL_BT / 4];

	const uint32_t tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
	const uint32_t nbm = p.m / PEARL_BT, nbn = p.n / PEARL_BT;
	const uint32_t in_group = PEARL_GROUP_M * nbn, pid = blockIdx.x;
	const uint32_t first = (pid / in_group) * PEARL_GROUP_M;
	const uint32_t gsz = min(nbm - first, (uint32_t)PEARL_GROUP_M);
	const uint32_t bm = first + (pid % in_group) % gsz;
	const uint32_t bn = (pid % in_group) / gsz;

	const uint32_t KW = p.k >> 2;
	const uint32_t steps = KW / PEARL_SUB;
	const uint32_t steps_per_chunk = (p.r >> 2) / PEARL_SUB;
	/* 512 uint4 per operand per step: thread copies tid and tid + 256 */
	const uint4 *ga = Ap + (size_t)bm * KW * (PEARL_BT / 4);
	const uint4 *gb = Bp + (size_t)bn * KW * (PEARL_BT / 4);
	uint4 *sa0 = &As[0][0][0], *sb0 = &Bs[0][0][0];
	const uint32_t buf_stride = PEARL_SUB * PEARL_BT / 4;

	int acc[8][8];
	#pragma unroll
	for (int i = 0; i < 8; i++)
		#pragma unroll
		for (int j = 0; j < 8; j++) acc[i][j] = 0;
	uint32_t win[16], jl[16], chunk = 0;
	#pragma unroll
	for (int i = 0; i < 16; i++) win[i] = 0;
	if (WIN == PEARL_WIN_LOCAL) {
		#pragma unroll
		for (int i = 0; i < 16; i++) jl[i] = 0;
	}

	uint4 ra0 = ga[tid], ra1 = ga[tid + 256], rb0 = gb[tid], rb1 = gb[tid + 256];
	sa0[tid] = ra0; sa0[tid + 256] = ra1; sb0[tid] = rb0; sb0[tid + 256] = rb1;
	__syncthreads();

	uint32_t chunk_step = 0;
	for (uint32_t s = 0; s < steps; s++) {
		const uint32_t cur = s & 1;
		if (s + 1 < steps) {
			const uint4 *na = ga + (size_t)(s + 1) * buf_stride, *nb = gb + (size_t)(s + 1) * buf_stride;
			ra0 = na[tid]; ra1 = na[tid + 256]; rb0 = nb[tid]; rb1 = nb[tid + 256];
		}
		#pragma unroll
		for (int kk = 0; kk < PEARL_SUB; kk++) {
			const uint4 a0 = As[cur][kk][ty], a1 = As[cur][kk][16 + ty];
			const uint4 b0 = Bs[cur][kk][tx], b1 = Bs[cur][kk][16 + tx];
			const int av[8] = { (int)a0.x, (int)a0.y, (int)a0.z, (int)a0.w, (int)a1.x, (int)a1.y, (int)a1.z, (int)a1.w };
			const int bv[8] = { (int)b0.x, (int)b0.y, (int)b0.z, (int)b0.w, (int)b1.x, (int)b1.y, (int)b1.z, (int)b1.w };
			#pragma unroll
			for (int i = 0; i < 8; i++)
				#pragma unroll
				for (int j = 0; j < 8; j++) acc[i][j] = __dp4a(av[i], bv[j], acc[i][j]);
		}
		if (++chunk_step == steps_per_chunk) {
			chunk_step = 0;
			uint32_t x = 0;
			#pragma unroll
			for (int i = 0; i < 8; i++)
				#pragma unroll
				for (int j = 0; j < 8; j++) x ^= (uint32_t)acc[i][j];
			if (WIN == PEARL_WIN_REG) {
				const uint32_t nw = ((win[0] << 13) | (win[0] >> 19)) ^ x;
				#pragma unroll
				for (int i = 0; i < 15; i++) win[i] = win[i + 1];
				win[15] = nw;
			} else {
				const uint32_t q = chunk & 15;
				jl[q] = ((jl[q] << 13) | (jl[q] >> 19)) ^ x;
			}
			chunk++;
		}
		if (s + 1 < steps) {
			uint4 *da = sa0 + (cur ^ 1) * buf_stride, *db = sb0 + (cur ^ 1) * buf_stride;
			da[tid] = ra0; da[tid + 256] = ra1; db[tid] = rb0; db[tid + 256] = rb1;
		}
		__syncthreads();
	}

	if (WIN == PEARL_WIN_LOCAL) {
		#pragma unroll
		for (int i = 0; i < 16; i++) win[i] = jl[i];
	}
	/* after C chunks win[q] holds word (C + q) % 16: rotate right by C % 16 */
	if (WIN == PEARL_WIN_REG) {
		const uint32_t rot = (steps / steps_per_chunk) & 15;
		for (uint32_t t = 0; t < rot; t++) {
			const uint32_t last = win[15];
			#pragma unroll
			for (int i = 15; i > 0; i--) win[i] = win[i - 1];
			win[0] = last;
		}
	}

	uint32_t hw[8];
	blake3_keyed_block(p.a_seed.w, win, hw);
	const uint32_t ri = bm * 16 + ty, ci = bn * 16 + tx;
	if (dbg) {
		uint32_t *d = dbg + ((size_t)ri * (p.n / 8) + ci) * 24;
		#pragma unroll
		for (int i = 0; i < 16; i++) d[i] = win[i];
		#pragma unroll
		for (int i = 0; i < 8; i++) d[16 + i] = hw[i];
	}
	if (pearl_meets_words(hw, p.bound)) {
		const uint32_t slot = atomicAdd(&hits[0], 1);
		if (slot < PEARL_MAX_HITS) {
			hits[1 + 2 * slot] = bm * PEARL_BT + ty * 4;
			hits[2 + 2 * slot] = bn * PEARL_BT + tx * 4;
		}
	}
}

#define pearl_tiles_dp4a_kernel pearl_tiles_dp4a_t<PEARL_WIN_REG>

/* 16-byte global -> shared async copy (sm_80+) */
__device__ __forceinline__ void pearl_cp_async16(void *smem, const void *gmem)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
	const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
	asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(s), "l"(gmem));
#endif
}

/* ===========================================================================
 * Wide IMMA kernel: warp tile 64 x 64 (4 x 8 m16n8k32 fragments), block 128 x 64*WN,
 * 2 x WN warps. A thread's accumulators (rows {g, g+8} + 16i, cols {2t, 2t+1} + 8j) are
 * one hash tile of PEARL_IMMA_ROWS x PEARL_IMMA_W_COLS. Panels in PEARL_PERM_IMMA_A /
 * _B64, copied straight to smem, one panel block per pipeline stage. The next slice's
 * fragments load while the current slice's MMAs run; the barrier sits before the last
 * slice. GM: row blocks per raster group.
 */
#define PEARL_IMMA_W_STEP_U4	(PEARL_SUB * PEARL_BT / 4)	/* 512 uint4 = one panel block step */

__host__ __device__ constexpr uint32_t pearl_imma_w_smem(int WN, int STAGES)
{
	return (uint32_t)STAGES * (1 + WN / 2) * PEARL_IMMA_W_STEP_U4 * 16;
}

template <int WN, int STAGES, int GM>
__global__ void __launch_bounds__(64 * WN, WN == 4 ? 1 : 2)
pearl_tiles_imma_w_t(pearl_gemm_params p, const uint4 *Ap, const uint4 *Bp, uint32_t *dbg, uint32_t *hits)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
	extern __shared__ uint4 pearl_w_smem[];
	constexpr uint32_t T = 64 * WN, BN = 64 * WN, NPB = WN / 2;
	constexpr uint32_t STEP = PEARL_IMMA_W_STEP_U4, STAGE = STEP * (1 + NPB), CPT = STEP / T;

	const uint32_t tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
	const uint32_t g = lane >> 2, t = lane & 3;
	const uint32_t wm = warp / WN, wn = warp % WN;
	const uint32_t nbm = p.m / PEARL_BT, nbn = p.n / BN;
	const uint32_t in_group = GM * nbn, pid = blockIdx.x;	/* GM row blocks per raster group */
	const uint32_t first = (pid / in_group) * GM;
	const uint32_t gsz = min(nbm - first, (uint32_t)GM);
	const uint32_t bm = first + (pid % in_group) % gsz;
	const uint32_t bn = (pid % in_group) / gsz;

	const uint32_t KW = p.k >> 2;
	const uint32_t steps = KW / PEARL_SUB;
	const uint32_t steps_per_chunk = (p.r >> 2) / PEARL_SUB;
	const size_t panel_u4 = (size_t)KW * (PEARL_BT / 4);	/* one 128-row panel */
	const uint4 *ga = Ap + bm * panel_u4 + tid;
	const uint4 *gb = Bp + bn * NPB * panel_u4 + tid;

	auto issue = [&](uint32_t s, uint32_t stage) {
		uint4 *d = pearl_w_smem + stage * STAGE + tid;
		const size_t so = (size_t)s * STEP;
		#pragma unroll
		for (int h = 0; h < (int)CPT; h++)
			pearl_cp_async16(d + T * h, ga + so + T * h);
		#pragma unroll
		for (int pb = 0; pb < (int)NPB; pb++)
			#pragma unroll
			for (int h = 0; h < (int)CPT; h++)
				pearl_cp_async16(d + (1 + pb) * STEP + T * h, gb + pb * panel_u4 + so + T * h);
	};

	int acc[4][8][4];	/* [i m-frag][j n-frag][d0..d3] */
	#pragma unroll
	for (int i = 0; i < 4; i++)
		#pragma unroll
		for (int j = 0; j < 8; j++)
			#pragma unroll
			for (int e = 0; e < 4; e++) acc[i][j][e] = 0;
	uint32_t win[16];
	#pragma unroll
	for (int i = 0; i < 16; i++) win[i] = 0;

	#pragma unroll
	for (uint32_t s = 0; s < STAGES - 1; s++) {
		if (s < steps) issue(s, s);
		asm volatile("cp.async.commit_group;");
	}

	const uint32_t aq = wm * 128 + lane, bq = (1 + (wn >> 1)) * STEP + (wn & 1) * 128 + lane;
	uint32_t rd = 0, wr = STAGES - 1, chunk_step = 0;
	/* jackpot fold of the chunk that ended with the previous step; asm volatile keeps ptxas
	 * from if-converting it into every step */
	auto fold_now = [&]() {
		uint32_t x = 0;
		#pragma unroll
		for (int i = 0; i < 4; i++)
			#pragma unroll
			for (int j = 0; j < 8; j++) {
				uint32_t y;
				asm volatile("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(y) : "r"(acc[i][j][0]), "r"(acc[i][j][1]), "r"(acc[i][j][2]));
				asm volatile("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(x) : "r"(x), "r"(y), "r"(acc[i][j][3]));
			}
		const uint32_t nw = ((win[0] << 13) | (win[0] >> 19)) ^ x;
		#pragma unroll
		for (int i = 0; i < 15; i++) win[i] = win[i + 1];
		win[15] = nw;
	};
	uint4 fa[2][4];
	uint32_t fb[2][8][2];
	auto load = [&](int slot, const uint4 *sq, int sl) {
		#pragma unroll
		for (int jp = 0; jp < 4; jp++) {
			const uint4 v = sq[bq + sl * 256 + 32 * jp];
			fb[slot][2 * jp][0] = v.x; fb[slot][2 * jp][1] = v.y;
			fb[slot][2 * jp + 1][0] = v.z; fb[slot][2 * jp + 1][1] = v.w;
		}
		#pragma unroll
		for (int i = 0; i < 4; i++) fa[slot][i] = sq[aq + sl * 256 + 32 * i];
	};
	asm volatile("cp.async.wait_group %0;" :: "n"(STAGES - 2));
	__syncthreads();
	load(0, pearl_w_smem, 0);
	for (uint32_t s = 0; s < steps; s++) {
		if (s != 0 && chunk_step == 0) fold_now();
		const uint4 *sq = pearl_w_smem + rd * STAGE;
		rd = rd == STAGES - 1 ? 0 : rd + 1;
		const uint4 *sn = pearl_w_smem + rd * STAGE;
		/* stage wr held step s - 1, whose last fragments were loaded before the last barrier */
		if (s + STAGES - 1 < steps) issue(s + STAGES - 1, wr);
		asm volatile("cp.async.commit_group;");
		wr = wr == STAGES - 1 ? 0 : wr + 1;
		#pragma unroll
		for (int sl = 0; sl < 2; sl++) {
			const int cur = sl & 1;
			if (sl == 1) {
				asm volatile("cp.async.wait_group %0;" :: "n"(STAGES - 2));	/* step s + 1 landed */
				__syncthreads();
				load(cur ^ 1, sn, 0);	/* harmless past the last step */
			} else
				load(cur ^ 1, sq, sl + 1);
			#pragma unroll
			for (int i = 0; i < 4; i++)
				#pragma unroll
				for (int j = 0; j < 8; j++)
					asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
						"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
						: "+r"(acc[i][j][0]), "+r"(acc[i][j][1]), "+r"(acc[i][j][2]), "+r"(acc[i][j][3])
						: "r"(fa[cur][i].x), "r"(fa[cur][i].y), "r"(fa[cur][i].z), "r"(fa[cur][i].w),
						  "r"(fb[cur][j][0]), "r"(fb[cur][j][1]));
		}
		chunk_step = chunk_step + 1 == steps_per_chunk ? 0 : chunk_step + 1;
	}
	if (steps != 0) fold_now();	/* the last chunk (k is a multiple of r) */
	asm volatile("cp.async.wait_group 0;");

	const uint32_t rot = (steps / steps_per_chunk) & 15;
	for (uint32_t q = 0; q < rot; q++) {
		const uint32_t last = win[15];
		#pragma unroll
		for (int i = 15; i > 0; i--) win[i] = win[i - 1];
		win[0] = last;
	}

	uint32_t hw[8];
	blake3_keyed_block(p.a_seed.w, win, hw);
	const uint32_t ro = bm * PEARL_BT + wm * 64 + g;	/* tile row offset (pattern min) */
	const uint32_t co = bn * BN + wn * 64 + 2 * t;		/* tile col offset */
	if (dbg) {
		const uint32_t ri = (ro / 64) * 8 + ro % 64, ci = (co / 64) * 4 + (co % 64) / 2;
		uint32_t *d = dbg + ((size_t)ri * (p.n / 16) + ci) * 24;
		#pragma unroll
		for (int i = 0; i < 16; i++) d[i] = win[i];
		#pragma unroll
		for (int i = 0; i < 8; i++) d[16 + i] = hw[i];
	}
	if (pearl_meets_words(hw, p.bound)) {
		const uint32_t slot = atomicAdd(&hits[0], 1);
		if (slot < PEARL_MAX_HITS) { hits[1 + 2 * slot] = ro; hits[2 + 2 * slot] = co; }
	}
#endif
}

/* shipping shape: 128 x 128 block, 4 warps, 2 CTAs/SM, 3 stages, pipelined; 48 KB of
 * dynamic smem, so no opt-in attribute is needed */
#define PEARL_IMMA_WN		2
#define PEARL_IMMA_W_STAGES	3
#define PEARL_IMMA_W_GM		4
#define PEARL_IMMA_W_SMEM	pearl_imma_w_smem(PEARL_IMMA_WN, PEARL_IMMA_W_STAGES)
static_assert(pearl_imma_w_smem(PEARL_IMMA_WN, PEARL_IMMA_W_STAGES) <= 48 * 1024, "needs cudaFuncAttributeMaxDynamicSharedMemorySize");

#endif /* PEARL_DEVICE_CUH */
