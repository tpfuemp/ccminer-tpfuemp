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

/* panel layout of X' = X + E_U[p0] - E_U[p1]; rows must be a multiple of 128 */
__global__ void pearl_operand_panel_kernel(const int8_t *base, const int8_t *U, const uint32_t *P,
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
	panel[idx] = res;
}

/* Panel prepass: one block = one 128-row panel x PEARL_PREP_KW k-words. U rows are
 * staged in smem with a 4-byte pad (bank-conflict free); base_panel may be NULL.
 * Needs r <= 256. */
#define PEARL_PREP_KW		64

__global__ void __launch_bounds__(256)
pearl_operand_panel_fast_kernel(const uint32_t *base_panel, const int8_t *U, const uint32_t *P,
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

	const uint32_t row = threadIdx.x % PEARL_BT;
	const int8_t *u = Us + row * ustride;
	for (uint32_t kk = threadIdx.x / PEARL_BT; kk < PEARL_PREP_KW && kw0 + kk < KW; kk += blockDim.x / PEARL_BT) {
		const uint32_t kw = kw0 + kk;
		const size_t o = ((size_t)rb * KW + kw) * PEARL_BT + row;
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
__global__ void pearl_patch_row0_kernel(const uint32_t *chunk, uint32_t *panel)
{
	panel[(size_t)threadIdx.x * PEARL_BT] = chunk[threadIdx.x];
}

/* base (rows x k, row-major int8) -> panel layout, once per job */
__global__ void pearl_to_panel_kernel(const int8_t *base, uint32_t rows, uint32_t k, uint32_t *panel)
{
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t KW = k >> 2;
	if (idx >= rows * KW) return;
	uint32_t rr = idx % PEARL_BT, rest = idx / PEARL_BT;
	uint32_t kw = rest % KW, rb = rest / KW;
	panel[idx] = *(const uint32_t *)(base + (size_t)(rb * PEARL_BT + rr) * k + kw * 4);
}

typedef struct {
	uint32_t m, n, k, r;		/* m, n multiples of 128; k multiple of 64; r multiple of 64 */
	pearl_key8 a_seed;
	uint32_t bound[8];
} pearl_gemm_params;

/* hits[0] = count; hits[1 + 2s] = tile row offset, hits[2 + 2s] = tile col offset.
 * dbg (optional): 24 words per tile in reference tile order (ri * n/8 + ci). */
__global__ void __launch_bounds__(256, 2)
pearl_tiles_dp4a_kernel(pearl_gemm_params p, const uint4 *Ap, const uint4 *Bp, uint32_t *dbg, uint32_t *hits)
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
	uint32_t win[16];
	#pragma unroll
	for (int i = 0; i < 16; i++) win[i] = 0;

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
			const uint32_t nw = ((win[0] << 13) | (win[0] >> 19)) ^ x;
			#pragma unroll
			for (int i = 0; i < 15; i++) win[i] = win[i + 1];
			win[15] = nw;
		}
		if (s + 1 < steps) {
			uint4 *da = sa0 + (cur ^ 1) * buf_stride, *db = sb0 + (cur ^ 1) * buf_stride;
			da[tid] = ra0; da[tid + 256] = ra1; db[tid] = rb0; db[tid + 256] = rb1;
		}
		__syncthreads();
	}

	/* after C chunks win[q] holds word (C + q) % 16: rotate right by C % 16 */
	const uint32_t rot = (steps / steps_per_chunk) & 15;
	for (uint32_t t = 0; t < rot; t++) {
		const uint32_t last = win[15];
		#pragma unroll
		for (int i = 15; i > 0; i--) win[i] = win[i - 1];
		win[0] = last;
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

/* ===========================================================================
 * IMMA kernel (mma.sync.m16n8k32.s8). Block tile 128 x 128, 8 warps as 2 x 4, warp
 * tile 64 x 32 = 4 x 4 fragments. A thread's accumulators (rows {g, g+8} + 16i,
 * cols {2t, 2t+1} + 8j) are one hash tile of PEARL_IMMA_ROWS x PEARL_IMMA_COLS.
 * smem word (kw, row) is stored at row ^ ((kw & 3) << 3), conflict-free for the
 * fragment loads. 3-stage cp.async pipeline, 48 KB static smem.
 */
#define PEARL_IMMA_STAGES	3

__device__ __forceinline__ void pearl_cp_async16(void *smem, const void *gmem)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
	const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
	asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(s), "l"(gmem));
#endif
}

__global__ void __launch_bounds__(256, 2)
pearl_tiles_imma_kernel(pearl_gemm_params p, const uint4 *Ap, const uint4 *Bp, uint32_t *dbg, uint32_t *hits)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
	__shared__ uint4 As[PEARL_IMMA_STAGES][PEARL_SUB][PEARL_BT / 4];
	__shared__ uint4 Bs[PEARL_IMMA_STAGES][PEARL_SUB][PEARL_BT / 4];

	const uint32_t tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
	const uint32_t g = lane >> 2, t = lane & 3;
	const uint32_t wm = warp >> 2, wn = warp & 3;		/* 2 x 4 warps */
	const uint32_t nbm = p.m / PEARL_BT, nbn = p.n / PEARL_BT;
	const uint32_t in_group = PEARL_GROUP_M * nbn, pid = blockIdx.x;
	const uint32_t first = (pid / in_group) * PEARL_GROUP_M;
	const uint32_t gsz = min(nbm - first, (uint32_t)PEARL_GROUP_M);
	const uint32_t bm = first + (pid % in_group) % gsz;
	const uint32_t bn = (pid % in_group) / gsz;

	const uint32_t KW = p.k >> 2;
	const uint32_t steps = KW / PEARL_SUB;
	const uint32_t steps_per_chunk = (p.r >> 2) / PEARL_SUB;
	const uint32_t step_u4 = PEARL_SUB * PEARL_BT / 4;	/* 512 uint4 per operand per step */
	const uint4 *ga = Ap + (size_t)bm * KW * (PEARL_BT / 4);
	const uint4 *gb = Bp + (size_t)bn * KW * (PEARL_BT / 4);

	/* this thread's two copy slots per operand: flat uint4 index q -> (kw, row4) */
	auto issue = [&](uint32_t s, uint32_t stage) {
		#pragma unroll
		for (int h = 0; h < 2; h++) {
			const uint32_t q = tid + 256 * h, kwi = q >> 5, r4 = q & 31;
			const uint32_t sw = r4 ^ ((kwi & 3) << 1);
			pearl_cp_async16(&As[stage][kwi][sw], ga + (size_t)s * step_u4 + q);
			pearl_cp_async16(&Bs[stage][kwi][sw], gb + (size_t)s * step_u4 + q);
		}
	};

	int acc[4][4][4];	/* [i m-frag][j n-frag][d0..d3] */
	#pragma unroll
	for (int i = 0; i < 4; i++)
		#pragma unroll
		for (int j = 0; j < 4; j++)
			#pragma unroll
			for (int e = 0; e < 4; e++) acc[i][j][e] = 0;
	uint32_t win[16];
	#pragma unroll
	for (int i = 0; i < 16; i++) win[i] = 0;

	#pragma unroll
	for (uint32_t s = 0; s < PEARL_IMMA_STAGES - 1; s++) {
		if (s < steps) issue(s, s);
		asm volatile("cp.async.commit_group;");
	}

	const uint32_t *Aw = (const uint32_t *)&As[0][0][0];
	const uint32_t *Bw = (const uint32_t *)&Bs[0][0][0];
	const uint32_t arow = wm * 64 + g, bcol = wn * 32 + g;
	uint32_t chunk_step = 0;
	for (uint32_t s = 0; s < steps; s++) {
		asm volatile("cp.async.wait_group %0;" :: "n"(PEARL_IMMA_STAGES - 2));
		__syncthreads();
		if (s + PEARL_IMMA_STAGES - 1 < steps) issue(s + PEARL_IMMA_STAGES - 1, (s + PEARL_IMMA_STAGES - 1) % PEARL_IMMA_STAGES);
		asm volatile("cp.async.commit_group;");

		const uint32_t stage = s % PEARL_IMMA_STAGES;
		const uint32_t *sa = Aw + stage * PEARL_SUB * PEARL_BT, *sb = Bw + stage * PEARL_SUB * PEARL_BT;
		#pragma unroll
		for (int sl = 0; sl < 2; sl++) {		/* two k32 slices per stage */
			const uint32_t k0 = sl * 8 + t, k1 = k0 + 4;	/* word t and t+4 of the slice */
			const uint32_t x0 = (k0 & 3) << 3, x1 = (k1 & 3) << 3;
			uint32_t af[4][4], bf[4][2];
			#pragma unroll
			for (int i = 0; i < 4; i++) {
				const uint32_t r0 = arow + 16 * i, r1 = r0 + 8;
				af[i][0] = sa[k0 * PEARL_BT + (r0 ^ x0)];
				af[i][1] = sa[k0 * PEARL_BT + (r1 ^ x0)];
				af[i][2] = sa[k1 * PEARL_BT + (r0 ^ x1)];
				af[i][3] = sa[k1 * PEARL_BT + (r1 ^ x1)];
			}
			#pragma unroll
			for (int j = 0; j < 4; j++) {
				const uint32_t c = bcol + 8 * j;
				bf[j][0] = sb[k0 * PEARL_BT + (c ^ x0)];
				bf[j][1] = sb[k1 * PEARL_BT + (c ^ x1)];
			}
			#pragma unroll
			for (int i = 0; i < 4; i++)
				#pragma unroll
				for (int j = 0; j < 4; j++)
					asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
						"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
						: "+r"(acc[i][j][0]), "+r"(acc[i][j][1]), "+r"(acc[i][j][2]), "+r"(acc[i][j][3])
						: "r"(af[i][0]), "r"(af[i][1]), "r"(af[i][2]), "r"(af[i][3]),
						  "r"(bf[j][0]), "r"(bf[j][1]));
		}
		if (++chunk_step == steps_per_chunk) {
			chunk_step = 0;
			uint32_t x = 0;
			#pragma unroll
			for (int i = 0; i < 4; i++)
				#pragma unroll
				for (int j = 0; j < 4; j++)
					#pragma unroll
					for (int e = 0; e < 4; e++) x ^= (uint32_t)acc[i][j][e];
			const uint32_t nw = ((win[0] << 13) | (win[0] >> 19)) ^ x;
			#pragma unroll
			for (int i = 0; i < 15; i++) win[i] = win[i + 1];
			win[15] = nw;
		}
	}
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
	const uint32_t co = bn * PEARL_BT + wn * 32 + 2 * t;	/* tile col offset */
	if (dbg) {
		const uint32_t ri = (ro / 64) * 8 + ro % 64, ci = (co / 32) * 4 + (co % 32) / 2;
		uint32_t *d = dbg + ((size_t)ri * (p.n / 8) + ci) * 24;
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

#endif /* PEARL_DEVICE_CUH */
