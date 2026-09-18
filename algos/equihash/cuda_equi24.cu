// ---------------------------------------------------------------------------
// Round-templated Equihash solver for the DIGITBITS=24 family (144/5, 192/7).
//
// Geometry lives in equi24_params.h, the arena layout in equi24_arena.h. Reuses
// blake2b_gpu_hash_regs (blake2b_tromp.cuh), the register-resident final block,
// and a Cantor pairing for slot references -- the formula differs from djeZo's,
// see equi24_arena.h.
// ---------------------------------------------------------------------------

#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "blake2/blake2.h"
#include "equi24_params.h"
#include "equi24_arena.h"

// Both variants build from this file as separate TUs with different
// -DEQ_WN/-DEQ_WK. The namespace and every extern "C" entry point are keyed on
// EQ_WN so the two objects link side by side.
namespace EQ24_NS {

typedef uint32_t u32;
typedef uint64_t u64;
typedef uint16_t u16;
typedef unsigned char uchar;

#include "blake2b_tromp.cuh"   // ROR2/G/ROUND + blake2b_gpu_hash_regs

// little-endian byte extraction from the u64 blake output; j must be a
// compile-time constant so the words stay in registers.
#define EQ24_BYTE(w, j) ((u32)((w)[(j) >> 3] >> (((j) & 7) * 8)) & 0xffu)

// unaligned little-endian dword out of the u64 blake output; j must be a
// compile-time constant so this folds to shifts rather than memory traffic
#define EQ24_WORD32(w, j) ((((j) & 7) <= 4)   ? (u32)((w)[(j) >> 3] >> (((j) & 7) * 8))   : (u32)(((w)[(j) >> 3] >> (((j) & 7) * 8)) | ((w)[((j) >> 3) + 1] << (64 - ((j) & 7) * 8))))


// Widest aligned access for EQ_SLOT_W words. EQ_SLOT_W is a multiple of 4 and
// every slot is 16-byte aligned by construction, so these are STG.128/LDG.128
// rather than the byte loop the first version used.
__device__ __forceinline__ void eq24_store_slot(u32 *dst, const u32 *src)
{
#pragma unroll
	for (u32 v = 0; v < EQ_SLOT_W / 4u; v++)
		*(uint4 *)(dst + v * 4u) = make_uint4(src[v*4+0], src[v*4+1], src[v*4+2], src[v*4+3]);
}

__device__ __forceinline__ void eq24_load_slot(u32 *dst, const u32 *src)
{
#pragma unroll
	for (u32 v = 0; v < EQ_SLOT_W / 4u; v++) {
		const uint4 q = *(const uint4 *)(src + v * 4u);
		dst[v*4+0] = q.x; dst[v*4+1] = q.y; dst[v*4+2] = q.z; dst[v*4+3] = q.w;
	}
}


// Warp-aggregated append. Lanes targeting the SAME counter elect one leader
// which performs a single atomicAdd for the group; each lane then takes
// base + its rank within the group.
//
// __match_any_sync is sm_70+ and this project ships sm_61/75/86, so the Pascal
// fallback is required, not optional. With destinations random over 65536
// buckets the groups are almost always size 1, so aggregation is an
// optimisation rather than a requirement.
#ifndef EQ24_NO_WARP_AGG
#define EQ24_NO_WARP_AGG 0   // 1 forces the per-lane path everywhere, for A/B
#endif

__device__ __forceinline__ u32 eq24_append(u32 *ctr, u32 key, u32 lane)
{
#if __CUDA_ARCH__ >= 700 && !EQ24_NO_WARP_AGG
	const u32 active = __activemask();
	const u32 same = __match_any_sync(active, key);
	const u32 leader = __ffs(same) - 1u;
	const u32 rank = __popc(same & ((1u << lane) - 1u));
	u32 base = 0;
	if (lane == leader) base = atomicAdd(ctr, __popc(same));
	return __shfl_sync(same, base, leader) + rank;
#else
	(void)key; (void)lane;
	return atomicAdd(ctr, 1u);
#endif
}

// ---------------------------------------------------------------------------
// digitH -- blake2b over the nonce space, bucketed by digit 0.
//
// Each blake2b call yields EQ_HASHESPERBLAKE hashes of EQ_WN/8 bytes. Of each
// hash, the top EQ_BUCKBITS (= 16 = exactly two bytes, because DIGITBITS=24 is
// byte-aligned) select the bucket; the remaining EQ_PAYLOAD(0) bytes are the
// payload, whose leading byte is the 8 rest bits round 1 collides on.
//
// No __launch_bounds__, deliberately: the register-resident blake needs ~90
// registers and capping it spills v[16] to local memory, which is the
// bottleneck this path exists to remove. Occupancy measured irrelevant on the
// equivalent tromp kernel (18.9% -> 60.7% bought -2%).
// ---------------------------------------------------------------------------

__global__ void digitH(const u64 *__restrict__ blake_h,
                       u64 m0, u64 m1lo, u64 counter,
                       u32 *__restrict__ heap,          // layer 0 lives here
                       u32 *__restrict__ counts,
                       u32 *__restrict__ overflow)
{
	const u32 id = blockIdx.x * blockDim.x + threadIdx.x;
	const u32 stride = gridDim.x * blockDim.x;

	u64 h[8], hh[7];
#pragma unroll
	for (int i = 0; i < 8; i++)
		h[i] = blake_h[i];

	for (u32 blk = id; blk < EQ_NBLAKES; blk += stride) {
		blake2b_gpu_hash_regs(h, m0, m1lo | ((u64)blk << 32), counter, hh);

#pragma unroll
		for (u32 i = 0; i < EQ_HASHESPERBLAKE; i++) {
			const int b = (EQ_WN / 8) * i;
			const u32 index = blk * EQ_HASHESPERBLAKE + i;

			// EQ_NBLAKES rounds UP, so the final blake emits hashes past
			// the end of the nonce space when HASHESPERBLAKE does not
			// divide NHASHES -- true for 144/5 (3 hashes/blake into 2^25).
			// Index 2^25 needs 26 bits but solutions pack indices in
			// DIGITBITS+1 = 25, so it truncates to 0: a proof our own
			// verifier accepts and the pool rejects. Guard it here rather
			// than at submit time.
			if (index >= EQ_NHASHES)
				continue;

			// top 16 bits of the 144/192-bit hash = two whole bytes
			const u32 bucket = (EQ24_BYTE(hh, b) << 8) | EQ24_BYTE(hh, b + 1);

			// Round 0 has no source bucket, so the partition is free; it only
			// has to stay balanced. The index is independent of the bucket
			// (the bucket is a hash of it), so its low bits are uniform within
			// any bucket -- which is what the capacity margin assumes.
			const u32 part = index & (EQ_NPARTS - 1u);

			u32 *ctr = &counts[(size_t)bucket * EQ_NPARTS + part];

			// Warp-aggregated append: lanes targeting the same counter
			// elect a leader that performs one atomicAdd for the group.
			// The counter is in global memory, so the saving is in atomic
			// traffic.
			const u32 slot = eq24_append(ctr, (u32)((size_t)ctr >> 2), threadIdx.x & 31);

			if (slot >= EQ_SLOTS_PER_PART) {
				// Counted, never silent: an overflowing stream is solution loss
				// and it is the quantity the capacity margin was derived from.
				atomicAdd(overflow, 1u);
				continue;
			}

			// One contiguous record: [ref | payload], inside a single sector.
			// Round 0's reference IS the blake index -- it has no parents.
			const size_t at = EQ_SLOT_INDEX(bucket, part, slot);
			u32 *rec = EQ_REC(heap, at, 0);
			rec[0] = index;
#pragma unroll
			for (u32 k = 0; k < (EQ_PAYLOAD(0) + 3u) / 4u; k++)
				rec[1 + k] = (u32)EQ24_WORD32(hh, b + 2 + 4 * k);
		}
	}
}

// ---------------------------------------------------------------------------
// digitR<R> -- one collision round. One block per source bucket.
//
// Two phases separated by a barrier, which is the structure all three fast
// solvers share and the fused tromp path does not:
//
//   A  stage the bucket into shared, counting-sort its items by the 8 rest
//      bits, then emit every colliding (i<j) pair into a shared pair list
//   B  the WHOLE block drains that list -- XOR, trivial-pair test, destination
//      bucket, warp-aggregated scatter
//
// Decoupling matters because phase A is branchy with a serial dependence
// through the sort, while phase B is uniform and embarrassingly parallel. Fused
// (tromp) the long-latency scatter sits inside the detection chain: measured
// ~93% long-scoreboard stalls at 19% occupancy. Split, a rest value with 8
// collisions and one with 0 no longer imbalance anything, because the list is
// flat and shared.
//
// Payload contract, after XOR of a colliding pair (all byte-aligned because
// DIGITBITS=24 is exactly 3 bytes):
//   byte 0    the matched rest byte, MUST be zero
//   byte 1..2 top 16 bits of digit R -> destination bucket
//   byte 3..  the new payload
// ---------------------------------------------------------------------------

#define EQ24_PIN(R)      EQ_PAYLOAD((R) - 1)
#define EQ24_POUT(R)     EQ_PAYLOAD(R)
#define EQ24_PIN_W(R)    ((EQ24_PIN(R) + 3) / 4)
#define EQ24_POUT_W(R)   ((EQ24_POUT(R) + 3) / 4)

// Fused detect/scatter is the measured default here (rounds 111.6 -> 103.3 ms,
// shared 24568 -> 16416 B, 5-7 blocks/SM instead of 4), against the usual
// advice to decouple the two through an explicit pair list.
//
// Decoupling exists to lift a serial linked-list chase out of the scatter path,
// which is what tromp's per-thread collisiondata pointer chain needs. This
// detector is a counting sort by rest byte: groups are contiguous and average
// two items, so a thread emits ~1 pair. With no chase to hide, the pair list is
// just an extra barrier plus 8 KB of the binding resource.
//
// Set to 0 for the decoupled form; both are correct.
#ifndef EQ24_FUSED
#define EQ24_FUSED 1
#endif
#define EQ24_TPB     256
// Pair-list capacity is DERIVED from the shared budget, not picked. Ampere
// sm_86 offers 100 KB of shared per SM and the driver reserves 1 KB per block
// on top of the static allocation, so N blocks/SM allows (102400 - N*1024)/N
// bytes. For N=4 that is 24576 -- a list sized by eye to a round 1024 entries
// lands at 24616 and silently caps the kernel at 3 blocks/SM.
#define EQ24_SHARED_PER_SM   102400u
#define EQ24_DRIVER_SHARED   1024u
#define EQ24_TARGET_BLOCKS   4u
#define EQ24_SHARED_BUDGET   ((EQ24_SHARED_PER_SM - EQ24_TARGET_BLOCKS * EQ24_DRIVER_SHARED) / EQ24_TARGET_BLOCKS)

#define EQ24_FIXED_SHARED(R) (EQ_NSLOTS * EQ24_PIN_W(R) * 4u          /* s_pay */ \
                            + EQ_NRESTS * 4u * 2u               /* counts   */ \
                            + EQ_NSLOTS * 2u                    /* s_order  */ \
                            + EQ_NPARTS * 4u + 16u)             /* s_cnt ++ */
#define EQ24_MAXPAIR_R(R)    ((EQ24_SHARED_BUDGET - EQ24_FIXED_SHARED(R)) / 4u)

template <u32 R>
__global__ void digitR(const u32 *__restrict__ heapIn,   // holds layer R-1
                       u32 *__restrict__ heapOut,         // receives layer R
                       const u32 *__restrict__ cntIn,
                       u32 *__restrict__ cntOut,
                       u32 *__restrict__ overflow)
{
	const u32 PINW = EQ24_PIN_W(R), POUTW = EQ24_POUT_W(R);
	const u32 bucket = blockIdx.x;
	const u32 tid = threadIdx.x;

	// Staged BY SLOT, not by a compacted item index. Costs a pass over the
	// 1024-slot capacity instead of the ~512 live items, and buys three things:
	// the slot id is the array index so no s_slot[] table is needed (-2 KB),
	// thread k reads slot k so the staging load is exactly coalesced, and the
	// per-item partition search disappears.
	//
	// The inner stride is NOT padded to break shared-memory banking. The +1
	// stride is the standard fix and this tree's tromp digitWB carries it, but
	// measured here it made conflicts WORSE (16.9M -> 19.1M) and cost 4 KB of
	// the binding resource: the hot access is s_pay[ia][w] with ia from the
	// pair enumeration, i.e. random slots, and padding only re-maps a
	// systematic stride.
	__shared__ u32 s_pay[EQ_NSLOTS][EQ24_PIN_W(R)];
	__shared__ u32 s_restCnt[EQ_NRESTS];
	__shared__ u32 s_restPos[EQ_NRESTS];
	__shared__ u16 s_order[EQ_NSLOTS];       // slot ids grouped by rest
#if !EQ24_FUSED
	__shared__ u32 s_pairs[EQ24_MAXPAIR_R(R)];
#endif
	__shared__ u32 s_cnt[EQ_NPARTS];
#if !EQ24_FUSED
	__shared__ u32 s_np, s_drop;
#endif
#ifdef EQ24_ABLATE_REFS
	__shared__ u32 s_sink;
#endif

#if !EQ24_FUSED
	if (tid == 0) { s_np = 0; s_drop = 0; }
#endif
#ifdef EQ24_ABLATE_REFS
	if (tid == 0) s_sink = 0;
#endif
	for (u32 i = tid; i < EQ_NRESTS; i += EQ24_TPB) s_restCnt[i] = 0;
	if (tid < EQ_NPARTS) {
		const u32 c = cntIn[(size_t)bucket * EQ_NPARTS + tid];
		s_cnt[tid] = (c < EQ_SLOTS_PER_PART) ? c : EQ_SLOTS_PER_PART;
	}
	__syncthreads();                                              // [1]

	// ---- stage + histogram ------------------------------------------------
	for (u32 slot = tid; slot < EQ_NSLOTS; slot += EQ24_TPB) {
		const u32 p = slot / EQ_SLOTS_PER_PART;
		if ((slot & (EQ_SLOTS_PER_PART - 1u)) >= s_cnt[p]) continue;   // empty
		// The RECORD is 4-byte aligned (staggered) but the SLOT is 32-byte
		// aligned, so read the whole slot as two uint4 and pick the words
		// out: PINW scalar loads become 2 vector loads.
		const uint4 *q = (const uint4 *)(heapIn + ((size_t)bucket * EQ_NSLOTS + slot) * EQ_HEAP_W);
		u32 sw[EQ_HEAP_W];
		{ const uint4 q0 = q[0], q1 = q[1];
		  sw[0]=q0.x; sw[1]=q0.y; sw[2]=q0.z; sw[3]=q0.w;
		  sw[4]=q1.x; sw[5]=q1.y; sw[6]=q1.z; sw[7]=q1.w; }
#pragma unroll
		for (u32 w = 0; w < PINW; w++) s_pay[slot][w] = sw[EQ_LAYER_OFF(R - 1) + 1 + w];
		atomicAdd(&s_restCnt[s_pay[slot][0] & 0xffu], 1u);
	}
	__syncthreads();                                              // [2]

	// Serial over EQ_NRESTS on one thread. Priced by redundant repetition at
	// under 1%: with 65536 blocks in flight, a serial section inside one block
	// is covered by the thousands of others.
	if (tid == 0) {
		u32 acc = 0;
		for (u32 r = 0; r < EQ_NRESTS; r++) { s_restPos[r] = acc; acc += s_restCnt[r]; }
	}
	__syncthreads();                                              // [3]

	for (u32 slot = tid; slot < EQ_NSLOTS; slot += EQ24_TPB) {
		const u32 p = slot / EQ_SLOTS_PER_PART;
		if ((slot & (EQ_SLOTS_PER_PART - 1u)) >= s_cnt[p]) continue;
		s_order[atomicAdd(&s_restPos[s_pay[slot][0] & 0xffu], 1u)] = (u16)slot;
	}
	__syncthreads();                                              // [4]

	// ---- phase A: emit colliding pairs -------------------------------------
	// No prefix REBUILD pass. After the scatter s_restPos[r] holds the END of
	// group r and s_restCnt[r] is still live, so the start is end - count.
	// That removes a serial pass and a barrier with it.
#if !EQ24_FUSED
	for (u32 r = tid; r < EQ_NRESTS; r += EQ24_TPB) {
		const u32 cnt = s_restCnt[r], start = s_restPos[r] - cnt;
		for (u32 a = 0; a < cnt; a++)
			for (u32 b = a + 1; b < cnt; b++) {
				const u32 slot = atomicAdd(&s_np, 1u);
				if (slot < EQ24_MAXPAIR_R(R)) s_pairs[slot] = ((u32)s_order[start + a] << 16) | s_order[start + b];
				else atomicAdd(&s_drop, 1u);
			}
	}
	__syncthreads();                                              // [5]

	const u32 np = s_np < EQ24_MAXPAIR_R(R) ? s_np : EQ24_MAXPAIR_R(R);
	if (tid == 0 && s_drop) atomicAdd(overflow + 1, s_drop);
#endif

	// ---- phase B: the whole block drains the list --------------------------
	const u32 part = bucket >> EQ_REFBUCKBITS;

#if EQ24_FUSED
	// FUSED: each thread enumerates its own rest group and scatters inline --
	// no pair list, no barrier [5].
	for (u32 r = tid; r < EQ_NRESTS; r += EQ24_TPB) {
	  const u32 gcnt = s_restCnt[r], gstart = s_restPos[r] - gcnt;
	  for (u32 ga = 0; ga < gcnt; ga++)
	  for (u32 gb = ga + 1; gb < gcnt; gb++) {
		const u32 ia = s_order[gstart + ga], ib = s_order[gstart + gb];
#else
	for (u32 e = tid; e < np; e += EQ24_TPB) {
		const u32 ia = s_pairs[e] >> 16, ib = s_pairs[e] & 0xffffu;
#endif

		u32 x[EQ24_PIN_W(R)];
#pragma unroll
		for (u32 w = 0; w < PINW; w++) x[w] = s_pay[ia][w] ^ s_pay[ib][w];

		if ((x[0] & 0xffu) != 0u) continue;

		const u32 dst = (((x[0] >> 8) & 0xffu) << 8) | ((x[0] >> 16) & 0xffu);

		u32 y[EQ24_POUT_W(R) + 1];
#pragma unroll
		for (u32 w = 0; w < POUTW; w++)
			y[w] = (x[w] >> 24) | ((w + 1 < PINW) ? (x[w + 1] << 8) : 0u);

		u32 nz = 0;
#pragma unroll
		for (u32 w = 0; w < POUTW; w++) nz |= y[w];
		if (!nz) continue;

		u32 *ctr = &cntOut[(size_t)dst * EQ_NPARTS + part];
		const u32 slot = eq24_append(ctr, dst, tid & 31);
		if (slot >= EQ_SLOTS_PER_PART) { atomicAdd(overflow, 1u); continue; }

		const size_t at = EQ_SLOT_INDEX(dst, part, slot);
		u32 *rec = EQ_REC(heapOut, at, R);
#if !defined(EQ24_ABLATE_REFS)
		rec[0] = eq24_ref_pack(bucket, ia, ib);
#else
		// Ablation only. Keeps the Cantor packing so the delta isolates the
		// random GLOBAL store rather than the arithmetic feeding it: the
		// packed word goes to a shared dummy. Proofs are unreconstructable here.
		s_sink |= eq24_ref_pack(bucket, ia, ib);
#endif

		// Payload follows the reference in the SAME record, so the whole item
		// is one write region inside one sector.
#pragma unroll
		for (u32 w = 0; w < POUTW; w++) {
			u32 v = y[w];
			const u32 keep = EQ24_POUT(R) - w * 4u;
			if (keep < 4u) v &= (1u << (keep * 8u)) - 1u;
			rec[1 + w] = v;
		}
	}
#if EQ24_FUSED
	}
#endif
#ifdef EQ24_ABLATE_REFS
	__syncthreads();
	if (tid == 0 && s_sink == 0xffffffffu) atomicAdd(overflow + 2, 1u);   // 1 store/block
#endif
}

// ---------------------------------------------------------------------------
// digitK -- the final round. Wagner's last step cancels TWO digits at once.
//
// Entering it an item carries EQ_PAYLOAD(WK-1) bytes: the 8 rest bits of digit
// WK-1 plus the whole of digit WK. Two items in the same bucket already agree
// on the 16 bucket bits, so a full solution is exactly a pair whose remaining
// payload XORs to zero -- no scatter, just detect and report.
//
// Wagner puts ~2 real solutions in the whole arena per nonce, so this kernel
// finds almost nothing almost all of the time; that is expected.
// ---------------------------------------------------------------------------

#define EQ24_KW  ((EQ_PAYLOAD(EQ_WK - 1) + 3) / 4)

__global__ void digitK(const u32 *__restrict__ heapIn,   // holds layer WK-1
                       const u32 *__restrict__ cntIn,
                       u32 *__restrict__ sols,     // [maxsols][3] = bucket, s0, s1
                       u32 *__restrict__ nsols,
                       u32 maxsols)
{
	const u32 bucket = blockIdx.x, tid = threadIdx.x;

	__shared__ u32 s_pay[EQ_NSLOTS][EQ24_KW];
	__shared__ u16 s_slot[EQ_NSLOTS];
	__shared__ u32 s_restCnt[EQ_NRESTS];
	__shared__ u32 s_restPos[EQ_NRESTS];
	__shared__ u16 s_order[EQ_NSLOTS];
	__shared__ u32 s_base[EQ_NPARTS + 1];
	__shared__ u32 s_n;

	if (tid == 0) s_n = 0;
	for (u32 i = tid; i < EQ_NRESTS; i += EQ24_TPB) s_restCnt[i] = 0;
	if (tid == 0) {
		u32 acc = 0;
		for (u32 p = 0; p < EQ_NPARTS; p++) {
			s_base[p] = acc;
			const u32 c = cntIn[(size_t)bucket * EQ_NPARTS + p];
			acc += (c < EQ_SLOTS_PER_PART) ? c : EQ_SLOTS_PER_PART;
		}
		s_base[EQ_NPARTS] = acc; s_n = acc;
	}
	__syncthreads();
	const u32 n = s_n;

	for (u32 k = tid; k < n; k += EQ24_TPB) {
		u32 p = 0;
		while (p + 1 < EQ_NPARTS && k >= s_base[p + 1]) p++;
		const u32 slot = p * EQ_SLOTS_PER_PART + (k - s_base[p]);
		const u32 *rec = EQ_REC(heapIn, (size_t)bucket * EQ_NSLOTS + slot, EQ_WK - 1);
#pragma unroll
		for (u32 w = 0; w < EQ24_KW; w++) s_pay[k][w] = rec[1 + w];
		s_slot[k] = (u16)slot;
		atomicAdd(&s_restCnt[s_pay[k][0] & 0xffu], 1u);
	}
	__syncthreads();
	if (tid == 0) { u32 a = 0; for (u32 r = 0; r < EQ_NRESTS; r++) { s_restPos[r] = a; a += s_restCnt[r]; } }
	__syncthreads();
	for (u32 k = tid; k < n; k += EQ24_TPB)
		s_order[atomicAdd(&s_restPos[s_pay[k][0] & 0xffu], 1u)] = (u16)k;
	__syncthreads();
	if (tid == 0) { u32 a = 0; for (u32 r = 0; r < EQ_NRESTS; r++) { s_restPos[r] = a; a += s_restCnt[r]; } }
	__syncthreads();

	for (u32 r = tid; r < EQ_NRESTS; r += EQ24_TPB) {
		const u32 cnt = s_restCnt[r], start = s_restPos[r];
		for (u32 a = 0; a < cnt; a++)
			for (u32 b = a + 1; b < cnt; b++) {
				const u32 ia = s_order[start + a], ib = s_order[start + b];
				u32 nz = 0;
#pragma unroll
				for (u32 w = 0; w < EQ24_KW; w++) nz |= s_pay[ia][w] ^ s_pay[ib][w];
				if (nz) continue;                       // not a full collision
				const u32 at = atomicAdd(nsols, 1u);
				if (at < maxsols) {
					sols[at * 3 + 0] = bucket;
					sols[at * 3 + 1] = s_slot[ia];
					sols[at * 3 + 2] = s_slot[ib];
				}
			}
	}
}

} // namespace EQ24_NS

// ---- host entry points ----------------------------------------------------

extern "C" size_t EQ24_API(arena_bytes)(void) { return EQ_ARENA_BYTES; }
extern "C" unsigned EQ24_API(nbuckets)(void)  { return EQ_NBUCKETS; }
extern "C" unsigned EQ24_API(nslots)(void)    { return EQ_NSLOTS; }
extern "C" unsigned EQ24_API(nparts)(void)    { return EQ_NPARTS; }
extern "C" unsigned EQ24_API(payload0)(void)  { return EQ_PAYLOAD(0); }
extern "C" unsigned EQ24_API(nblakes)(void)   { return EQ_NBLAKES; }

// Runs digitH over the whole nonce space for one header. `state` is the
// 128-byte-absorbed midstate produced by the host blake2b (buflen must be 12,
// i.e. a 140-byte header+nonce, which is what stratum always delivers).
extern "C" int EQ24_API(digitH)(const void *blake_state,
                             unsigned *d_heap,
                             unsigned *d_counts, unsigned *d_overflow,
                             int blocks, int tpb)
{
	const blake2b_state *st = (const blake2b_state *)blake_state;
	if (st->buflen != 12)
		return -1;

	const uint64_t m0 = *(const uint64_t *)(st->buf);
	const uint64_t m1lo = *(const uint32_t *)(st->buf + 8);
	const uint64_t counter = st->counter + 16;   // 12 buffered + 4-byte index

	uint64_t *d_h = NULL;
	if (cudaMalloc(&d_h, 8 * sizeof(uint64_t)) != cudaSuccess) return -2;
	cudaMemcpy(d_h, st->h, 8 * sizeof(uint64_t), cudaMemcpyHostToDevice);

	cudaMemset(d_counts, 0, (size_t)EQ_NBUCKETS * EQ_NPARTS * sizeof(unsigned));
	cudaMemset(d_overflow, 0, sizeof(unsigned));

	EQ24_NS::digitH<<<blocks, tpb>>>(d_h, m0, m1lo, counter,
	                                d_heap, d_counts, d_overflow);
	cudaError_t e = cudaDeviceSynchronize();
	cudaFree(d_h);
	return e == cudaSuccess ? 0 : -3;
}

// Launch one collision round. Exposed individually so each round can be gated
// in isolation rather than only end-to-end.
#define EQ24_TPB_HOST 256

extern "C" int EQ24_API(digitR)(int round,
                             const unsigned *d_heapIn, unsigned *d_heapOut,
                             const unsigned *d_cntIn,
                             unsigned *d_cntOut, unsigned *d_overflow)
{
	cudaMemset(d_cntOut, 0, (size_t)EQ_NBUCKETS * EQ_NPARTS * sizeof(unsigned));
	switch (round) {
#define EQ24_CASE(R) case R: \
	EQ24_NS::digitR<R><<<EQ_NBUCKETS, EQ24_TPB_HOST>>>(d_heapIn, d_heapOut, d_cntIn, d_cntOut, d_overflow); break;
	EQ24_CASE(1) EQ24_CASE(2) EQ24_CASE(3) EQ24_CASE(4)
#if EQ_WK > 5
	EQ24_CASE(5) EQ24_CASE(6)
#endif
#undef EQ24_CASE
	default: return -1;
	}
	return cudaDeviceSynchronize() == cudaSuccess ? 0 : -3;
}

// Final round. Returns the number of candidate solutions found (may exceed
// maxsols, in which case the surplus was counted but not stored -- the caller
// must report that rather than silently clamp).
extern "C" int EQ24_API(digitK)(const unsigned *d_heapIn, const unsigned *d_cntIn,
                             unsigned *d_sols, unsigned *d_nsols, unsigned maxsols)
{
	cudaMemset(d_nsols, 0, sizeof(unsigned));
	EQ24_NS::digitK<<<EQ_NBUCKETS, EQ24_TPB_HOST>>>(d_heapIn, d_cntIn, d_sols, d_nsols, maxsols);
	if (cudaDeviceSynchronize() != cudaSuccess) return -3;
	unsigned n = 0;
	cudaMemcpy(&n, d_nsols, sizeof(unsigned), cudaMemcpyDeviceToHost);
	return (int)n;
}

extern "C" unsigned EQ24_API(proofsize)(void) { return EQ_PROOFSIZE; }
extern "C" unsigned EQ24_API(heap_w)(void)    { return EQ_HEAP_W; }
extern "C" size_t EQ24_API(heap_bytes)(void){ return EQ_HEAP_BYTES; }

// ---------------------------------------------------------------------------
// Proof assembly, on the device.
//
// Walking the reference tree host-side needs the heaps copied back (4.3 GB) or
// ~62 scattered 4-byte reads per candidate, so one thread per candidate expands
// its own proof here and the host copies back only the finished index lists.
//
// Ordering is consensus-relevant: Equihash binds the proof to Wagner's flow, so
// each pair of sub-lists must lead with the smaller index. Done bottom-up after
// a raw expansion, which is equivalent to the recursive form the verifier
// expects: for each block size 2^L, if the left half's first index exceeds the
// right half's, swap the halves.
// ---------------------------------------------------------------------------

namespace EQ24_NS {

__global__ void expandSols(const u32 *__restrict__ heap0,
                           const u32 *__restrict__ heap1,
                           const u32 *__restrict__ sols, u32 nsols,
                           u32 *__restrict__ out)        // [nsols][EQ_PROOFSIZE]
{
	const u32 t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= nsols) return;

	u32 idx[EQ_PROOFSIZE];
	u32 bkt[EQ_PROOFSIZE], slt[EQ_PROOFSIZE];

	// digitK hands us two slots of the final layer, in one bucket
	bkt[0] = sols[t * 3 + 0]; slt[0] = sols[t * 3 + 1];
	bkt[1] = sols[t * 3 + 0]; slt[1] = sols[t * 3 + 2];
	u32 n = 2;

	// walk down: each node at layer L becomes two nodes at layer L-1
	for (int layer = (int)EQ_WK - 1; layer >= 1; layer--) {
		const u32 *heap = EQ_LAYER_HEAP(layer) ? heap1 : heap0;
		for (int i = (int)n - 1; i >= 0; i--) {
			const size_t at = (size_t)bkt[i] * EQ_NSLOTS + slt[i];
			const u32 ref = heap[at * EQ_HEAP_W + EQ_LAYER_OFF(layer)];
			const u32 part = slt[i] / EQ_SLOTS_PER_PART;
			const u32 src = eq24_ref_bucket(ref, part);
			u32 s0, s1; eq24_ref_slots(ref, &s0, &s1);
			bkt[2*i] = src; slt[2*i] = s0;
			bkt[2*i+1] = src; slt[2*i+1] = s1;
		}
		n *= 2;
	}

	// layer 0 stores the blake index itself
	const u32 *h0 = EQ_LAYER_HEAP(0) ? heap1 : heap0;
	for (u32 i = 0; i < n; i++)
		idx[i] = h0[((size_t)bkt[i] * EQ_NSLOTS + slt[i]) * EQ_HEAP_W + EQ_LAYER_OFF(0)];

	// bottom-up canonical ordering
	for (u32 blk = 2; blk <= EQ_PROOFSIZE; blk <<= 1)
		for (u32 base = 0; base < EQ_PROOFSIZE; base += blk)
			if (idx[base] > idx[base + blk / 2])
				for (u32 k = 0; k < blk / 2; k++) {
					const u32 tmp = idx[base + k];
					idx[base + k] = idx[base + blk / 2 + k];
					idx[base + blk / 2 + k] = tmp;
				}

	for (u32 i = 0; i < EQ_PROOFSIZE; i++)
		out[t * EQ_PROOFSIZE + i] = idx[i];
}

} // namespace EQ24_NS

extern "C" int EQ24_API(expand)(const unsigned *d_heap0, const unsigned *d_heap1,
                             const unsigned *d_sols, unsigned nsols, unsigned *d_out)
{
	if (!nsols) return 0;
	EQ24_NS::expandSols<<<(nsols + 63) / 64, 64>>>(d_heap0, d_heap1, d_sols, nsols, d_out);
	return cudaDeviceSynchronize() == cudaSuccess ? 0 : -1;
}

// ---------------------------------------------------------------------------
// Miner-facing C API. Mirrors tromp144_init/solve/free so the dispatch in
// equihash.cpp is a one-line switch and the two solvers stay swappable.
// equihash.cpp re-verifies every solution before submit, picking the verifier
// that matches the variant.
// ---------------------------------------------------------------------------

#include <stdlib.h>
#include <string.h>

#ifndef htole32
#define htole32(x) (x)
#endif

namespace EQ24_NS {

struct solver_ctx {
	uint32_t *heap[2];   // staggered arenas; layer r lives in heap[r & 1]
	uint32_t *cnt[2];
	uint32_t *over;      // [0] slot drops, [1] pair drops, [2] spare
	uint32_t *sols;      // [MAXSOL][3] = bucket, s0, s1
	uint32_t *nsols;
	uint32_t *idx;       // [MAXSOL][PROOFSIZE] expanded proofs
	uint32_t *hidx;      // host mirror
	uint32_t  maxsol;
};

static void setheader24(blake2b_state *ctx, const char *hdr, const char *personal)
{
	char p16[16];
	memcpy(p16, personal, 8);
	uint32_t n = htole32(EQ_WN), k = htole32(EQ_WK);
	memcpy(p16 + 8, &n, 4);
	memcpy(p16 + 12, &k, 4);
	blake2b_param P[1];
	memset(P, 0, sizeof(blake2b_param));
	P->digest_length = EQ_HASHOUT;
	P->fanout = 1;
	P->depth = 1;
	memcpy(P->personal, (const uint8_t *)p16, 16);
	eq_blake2b_init_param(ctx, P);
	eq_blake2b_update(ctx, (const uint8_t *)hdr, 140);
}

} // namespace EQ24_NS

extern "C" void EQ24_API(free)(void *ctx);

extern "C" void *EQ24_API(init)(void)
{
	EQ24_NS::solver_ctx *c = (EQ24_NS::solver_ctx *)calloc(1, sizeof(EQ24_NS::solver_ctx));
	if (!c) return NULL;
	c->maxsol = 64;

	// ~4.3 GiB of arena: materially more than the tromp path's 2.5 GiB, so a
	// card that ran 144/5 before may not run this. Fail here with a number
	// rather than somewhere later with a null pointer.
	bool ok = true;
	for (int i = 0; i < 2 && ok; i++) {
		ok &= cudaMalloc(&c->heap[i], EQ_HEAP_BYTES) == cudaSuccess;
		ok &= cudaMalloc(&c->cnt[i], (size_t)EQ_NBUCKETS * EQ_NPARTS * sizeof(uint32_t)) == cudaSuccess;
	}
	ok &= cudaMalloc(&c->over, 4 * sizeof(uint32_t)) == cudaSuccess;
	ok &= cudaMalloc(&c->sols, c->maxsol * 3 * sizeof(uint32_t)) == cudaSuccess;
	ok &= cudaMalloc(&c->nsols, sizeof(uint32_t)) == cudaSuccess;
	ok &= cudaMalloc(&c->idx, (size_t)c->maxsol * EQ_PROOFSIZE * sizeof(uint32_t)) == cudaSuccess;
	c->hidx = (uint32_t *)malloc((size_t)c->maxsol * EQ_PROOFSIZE * sizeof(uint32_t));
	if (!ok || !c->hidx) {
		fprintf(stderr, "equi24: device allocation failed (needs ~%.2f GiB)\n",
		        (double)(2.0 * EQ_HEAP_BYTES) / (1 << 30));
		EQ24_API(free)(c);
		return NULL;
	}
	return c;
}

extern "C" void EQ24_API(free)(void *ctx)
{
	EQ24_NS::solver_ctx *c = (EQ24_NS::solver_ctx *)ctx;
	if (!c) return;
	for (int i = 0; i < 2; i++) {
		if (c->heap[i]) cudaFree(c->heap[i]);
		if (c->cnt[i])  cudaFree(c->cnt[i]);
	}
	if (c->over)  cudaFree(c->over);
	if (c->sols)  cudaFree(c->sols);
	if (c->nsols) cudaFree(c->nsols);
	if (c->idx)   cudaFree(c->idx);
	if (c->hidx)  free(c->hidx);
	free(c);
}

extern "C" size_t EQ24_API(arena_needed)(void) { return 2 * (size_t)EQ_HEAP_BYTES; }

// Solve one 140-byte header+nonce. Emits each DISTINCT, non-trivial proof.
// Returns the count, or -1 on error.
extern "C" int EQ24_API(solve)(void *ctx, const char *headernonce, const char *personal,
                            void (*emit)(void *, const uint32_t *, uint32_t), void *ud)
{
	EQ24_NS::solver_ctx *c = (EQ24_NS::solver_ctx *)ctx;
	if (!c) return -1;

	blake2b_state mid;
	EQ24_NS::setheader24(&mid, headernonce, personal);
	cudaMemset(c->over, 0, 4 * sizeof(uint32_t));

	if (EQ24_API(digitH)(&mid, c->heap[0], c->cnt[0], c->over, 2048, 128)) return -1;
	for (unsigned R = 1; R < EQ_WK; R++)
		if (EQ24_API(digitR)(R, c->heap[(R - 1) & 1], c->heap[R & 1],
		                  c->cnt[(R - 1) & 1], c->cnt[R & 1], c->over)) return -1;

	const int found = EQ24_API(digitK)(c->heap[(EQ_WK - 1) & 1], c->cnt[(EQ_WK - 1) & 1],
	                                c->sols, c->nsols, c->maxsol);
	if (found < 0) return -1;
	const unsigned store = (unsigned)found < c->maxsol ? (unsigned)found : c->maxsol;
	if (!store) return 0;

	if (EQ24_API(expand)(c->heap[0], c->heap[1], c->sols, store, c->idx)) return -1;
	cudaMemcpy(c->hidx, c->idx, (size_t)store * EQ_PROOFSIZE * sizeof(uint32_t), cudaMemcpyDeviceToHost);

	// Trivial-solution filter. NOT optional: a proof of the form (A^B) ^ (A^C)
	// repeats a leaf, satisfies every XOR condition, and can never be a share.
	// tromp does the same in ctx_solve/duped().
	int emitted = 0;
	for (unsigned s = 0; s < store; s++) {
		const uint32_t *p = c->hidx + (size_t)s * EQ_PROOFSIZE;
		uint32_t sorted[EQ_PROOFSIZE];
		memcpy(sorted, p, sizeof(sorted));
		for (unsigned i = 1; i < EQ_PROOFSIZE; i++)          // insertion sort, PROOFSIZE items
			for (unsigned j = i; j && sorted[j - 1] > sorted[j]; j--) {
				const uint32_t t = sorted[j]; sorted[j] = sorted[j - 1]; sorted[j - 1] = t;
			}
		bool dup = false;
		for (unsigned i = 1; i < EQ_PROOFSIZE && !dup; i++) dup = sorted[i] == sorted[i - 1];
		if (dup) continue;
		if (emit) emit(ud, p, EQ_PROOFSIZE);
		emitted++;
	}
	return emitted;
}
