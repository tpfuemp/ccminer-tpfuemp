// ---------------------------------------------------------------------------
// equi24b -- Equihash solver for the DIGITBITS=24 variants (144/5, 192/7) on a
// heap of few large buckets. See equi24b_params.h for the geometry and why.
//
// Stages: digitH builds layer 0, digitR<R> collides each subsequent layer,
// digitK tests the last one for a full cancellation, expandSols walks the
// retained layers back to leaf indices.
// ---------------------------------------------------------------------------

#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "blake2/blake2.h"
#include "equi24b_params.h"

// blake2b_tromp.cuh uses these unqualified, so they must precede it
typedef uint32_t u32;
typedef uint64_t u64;
typedef uint16_t u16;
typedef unsigned char uchar;


#define EQ24B_CAT2(a, b) a##b
#define EQ24B_CAT(a, b)  EQ24B_CAT2(a, b)
#define EQ24B_NS         EQ24B_CAT(equi24b_, EQ_WN)
#define EQ24B_API(name)  EQ24B_CAT(EQ24B_CAT(equi24b_, EQ_WN), EQ24B_CAT(_, name))

#define EQ24B_BYTE(w, j) ((u32)((w)[(j) >> 3] >> (((j) & 7) * 8)) & 0xffu)

#define EQ24B_HASHESPERBLAKE (512 / EQ_WN)
#define EQ24B_NHASHES        (2u * (1u << EQ24B_DIGITBITS))
#define EQ24B_NBLAKES        ((EQ24B_NHASHES + EQ24B_HASHESPERBLAKE - 1) / EQ24B_HASHESPERBLAKE)
#define EQ24B_L0_STRIDE_W    (EQ24B_STRIDE(0) / 4u)

namespace EQ24B_NS {

// Included INSIDE the namespace, and necessarily BELOW the EQ24B_NS macro:
// this header defines __device__ functions at file scope with external
// linkage, so at global scope the 144/5 and 192/7 objects both emit them and
// the link fails. The u32/u64/uchar typedefs stay global -- the header finds
// them by outward lookup, and the extern "C" wrappers below need them too.
#include "blake2b_tromp.cuh"   // ROR2/G/ROUND + blake2b_gpu_hash_regs

// Warp-aggregated append: lanes hitting the same counter elect one leader that
// does a single atomicAdd for the group. With 4096 buckets instead of 65536 the
// same-destination hit rate is high enough that this matters.
__device__ __forceinline__ u32 eq24b_append(u32 *ctr, u32 key, u32 lane)
{
#if __CUDA_ARCH__ >= 700
	// MUST be the active mask, not 0xffffffff: both call sites sit after a
	// `continue`, so naming absent lanes is undefined behaviour -- it hung the
	// GPU at 100% with no progress rather than failing visibly.
	const u32 act = __activemask();
	const u32 peers = __match_any_sync(act, key);
	const u32 leader = __ffs(peers) - 1u;
	const u32 rank = __popc(peers & ((1u << lane) - 1u));
	u32 base = 0u;
	if (lane == leader) base = atomicAdd(ctr, (u32)__popc(peers));
	base = __shfl_sync(peers, base, leader);
	return base + rank;
#else
	// Pascal has no __match_any_sync: ballot repeatedly over the distinct keys
	// the warp holds. sm_61 is a shipped target, so this path is not optional.
	u32 active = __ballot(1);
	u32 peers = 0u, base = 0u;
	while (active) {
		const u32 ldr = (u32)__ffs(active) - 1u;
		const u32 k = __shfl(key, ldr);
		const u32 same = __ballot(k == key);
		if (k == key) peers = same;
		active &= ~same;
	}
	const u32 leader = (u32)__ffs(peers) - 1u;
	const u32 rank = __popc(peers & ((1u << lane) - 1u));
	if (lane == leader) base = atomicAdd(ctr, (u32)__popc(peers));
	base = __shfl(base, leader);
	return base + rank;
#endif
}

// ---------------------------------------------------------------------------
// digitH -- blake2b over the nonce space, bucketed by the top EQ24B_BUCKBITS
// bits of digit 0. One independent layer array: no stagger, no partitions.
// ---------------------------------------------------------------------------
__global__ void digitH(const u64 *__restrict__ blake_h,
                       u64 m0, u64 m1lo, u64 counter,
                       u32 *__restrict__ layer0,
                       u32 *__restrict__ counts,
                       u32 *__restrict__ overflow)
{
	const u32 id = blockIdx.x * blockDim.x + threadIdx.x;
	const u32 stride = gridDim.x * blockDim.x;

	u64 h[8], hh[7];
#pragma unroll
	for (int i = 0; i < 8; i++) h[i] = blake_h[i];

	for (u32 blk = id; blk < EQ24B_NBLAKES; blk += stride) {
		blake2b_gpu_hash_regs(h, m0, m1lo | ((u64)blk << 32), counter, hh);

#pragma unroll
		for (u32 i = 0; i < EQ24B_HASHESPERBLAKE; i++) {
			const int b = (EQ_WN / 8) * (int)i;
			const u32 index = blk * EQ24B_HASHESPERBLAKE + i;
			// NBLAKES rounds up, so the last blake emits hashes past the end of
			// the nonce space when HASHESPERBLAKE does not divide NHASHES. Those
			// indices do not fit the solution encoding; drop them here rather
			// than let a proof our own verifier accepts be rejected by a pool.
			if (index >= EQ24B_NHASHES) continue;

			const u32 d0 = (EQ24B_BYTE(hh, b) << 16) | (EQ24B_BYTE(hh, b + 1) << 8)
			             | EQ24B_BYTE(hh, b + 2);
			const u32 bucket = (d0 >> EQ24B_RESTBITS) & EQ24B_BUCKMASK;

			const u32 slot = eq24b_append(&counts[bucket], bucket, threadIdx.x & 31);
			if (slot >= EQ24B_NSLOTS) {          // counted, never silent
				atomicAdd(overflow, 1u);
				continue;
			}

			// One 32 B record: [index | pad | payload]. Two uint4 stores, not a
			// scalar loop: a scattered store costs a whole sector transaction
			// whatever its width, so only the NUMBER of requests matters.
			u32 *rec = layer0 + (size_t)(bucket * EQ24B_NSLOTS + slot) * EQ24B_L0_STRIDE_W;
			u32 w[8];
#pragma unroll
			for (u32 k = 0; k < 8u; k++) w[k] = 0u;
			w[0] = index;
			// Payload: the bytes after the one holding the bucket index.
			// Bytes past the end of THIS hash must be zeroed, not taken -- the
			// blake output holds HASHESPERBLAKE hashes back to back, so reading
			// past byte EQ_WN/8 silently pulls in the neighbour's bits. The
			// payload is 17 B for 144/5, so the last word is 1 valid byte + 3
			// that belong to the next hash.
#pragma unroll
			for (u32 k = 0; k < 6u; k++) {
				if (k * 4u >= EQ24B_PAYLOAD(0)) break;
				u32 v = 0u;
#pragma unroll
				for (u32 j = 0; j < 4u; j++) {
					const u32 bi = 1u + k * 4u + j;          // byte within this hash
					if (bi < (u32)(EQ_WN / 8) && bi <= EQ24B_PAYLOAD(0))
						v |= (u32)EQ24B_BYTE(hh, b + (int)bi) << (8u * j);
				}
				w[2 + k] = v;
			}
			*(uint4 *)(rec + 0) = make_uint4(w[0], w[1], w[2], w[3]);
			*(uint4 *)(rec + 4) = make_uint4(w[4], w[5], w[6], w[7]);
		}
	}
}


// stride of a layer in words, device side (expandSols walks several layers)
__device__ __forceinline__ u32 eq24b_stride_w(int layer)
{
	switch (layer) {
	case 0: return EQ24B_STRIDE(0u) / 4u;
	case 1: return EQ24B_STRIDE(1u) / 4u;
	case 2: return EQ24B_STRIDE(2u) / 4u;
	case 3: return EQ24B_STRIDE(3u) / 4u;
#if EQ_WK > 5
	case 4: return EQ24B_STRIDE(4u) / 4u;
	case 5: return EQ24B_STRIDE(5u) / 4u;
	default: return EQ24B_STRIDE(6u) / 4u;
#else
	default: return EQ24B_STRIDE(4u) / 4u;
#endif
	}
}

// ---------------------------------------------------------------------------
// digitR<R> -- one collision round.
//
// Layout convention, fixed by digitH and preserved every round: a payload's
// first two bytes carry the 12 REST bits of the digit just consumed, in the low
// nibble of byte 0 and all of byte 1. Two items in the same bucket collide iff
// those match, so XORing a colliding pair zeroes payload bytes 0 and 1; the next
// digit is then payload bytes 2,3,4, its bucket is the top 12 of those, and the
// next payload starts at byte 3. PAYLOAD(r) = PAYLOAD(r-1) - 3 falls out of it.
//
// A bucket is processed in EQ24B_CHUNKS_R(R) passes split by rest PREFIX, which
// is lossless because items whose rest differs in the high bits cannot collide.
// Within a chunk, collisions are found by a per-rest singly-linked list built
// with atomicExch -- each item stores the previous head, so walking from an
// item's link visits exactly the items staged before it in its group and every
// unordered pair is enumerated once.
// ---------------------------------------------------------------------------
template <u32 R>
__global__ void digitR(const u32 *__restrict__ heapIn, u32 *__restrict__ heapOut,
                       const u32 *__restrict__ cntIn, u32 *__restrict__ cntOut,
                       u32 *__restrict__ overflow)
{
	const u32 bucket = blockIdx.x;
	const u32 tid = threadIdx.x, lane = tid & 31u;
	const u32 nin = min(cntIn[bucket], EQ24B_NSLOTS);

	const u32 PW   = EQ24B_STAGE_W(R);          // payload words staged
	const u32 SLOTS = EQ24B_STAGE_SLOTS(R);
	const u32 HEADS = EQ24B_HEADS(R);
	const u32 INW  = EQ24B_STRIDE(R - 1u) / 4u;  // input record stride, words
	const u32 OUTW = EQ24B_STRIDE(R) / 4u;

	extern __shared__ u32 sh[];
	u32 *s_pay  = sh;                              // [SLOTS][PW]
	u16 *s_next = (u16 *)(sh + SLOTS * PW);        // [SLOTS]
	u16 *s_slot = s_next + SLOTS;                  // [SLOTS] origin slot
	u32 *s_head = (u32 *)(s_slot + SLOTS);         // [HEADS], u32: no 16-bit atomicExch
	__shared__ u32 s_n;

	for (u32 chunk = 0; chunk < EQ24B_CHUNKS_R(R); chunk++) {
		for (u32 i = tid; i < HEADS; i += blockDim.x) s_head[i] = EQ24B_LIST_END;
		if (tid == 0) s_n = 0u;
		__syncthreads();

		// ---- pass A: stage the items whose rest prefix selects this chunk ----
		for (u32 slot = tid; slot < nin; slot += blockDim.x) {
			const u32 *rec = heapIn + (size_t)(bucket * EQ24B_NSLOTS + slot) * INW;
			const u32 p0 = rec[2];                       // payload word 0
			const u32 rest = ((p0 & 0x0fu) << 8) | ((p0 >> 8) & 0xffu);
			if (rest / HEADS != chunk) continue;

			const u32 s = eq24b_append(&s_n, chunk, lane);
			if (s >= SLOTS) { if (lane == 0) atomicAdd(overflow + 1, 1u); continue; }

			for (u32 w = 0; w < PW; w++) s_pay[s * PW + w] = rec[2 + w];
			s_slot[s] = (u16)slot;
			s_next[s] = (u16)atomicExch(&s_head[rest % HEADS], s);
		}
		__syncthreads();

		// ---- pass B: walk each item's list and emit its pairs ----------------
		const u32 nst = min(s_n, SLOTS);
		for (u32 s = tid; s < nst; s += blockDim.x) {
			for (u32 t = s_next[s]; t != EQ24B_LIST_END; t = s_next[t]) {
				u32 x[8];
				for (u32 w = 0; w < PW; w++) x[w] = s_pay[s * PW + w] ^ s_pay[t * PW + w];
				// A pair whose payload cancels ENTIRELY agrees on every remaining bit,
				// which only happens when the two subtrees share leaves. Dropping it
				// is not an optimisation: such a child has rest 0, so every one of
				// them lands in bucket 0 and pairs with the others, growing
				// quadratically per round until it swamps the final round.
				u32 nz = 0u;
				for (u32 w = 0; w < PW; w++) nz |= x[w];
				if (!nz) continue;
				// bytes 0,1 of the xor are zero by construction (same rest)
				const u32 b2 = (x[0] >> 16) & 0xffu, b3 = (x[0] >> 24) & 0xffu;
				const u32 dst = ((b2 << 4) | (b3 >> 4)) & EQ24B_BUCKMASK;

				const u32 os = eq24b_append(&cntOut[dst], dst, lane);
				if (os >= EQ24B_NSLOTS) { atomicAdd(overflow, 1u); continue; }

				u32 *out = heapOut + (size_t)(dst * EQ24B_NSLOTS + os) * OUTW;
				// reference: source bucket + Cantor of the two ORIGIN slots
				const u32 a = max(s_slot[s], s_slot[t]), bb = min(s_slot[s], s_slot[t]);
				const eq_u64 ref = ((eq_u64)bucket << EQ24B_CANTORBITS)
				                 | (((eq_u64)a * (a - 1u)) / 2u + bb);
				u32 o[8];
				for (u32 w = 0; w < 8u; w++) o[w] = 0u;
				o[0] = (u32)ref; o[1] = (u32)(ref >> 32);
				// next payload starts at byte 3 of the xor
				for (u32 w = 0; w * 4u < EQ24B_PAYLOAD(R) && w < 6u; w++) {
					u32 v = 0u;
					for (u32 j = 0; j < 4u; j++) {
						const u32 bi = 3u + w * 4u + j;
						// bytes past PAYLOAD(R) must be ZERO, not whatever the xor
						// left there: digitK tests the final payload for being
						// entirely zero, and stale padding would break it.
						if (w * 4u + j < EQ24B_PAYLOAD(R) && bi < PW * 4u)
							v |= ((x[bi >> 2] >> ((bi & 3u) * 8u)) & 0xffu) << (8u * j);
					}
					o[2 + w] = v;
				}
				*(uint4 *)(out + 0) = make_uint4(o[0], o[1], o[2], o[3]);
				if (OUTW > 4u) *(uint4 *)(out + 4) = make_uint4(o[4], o[5], o[6], o[7]);
			}
		}
		__syncthreads();
	}
}


// ---------------------------------------------------------------------------
// digitK -- the final round. After EQ_WK-1 collision rounds a payload holds the
// rest of the last-but-one digit plus the whole final digit: 36 bits for both
// variants. A pair in the same bucket is a solution iff ALL of it cancels, so
// this is digitR's machinery with the emit replaced by a zero test.
// ---------------------------------------------------------------------------
__global__ void digitK(const u32 *__restrict__ heapIn,
                       const u32 *__restrict__ cntIn,
                       u32 *__restrict__ sols, u32 *__restrict__ nsols, u32 maxsols,
                       u32 *__restrict__ overflow)
{
	const u32 bucket = blockIdx.x;
	const u32 tid = threadIdx.x, lane = tid & 31u;
	const u32 nin = min(cntIn[bucket], EQ24B_NSLOTS);

	const u32 PW = (EQ24B_PAYLOAD(EQ_WK - 1u) + 3u) / 4u;
	const u32 SLOTS = EQ24B_STAGE_SLOTS(EQ_WK - 1u);
	const u32 HEADS = EQ24B_HEADS(EQ_WK - 1u);
	const u32 INW = EQ24B_STRIDE(EQ_WK - 1u) / 4u;

	extern __shared__ u32 sh[];
	u32 *s_pay  = sh;
	u16 *s_next = (u16 *)(sh + SLOTS * PW);
	u16 *s_slot = s_next + SLOTS;
	u32 *s_head = (u32 *)(s_slot + SLOTS);
	__shared__ u32 s_n;

	for (u32 chunk = 0; chunk < EQ24B_CHUNKS_R(EQ_WK - 1u); chunk++) {
		for (u32 i = tid; i < HEADS; i += blockDim.x) s_head[i] = EQ24B_LIST_END;
		if (tid == 0) s_n = 0u;
		__syncthreads();

		for (u32 slot = tid; slot < nin; slot += blockDim.x) {
			const u32 *rec = heapIn + (size_t)(bucket * EQ24B_NSLOTS + slot) * INW;
			const u32 p0 = rec[2];
			const u32 rest = ((p0 & 0x0fu) << 8) | ((p0 >> 8) & 0xffu);
			if (rest / HEADS != chunk) continue;
			const u32 s = eq24b_append(&s_n, chunk, lane);
			// never silent: a dropped stage here is a LOST SOLUTION and looks
			// exactly like the solver being slightly unlucky.
			if (s >= SLOTS) { atomicAdd(overflow + 2, 1u); continue; }
			for (u32 w = 0; w < PW; w++) s_pay[s * PW + w] = rec[2 + w];
			s_slot[s] = (u16)slot;
			s_next[s] = (u16)atomicExch(&s_head[rest % HEADS], s);
		}
		__syncthreads();

		const u32 nst = min(s_n, SLOTS);
		for (u32 s = tid; s < nst; s += blockDim.x) {
			for (u32 t = s_next[s]; t != EQ24B_LIST_END; t = s_next[t]) {
				u32 z = 0u;
				for (u32 w = 0; w < PW; w++) z |= s_pay[s * PW + w] ^ s_pay[t * PW + w];
				// digitR zeroes payload padding, so a solution is simply an
				// all-zero xor across the staged words.
				if (z) continue;
				const u32 at = atomicAdd(nsols, 1u);
				if (at < maxsols) {
					sols[at * 3 + 0] = bucket;
					sols[at * 3 + 1] = s_slot[s];
					sols[at * 3 + 2] = s_slot[t];
				}
			}
		}
		__syncthreads();
	}
}

// ---------------------------------------------------------------------------
// expandSols -- walk the reference tree to the blake indices.
//
// The arena retains every layer, so a node can be read in full;
// only the 8-byte reference is actually needed. Ordering is consensus-relevant:
// each pair of sub-lists must lead with the smaller index, done bottom-up after
// a raw expansion.
// ---------------------------------------------------------------------------
__global__ void expandSols(u32 *const *__restrict__ layers,
                           const u32 *__restrict__ sols, u32 nsols,
                           u32 *__restrict__ out)
{
	const u32 t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= nsols) return;

	u32 idx[EQ24B_PROOFSIZE], bkt[EQ24B_PROOFSIZE], slt[EQ24B_PROOFSIZE];
	bkt[0] = sols[t * 3 + 0]; slt[0] = sols[t * 3 + 1];
	bkt[1] = sols[t * 3 + 0]; slt[1] = sols[t * 3 + 2];
	u32 n = 2;

	for (int layer = (int)EQ_WK - 1; layer >= 1; layer--) {
		const u32 *L = layers[layer];
		const u32 SW = eq24b_stride_w(layer);
		for (int i = (int)n - 1; i >= 0; i--) {
			const u32 *rec = L + (size_t)(bkt[i] * EQ24B_NSLOTS + slt[i]) * SW;
			const eq_u64 ref = ((eq_u64)rec[1] << 32) | rec[0];
			const u32 src = (u32)(ref >> EQ24B_CANTORBITS);
			const eq_u64 c = ref & EQ24B_CANTORMASK;
			u32 a = (u32)((1.0 + sqrt(1.0 + 8.0 * (double)c)) * 0.5);
			while (a > 1u && (eq_u64)a * (a - 1u) / 2u > c) a--;
			while ((eq_u64)(a + 1u) * a / 2u <= c) a++;
			const u32 b0 = (u32)(c - (eq_u64)a * (a - 1u) / 2u);
			bkt[2*i] = src; slt[2*i] = b0;
			bkt[2*i+1] = src; slt[2*i+1] = a;
		}
		n *= 2;
	}
	const u32 *L0 = layers[0];
	const u32 SW0 = eq24b_stride_w(0);
	for (u32 i = 0; i < n; i++)
		idx[i] = L0[(size_t)(bkt[i] * EQ24B_NSLOTS + slt[i]) * SW0];

	for (u32 blk = 2; blk <= EQ24B_PROOFSIZE; blk <<= 1)
		for (u32 base = 0; base < EQ24B_PROOFSIZE; base += blk)
			if (idx[base] > idx[base + blk / 2])
				for (u32 k = 0; k < blk / 2; k++) {
					const u32 tmp = idx[base + k];
					idx[base + k] = idx[base + blk / 2 + k];
					idx[base + blk / 2 + k] = tmp;
				}
	for (u32 i = 0; i < EQ24B_PROOFSIZE; i++) out[t * EQ24B_PROOFSIZE + i] = idx[i];
}

}  // namespace EQ24B_NS

// ---------------------------------------------------------------------------
// host side
// ---------------------------------------------------------------------------
extern "C" unsigned long long EQ24B_API(layer_bytes)(unsigned r)
{
	switch (r) {
	case 0: return (unsigned long long)EQ24B_LAYER_BYTES(0);
	case 1: return (unsigned long long)EQ24B_LAYER_BYTES(1);
	case 2: return (unsigned long long)EQ24B_LAYER_BYTES(2);
	case 3: return (unsigned long long)EQ24B_LAYER_BYTES(3);
	case 4: return (unsigned long long)EQ24B_LAYER_BYTES(4);
#if EQ_WK > 5
	case 5: return (unsigned long long)EQ24B_LAYER_BYTES(5);
	case 6: return (unsigned long long)EQ24B_LAYER_BYTES(6);
#endif
	default: return 0;
	}
}

extern "C" unsigned EQ24B_API(nbuckets)(void) { return EQ24B_NBUCKETS; }
extern "C" unsigned EQ24B_API(nslots)(void)   { return EQ24B_NSLOTS; }
extern "C" unsigned EQ24B_API(stride0)(void)  { return EQ24B_STRIDE(0); }

#include <stdlib.h>
#include <string.h>

// forward declarations: the solve path below calls these, and they are
// defined further down in this file.
extern "C" unsigned long long EQ24B_API(layer_bytes)(unsigned);
extern "C" unsigned EQ24B_API(shared)(unsigned);
extern "C" void EQ24B_API(free)(void *);
extern "C" int EQ24B_API(digitH)(const void *, unsigned *, unsigned *, unsigned *, int, int);
extern "C" int EQ24B_API(digitR)(int, const unsigned *, unsigned *, const unsigned *, unsigned *, unsigned *, int);

// ---------------------------------------------------------------------------
// solver context: every layer retained, which is what lets expandSols read a
// record instead of only a self-sufficient reference word.
// ---------------------------------------------------------------------------
namespace EQ24B_NS {
struct ctx {
	u32 *layer[EQ_WK];
	u32 *cnt[EQ_WK];
	u32 **d_layers;
	u32 *over, *sols, *nsols, *idx;
	u32 *hidx;
	u32 maxsol;
};
static void setheader(blake2b_state *c, const char *hdr, const char *personal)
{
	char p16[16];
	memcpy(p16, personal, 8);
	u32 n = EQ_WN, k = EQ_WK;
	memcpy(p16 + 8, &n, 4); memcpy(p16 + 12, &k, 4);
	blake2b_param P[1];
	memset(P, 0, sizeof(blake2b_param));
	P->digest_length = (512 / EQ_WN) * EQ_WN / 8;
	P->fanout = P->depth = 1;
	memcpy(P->personal, p16, 16);
	eq_blake2b_init_param(c, P);
	eq_blake2b_update(c, (const uint8_t *)hdr, 140);
}
}  // namespace EQ24B_NS

extern "C" void *EQ24B_API(init)(void)
{
	EQ24B_NS::ctx *c = (EQ24B_NS::ctx *)calloc(1, sizeof(EQ24B_NS::ctx));
	if (!c) return NULL;
	// Candidate capacity for digitK. NOT a solution count: digitK emits every
	// colliding pair and the host filter removes the trivial ones afterwards.
	// Few, large buckets mean large rest groups, so digitK emits far more
	// candidates per nonce than a many-bucket geometry would. Sizing this to
	// the solution count instead loses real solutions to an arrival-order
	// discard. It is ~10x the observed peak and costs 33.6 MB.
	c->maxsol = 262144;
	bool okk = true;
	u32 *hostptr[EQ_WK];
	for (unsigned r = 0; r < EQ_WK; r++) {
		const unsigned long long lb = EQ24B_API(layer_bytes)(r);
		if (!lb) { okk = false; break; }        // a layer with no size is a wiring bug
		okk &= cudaMalloc(&c->layer[r], (size_t)lb) == cudaSuccess;
		okk &= cudaMalloc(&c->cnt[r], (size_t)EQ24B_NBUCKETS * sizeof(u32)) == cudaSuccess;
		hostptr[r] = c->layer[r];
	}
	okk &= cudaMalloc(&c->d_layers, EQ_WK * sizeof(u32 *)) == cudaSuccess;
	okk &= cudaMalloc(&c->over, 4 * sizeof(u32)) == cudaSuccess;
	okk &= cudaMalloc(&c->sols, c->maxsol * 3 * sizeof(u32)) == cudaSuccess;
	okk &= cudaMalloc(&c->nsols, sizeof(u32)) == cudaSuccess;
	okk &= cudaMalloc(&c->idx, (size_t)c->maxsol * EQ24B_PROOFSIZE * sizeof(u32)) == cudaSuccess;
	c->hidx = (u32 *)malloc((size_t)c->maxsol * EQ24B_PROOFSIZE * sizeof(u32));
	if (!okk || !c->hidx) { EQ24B_API(free)(c); return NULL; }
	cudaMemcpy(c->d_layers, hostptr, EQ_WK * sizeof(u32 *), cudaMemcpyHostToDevice);
	return c;
}

extern "C" void EQ24B_API(free)(void *p)
{
	EQ24B_NS::ctx *c = (EQ24B_NS::ctx *)p;
	if (!c) return;
	for (unsigned r = 0; r < EQ_WK; r++) { if (c->layer[r]) cudaFree(c->layer[r]); if (c->cnt[r]) cudaFree(c->cnt[r]); }
	if (c->d_layers) cudaFree(c->d_layers);
	if (c->over) cudaFree(c->over);
	if (c->sols) cudaFree(c->sols);
	if (c->nsols) cudaFree(c->nsols);
	if (c->idx) cudaFree(c->idx);
	if (c->hidx) ::free(c->hidx);
	::free(c);
}

// Loss counters from the last solve, all silent without this accessor:
//   [0] digitR scatter overflow (destination bucket full)
//   [1] digitR staging overflow (chunk full)
//   [2] digitK staging overflow (chunk full)
//   [3] digitK candidates counted but not stored (maxsol clamp)
extern "C" void EQ24B_API(drops)(void *p, unsigned *out4)
{
	EQ24B_NS::ctx *c = (EQ24B_NS::ctx *)p;
	out4[0] = out4[1] = out4[2] = out4[3] = 0u;
	if (!c) return;
	cudaMemcpy(out4, c->over, 3 * sizeof(u32), cudaMemcpyDeviceToHost);
	u32 found = 0; cudaMemcpy(&found, c->nsols, sizeof(u32), cudaMemcpyDeviceToHost);
	out4[3] = found > c->maxsol ? found - c->maxsol : 0u;
}

extern "C" int EQ24B_API(solve)(void *p, const char *headernonce, const char *personal,
                                void (*emit)(void *, const uint32_t *, uint32_t), void *ud)
{
	EQ24B_NS::ctx *c = (EQ24B_NS::ctx *)p;
	if (!c) return -1;
	blake2b_state mid;
	EQ24B_NS::setheader(&mid, headernonce, personal);
	cudaMemset(c->over, 0, 4 * sizeof(u32));
	cudaMemset(c->nsols, 0, sizeof(u32));

	if (EQ24B_API(digitH)(&mid, c->layer[0], c->cnt[0], c->over, 2048, 128)) return -1;
	for (unsigned r = 1; r < EQ_WK; r++)
		if (EQ24B_API(digitR)((int)r, c->layer[r-1], c->layer[r], c->cnt[r-1], c->cnt[r], c->over, 256))
			return -1;

	const unsigned shK = EQ24B_API(shared)(EQ_WK - 1u);
	EQ24B_NS::digitK<<<EQ24B_NBUCKETS, 256, shK>>>(c->layer[EQ_WK-1], c->cnt[EQ_WK-1],
	                                               c->sols, c->nsols, c->maxsol, c->over);
	if (cudaGetLastError() != cudaSuccess) return -4;
	if (cudaDeviceSynchronize() != cudaSuccess) return -3;

	u32 found = 0;
	cudaMemcpy(&found, c->nsols, sizeof(u32), cudaMemcpyDeviceToHost);
	const u32 store = found < c->maxsol ? found : c->maxsol;
	if (!store) return 0;

	EQ24B_NS::expandSols<<<(store + 63) / 64, 64>>>(c->d_layers, c->sols, store, c->idx);
	if (cudaGetLastError() != cudaSuccess) return -4;
	if (cudaDeviceSynchronize() != cudaSuccess) return -3;
	cudaMemcpy(c->hidx, c->idx, (size_t)store * EQ24B_PROOFSIZE * sizeof(u32), cudaMemcpyDeviceToHost);

	// Trivial-solution filter: a proof repeating a leaf satisfies every XOR
	// condition and can never be a share.
	int emitted = 0;
	for (u32 s = 0; s < store; s++) {
		const u32 *q = c->hidx + (size_t)s * EQ24B_PROOFSIZE;
		u32 sorted[EQ24B_PROOFSIZE];
		memcpy(sorted, q, sizeof(sorted));
		for (u32 i = 1; i < EQ24B_PROOFSIZE; i++)
			for (u32 j = i; j && sorted[j-1] > sorted[j]; j--) {
				const u32 tv = sorted[j]; sorted[j] = sorted[j-1]; sorted[j-1] = tv;
			}
		bool dup = false;
		for (u32 i = 1; i < EQ24B_PROOFSIZE && !dup; i++) dup = sorted[i] == sorted[i-1];
		if (dup) continue;
		if (emit) emit(ud, q, EQ24B_PROOFSIZE);
		emitted++;
	}
	return emitted;
}

extern "C" unsigned EQ24B_API(chunks)(unsigned r)
{
	switch (r) { case 1: return EQ24B_CHUNKS_R(1u); case 2: return EQ24B_CHUNKS_R(2u);
	             case 3: return EQ24B_CHUNKS_R(3u); case 4: return EQ24B_CHUNKS_R(4u);
#if EQ_WK > 5
	             case 5: return EQ24B_CHUNKS_R(5u); case 6: return EQ24B_CHUNKS_R(6u);
#endif
	             default: return 0; }
}

// shared bytes a round needs: staged payloads + links + origin slots + heads
static unsigned eq24b_shared(unsigned r)
{
	switch (r) {
#define SH(R) case R: return EQ24B_STAGE_SLOTS(R##u) * EQ24B_STAGE_B(R##u) \
                           + EQ24B_HEADS(R##u) * 4u;
	SH(1) SH(2) SH(3) SH(4)
#if EQ_WK > 5
	SH(5) SH(6)
#endif
#undef SH
	default: return 0;
	}
}

extern "C" unsigned EQ24B_API(shared)(unsigned r) { return eq24b_shared(r); }

// Device bytes a context will try to allocate. Every layer is RETAINED, which
// is what lets expandSols read a record rather than only a reference word, so
// this grows with the round count and differs between the variants.
extern "C" unsigned long long EQ24B_API(arena_needed)(void)
{
	unsigned long long t = 0;
	for (unsigned r = 0; r < EQ_WK; r++)
		t += EQ24B_API(layer_bytes)(r) + (unsigned long long)EQ24B_NBUCKETS * sizeof(unsigned);
	// candidate buffers: pair records + expanded proofs + the host mirror's twin
	t += (unsigned long long)262144u * 3u * sizeof(unsigned);
	t += (unsigned long long)262144u * EQ24B_PROOFSIZE * sizeof(unsigned);
	return t;
}

extern "C" int EQ24B_API(digitR)(int r, const unsigned *d_in, unsigned *d_out,
                                 const unsigned *d_cntIn, unsigned *d_cntOut,
                                 unsigned *d_over, int tpb)
{
	cudaMemset(d_cntOut, 0, (size_t)EQ24B_NBUCKETS * sizeof(unsigned));
	const unsigned sh = eq24b_shared((unsigned)r);
	if (!sh) return -1;
	switch (r) {
	case 1: EQ24B_NS::digitR<1><<<EQ24B_NBUCKETS, tpb, sh>>>(d_in, d_out, d_cntIn, d_cntOut, d_over); break;
	case 2: EQ24B_NS::digitR<2><<<EQ24B_NBUCKETS, tpb, sh>>>(d_in, d_out, d_cntIn, d_cntOut, d_over); break;
	case 3: EQ24B_NS::digitR<3><<<EQ24B_NBUCKETS, tpb, sh>>>(d_in, d_out, d_cntIn, d_cntOut, d_over); break;
	case 4: EQ24B_NS::digitR<4><<<EQ24B_NBUCKETS, tpb, sh>>>(d_in, d_out, d_cntIn, d_cntOut, d_over); break;
#if EQ_WK > 5
	case 5: EQ24B_NS::digitR<5><<<EQ24B_NBUCKETS, tpb, sh>>>(d_in, d_out, d_cntIn, d_cntOut, d_over); break;
	case 6: EQ24B_NS::digitR<6><<<EQ24B_NBUCKETS, tpb, sh>>>(d_in, d_out, d_cntIn, d_cntOut, d_over); break;
#endif
	default: return -1;
	}
	// A launch that never starts (bad shared size, bad geometry) leaves
	// cudaDeviceSynchronize happy. Check the LAUNCH separately or a broken
	// kernel reads as "ran, produced nothing".
	const cudaError_t le = cudaGetLastError();
	if (le != cudaSuccess) return -4;
	return cudaDeviceSynchronize() == cudaSuccess ? 0 : -3;
}

extern "C" int EQ24B_API(digitH)(const void *blake_state,
                                 unsigned *d_layer0,
                                 unsigned *d_counts, unsigned *d_overflow,
                                 int blocks, int tpb)
{
	const blake2b_state *st = (const blake2b_state *)blake_state;
	if (st->buflen != 12) return -1;

	const u64 m0 = *(const u64 *)(st->buf);
	const u64 m1lo = *(const u32 *)(st->buf + 8);
	const u64 counter = st->counter + 16;      // 12 buffered + the 4-byte index

	u64 *d_h = NULL;
	if (cudaMalloc(&d_h, 8 * sizeof(u64)) != cudaSuccess) return -2;
	cudaMemcpy(d_h, st->h, 8 * sizeof(u64), cudaMemcpyHostToDevice);

	cudaMemset(d_counts, 0, (size_t)EQ24B_NBUCKETS * sizeof(unsigned));
	cudaMemset(d_overflow, 0, sizeof(unsigned));

	EQ24B_NS::digitH<<<blocks, tpb>>>(d_h, m0, m1lo, counter,
	                                  d_layer0, d_counts, d_overflow);
	const cudaError_t e = cudaDeviceSynchronize();
	cudaFree(d_h);
	return e == cudaSuccess ? 0 : -3;
}
