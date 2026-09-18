// ---------------------------------------------------------------------------
// Arena layout and reference-word packing for the DIGITBITS=24 solver.
//
// Items are AoS records held in two staggered heaps (see "arena addressing"
// below). Within a bucket the 1024 slots are 8 contiguous partition runs of
// 128, so both access patterns stay contiguous: a scatter writes one partition
// of many destination buckets, a stage reads all 8 partitions of one bucket.
// ---------------------------------------------------------------------------

#pragma once

#include "equi24_params.h"

// ---- reference word -------------------------------------------------------
//
//   [ bucket : EQ_REFBUCKBITS | cantor(s0,s1) : EQ_CANTORBITS ]
//
// The destination partition is NOT stored: it is implied by which partition
// stream the pair was written into, which is what frees the 3 bucket bits that
// make the word fit 32.
//
// The Cantor form here is a*(a-1)/2 + b, NOT djeZo's a*(a+1)/2 + b. At
// NSLOTS=1024 their form peaks at 524798 against our 19-bit ceiling of 524288
// and overflows into the bucket field -- reconstructing garbage proofs rather
// than failing loudly. Their form is correct for THEIR geometry (NSLOTS=1248
// with 20 Cantor bits, max 779374). Verified exhaustively: the form below is a
// bijection over all 523776 pairs with max code 523775.

#define EQ_CANTOR(a, b)  (((a) * ((a) - 1u)) / 2u + (b))

#ifdef __cplusplus
// EQ_CANTOR_MAX in equi24_params.h is NSLOTS*(NSLOTS-1)/2 = 523776, one past
// the largest code this form emits, so the assert there is correct and
// conservative for this formula. It would NOT be correct for djeZo's.
static_assert(((EQ_NSLOTS - 1u) * (EQ_NSLOTS - 2u)) / 2u + (EQ_NSLOTS - 2u) < (1u << EQ_CANTORBITS),
	"Cantor form overflows the reference word at this NSLOTS");
#endif

#if defined(__CUDACC__)
#define EQ_HD __host__ __device__ __forceinline__
#else
#define EQ_HD static inline
#endif

// Pack an unordered slot pair. Order-independent by construction.
EQ_HD eq_u32 eq24_cantor(eq_u32 s0, eq_u32 s1)
{
	const eq_u32 a = s0 > s1 ? s0 : s1;
	const eq_u32 b = s0 > s1 ? s1 : s0;
	return EQ_CANTOR(a, b);
}

EQ_HD eq_u32 eq24_ref_pack(eq_u32 bucket, eq_u32 s0, eq_u32 s1)
{
	// bucket arrives full-width; the low EQ_REFBUCKBITS are stored and the
	// top EQ_PARTBITS are recovered from the partition stream.
	return ((bucket & ((1u << EQ_REFBUCKBITS) - 1u)) << EQ_CANTORBITS)
	     | (eq24_cantor(s0, s1) & EQ_CANTORMASK);
}

// `part` is the partition the reference was READ from -- it supplies the
// bucket's top bits, which is the whole point of partitioning.
EQ_HD eq_u32 eq24_ref_bucket(eq_u32 ref, eq_u32 part)
{
	return ((part & (EQ_NPARTS - 1u)) << EQ_REFBUCKBITS)
	     | ((ref >> EQ_CANTORBITS) & ((1u << EQ_REFBUCKBITS) - 1u));
}

// Inverse Cantor: a = floor((1 + sqrt(1+8c))/2), b = c - a(a-1)/2.
// 8c+1 < 2^22 here, so a float sqrt is exact (24-bit mantissa); the +/-1
// correction costs nothing and removes any doubt about rounding mode.
EQ_HD void eq24_ref_slots(eq_u32 ref, eq_u32 *s0, eq_u32 *s1)
{
	const eq_u32 c = ref & EQ_CANTORMASK;
#if defined(__CUDA_ARCH__)
	eq_u32 a = (eq_u32)((1.0f + sqrtf((float)(8u * c + 1u))) * 0.5f);
#else
	eq_u32 a = (eq_u32)((1.0 + sqrt((double)(8u * c + 1u))) * 0.5);
#endif
	while (a > 1u && EQ_CANTOR(a, 0u) > c) a--;
	while (EQ_CANTOR(a + 1u, 0u) <= c) a++;
	*s1 = a;
	*s0 = c - EQ_CANTOR(a, 0u);
}

// ---- arena addressing: AoS records in two staggered heaps -----------------
//
// Layer r lives in heap[r & 1] at word offset (r >> 1) inside a 32-byte slot,
// as one contiguous record:   [ ref u32 | payload words ]
//
// Layers r and r+2 share a heap, with r+2 starting ONE WORD LATER. That word
// is layer r's first payload word -- dead by the time round r+2 runs, because
// round r+1 already consumed it -- while layer r's REFERENCE, at word r>>1,
// is never touched. So every reference survives to proof assembly without a
// separate retained array. (tromp's 4-byte stagger; the static_asserts below
// check it for both variants.)
//
// The 32-byte stride keeps a staggered record inside a single sector. The
// predecessor layout held references and payload in separate arrays and cost
// TWO random sectors per item instead of one, measured at 3.20x store
// amplification; a 20- or 24-byte stride would straddle and undo it.

#define EQ_HEAP_W         8u        // u32 per slot per heap = 32 B = one sector
#define EQ_LAYER_HEAP(r)  ((r) & 1u)
#define EQ_LAYER_OFF(r)   ((r) >> 1)
#define EQ_HEAP_SLOTS     ((size_t)EQ_NBUCKETS * EQ_NSLOTS)
#define EQ_HEAP_BYTES     (EQ_HEAP_SLOTS * EQ_HEAP_W * 4u)

// record base for layer r at absolute slot index `at` inside its own heap
#define EQ_REC(heap, at, r)  ((heap) + (size_t)(at) * EQ_HEAP_W + EQ_LAYER_OFF(r))

#ifdef __cplusplus
// every layer's record must fit the slot: offset + ref + payload <= EQ_HEAP_W
#define EQ_LAYER_END(r) (EQ_LAYER_OFF(r) + 1u + ((EQ_PAYLOAD(r) + 3u) / 4u))
static_assert(EQ_LAYER_END(0) <= EQ_HEAP_W && EQ_LAYER_END(EQ_WK - 1) <= EQ_HEAP_W,
	"a layer record overflows the 32-byte slot -- widen EQ_HEAP_W or re-check the payload schedule");
// and layer r+2 must start strictly past layer r's reference word
static_assert(EQ_LAYER_OFF(2) > EQ_LAYER_OFF(0),
	"the stagger does not advance: layer r+2 would clobber layer r's reference");
#endif

// absolute slot index within a bucket, from (partition, index in partition)
#define EQ_SLOT_INDEX(bucket, part, i) \
	((size_t)(bucket) * EQ_NSLOTS + (size_t)(part) * EQ_SLOTS_PER_PART + (i))

// total device bytes the solver allocates for item storage
#define EQ_ARENA_BYTES   (2u * EQ_HEAP_BYTES)
