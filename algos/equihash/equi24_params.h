// ---------------------------------------------------------------------------
// Geometry for the DIGITBITS=24 Equihash variants (144/5 and 192/7).
//
// Both derive DIGITBITS = WN/(WK+1) = 24, so they share the bucket geometry,
// the reference-word layout and the partition scheme, and differ only in the
// round count and the payload schedule. That is what lets one templated solver
// serve both.
//
// Every constant below is derived except RESTBITS and the partition count, and
// both of those are pinned by the static_asserts at the bottom.
// ---------------------------------------------------------------------------

#pragma once

#include <stdint.h>

#if !defined(EQ_WN) || !defined(EQ_WK)
#error "equi24_params.h needs -DEQ_WN and -DEQ_WK (144/5 or 192/7)"
#endif

// Symbol keying. Both variants build from one source as separate TUs, so the
// namespace and every exported entry point carry EQ_WN. Defined here rather
// than in the .cu so that anything linking against the solver derives the same
// names -- keying the solver without keying its callers leaves them declaring
// symbols that do not exist.
#define EQ24_CAT2(a, b) a##b
#define EQ24_CAT(a, b)  EQ24_CAT2(a, b)
#define EQ24_NS         EQ24_CAT(equi24_, EQ_WN)
#define EQ24_API(name)  EQ24_CAT(EQ24_CAT(equi24_, EQ_WN), EQ24_CAT(_, name))

typedef uint32_t eq_u32;
typedef uint64_t eq_u64;

// ---- derived geometry -----------------------------------------------------

#define EQ_NDIGITS        (EQ_WK + 1)
#define EQ_DIGITBITS      (EQ_WN / EQ_NDIGITS)

// RESTBITS = 8 is not a free parameter: it makes the bucket index exactly two
// bytes at DIGITBITS=24, and the reference-word assert below fails if it moves.
#define EQ_RESTBITS       8
#define EQ_NRESTS         (1u << EQ_RESTBITS)
#define EQ_RESTMASK       (EQ_NRESTS - 1u)

#define EQ_BUCKBITS       (EQ_DIGITBITS - EQ_RESTBITS)
#define EQ_NBUCKETS       (1u << EQ_BUCKBITS)
#define EQ_BUCKMASK       (EQ_NBUCKETS - 1u)

// Partitions exist to buy reference-word bits, not to give threads private
// cursors: 65536 source buckets over 8 partitions means 8192 blocks share each
// one, so atomic traffic is handled by the warp-aggregated append instead.
// P=8 is the smallest count whose per-stream overflow margin is real (8.0
// sigma, against 4.6 at P=4, which overflows ~3 streams per solve).
#define EQ_PARTBITS       3
#define EQ_NPARTS         (1u << EQ_PARTBITS)

#define EQ_NSLOTS         1024u                       // capacity per bucket
#define EQ_SLOTS_PER_PART (EQ_NSLOTS / EQ_NPARTS)     // 128; mean is 64

#define EQ_PROOFSIZE      (1u << EQ_WK)
#define EQ_BASE           (1u << EQ_DIGITBITS)
#define EQ_NHASHES        (2u * EQ_BASE)              // Wagner's invariant
#define EQ_MEAN_PER_BUCKET (EQ_NHASHES / EQ_NBUCKETS) // 512

#define EQ_HASHESPERBLAKE (512 / EQ_WN)
#define EQ_HASHOUT        (EQ_HASHESPERBLAKE * EQ_WN / 8)
#define EQ_NBLAKES        ((EQ_NHASHES + EQ_HASHESPERBLAKE - 1) / EQ_HASHESPERBLAKE)

// ---- reference word -------------------------------------------------------
// A u32 holding [ bucket(13) | cantor(19) ]. The destination partition is
// implied by WHICH partition stream the pair was written into, which is where
// the missing 3 bucket bits come from.
#define EQ_REFBUCKBITS    (EQ_BUCKBITS - EQ_PARTBITS)     // 13
#define EQ_CANTORBITS     (32 - EQ_REFBUCKBITS)           // 19
#define EQ_CANTORMASK     ((1u << EQ_CANTORBITS) - 1u)
#define EQ_CANTOR_MAX     ((EQ_NSLOTS * (EQ_NSLOTS - 1u)) / 2u)

// ---- payload schedule -----------------------------------------------------
// Bits still to be cancelled after round r, minus the BUCKBITS that the
// destination bucket index already encodes (free, because position carries
// them). Round 0 is digitH's output.
#define EQ_HASHBITS(r)    (EQ_WN - (r) * EQ_DIGITBITS - EQ_BUCKBITS)
#define EQ_PAYLOAD(r)     (((EQ_HASHBITS(r)) + 7) / 8)
#define EQ_PAYLOAD_U32(r) ((EQ_PAYLOAD(r) + 3) / 4)

#define EQ_MAXPAYLOAD     EQ_PAYLOAD(0)

// Slot stride in u32 words, rounded up to 4 so every slot is 16-byte aligned
// and a payload moves in ONE vector access. 144/5: 16 B -> 4 words (already
// exact). 192/7: 22 B -> 6 -> 8 words (32 B), costing 45% more payload arena to
// turn a 22-byte byte-loop into two aligned stores. Without it the payload
// copies a byte at a time, which measured ~2x slower than the tromp path.
#define EQ_ROUNDUP4(x)    (((x) + 3u) & ~3u)
#define EQ_SLOT_W         EQ_ROUNDUP4(((EQ_MAXPAYLOAD) + 3u) / 4u)
#define EQ_SLOT_BYTES     (EQ_SLOT_W * 4u)

// ---- compile-time checks --------------------------------------------------
// A packed-index bit budget that is only correct by argument will be wrong the
// first time someone edits a constant. These make it a build error.

#ifdef __cplusplus
static_assert(EQ_DIGITBITS == 24,
	"equi24 is the DIGITBITS=24 family only (144/5, 192/7)");

static_assert(EQ_REFBUCKBITS + EQ_CANTORBITS == 32,
	"reference word must be exactly 32 bits");

// The tight one: at NSLOTS=1024 the Cantor value peaks at 523776 against a
// 19-bit ceiling of 524288 -- 0.1% of headroom. Raising NSLOTS by one step
// breaks the reference word, which is why this is an assert and not a comment.
static_assert(EQ_CANTOR_MAX < (1u << EQ_CANTORBITS),
	"NSLOTS too large: the Cantor value no longer fits the reference word");

static_assert(EQ_NSLOTS % EQ_NPARTS == 0,
	"NSLOTS must divide evenly into partitions");

// Capacity must clear its mean by a MARGIN, not merely exceed it. A stream
// holds mean = NHASHES/(NBUCKETS*NPARTS) items with sigma = sqrt(mean), and the
// design calls for >= 6 sigma (P=8 gives 8.0). Written squared so it stays
// integer: (cap - mean)^2 >= 36 * mean. An ordering-only form ("capacity >
// mean") admits NSLOTS=600, a 1.4 sigma margin that overflows on nearly every
// bucket.
#define EQ_MEAN_PER_PART  (EQ_NHASHES / (EQ_NBUCKETS * EQ_NPARTS))
static_assert(EQ_SLOTS_PER_PART > EQ_MEAN_PER_PART &&
	(EQ_SLOTS_PER_PART - EQ_MEAN_PER_PART) * (EQ_SLOTS_PER_PART - EQ_MEAN_PER_PART)
		>= 36u * EQ_MEAN_PER_PART,
	"per-partition capacity is under 6 sigma of its mean occupancy -- buckets will overflow");

// Wagner keeps the table size invariant, so every round must fit the same arena.
static_assert(EQ_NBUCKETS * EQ_NSLOTS >= EQ_NHASHES,
	"arena capacity below the per-round item count");

static_assert(EQ_PAYLOAD(EQ_WK - 1) >= 4,
	"last round's payload underflowed -- check the digit schedule");
#endif
