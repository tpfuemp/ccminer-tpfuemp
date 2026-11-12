// ---------------------------------------------------------------------------
// Geometry for the DIGITBITS=24 solver (144/5 and 192/7): few large buckets.
//
// The scatter is bound by its write frontier, one open 32 B sector per
// destination bucket; with 4096 buckets that frontier stays resident in L2.
//
//   * a bucket is never fully staged in shared memory; each round processes
//     it in chunks split by REST PREFIX, which is lossless because items whose
//     rest differs in the high bits cannot collide
//   * collisions are found with a per-rest linked list in shared memory
//     (atomicExch on a head array)
//   * layers are independent arrays, so nothing is overwritten and the
//     reference may be 8 bytes
//
// Record strides are 8, 16 or 32 B only: a scattered record costs one sector
// transaction, and any other size straddles sectors.
// ---------------------------------------------------------------------------
#ifndef EQUI24B_PARAMS_H
#define EQUI24B_PARAMS_H

#if !defined(EQ_WN) || !defined(EQ_WK)
#error "equi24b_params.h needs -DEQ_WN and -DEQ_WK (144/5 or 192/7)"
#endif

#define EQ24B_NDIGITS      (EQ_WK + 1)
#define EQ24B_DIGITBITS    (EQ_WN / EQ24B_NDIGITS)

#define EQ24B_BUCKBITS     12u
#define EQ24B_RESTBITS     (EQ24B_DIGITBITS - EQ24B_BUCKBITS)
#define EQ24B_NBUCKETS     (1u << EQ24B_BUCKBITS)
#define EQ24B_NRESTS       (1u << EQ24B_RESTBITS)
#define EQ24B_BUCKMASK     (EQ24B_NBUCKETS - 1u)
#define EQ24B_RESTMASK     (EQ24B_NRESTS - 1u)

// Slots per bucket: mean 2^25 / 4096 = 8192, sigma 90.5, so 8704 is mean + 5.7 sigma.
#define EQ24B_NSLOTS       8704u

// ---- reference word: 8 bytes (source bucket + Cantor code of two slots) ------
#define EQ24B_CANTOR_MAX   (((eq_u64)EQ24B_NSLOTS * (EQ24B_NSLOTS - 1u)) / 2u)
#define EQ24B_CANTORBITS   26u
#define EQ24B_CANTORMASK   ((((eq_u64)1) << EQ24B_CANTORBITS) - 1u)
#define EQ24B_REFBITS      (EQ24B_BUCKBITS + EQ24B_CANTORBITS)

// ---- payload schedule and record layout ------------------------------------
// Bits still to cancel after round r, less the bucket index the position encodes.
#define EQ24B_HASHBITS(r)  (EQ_WN - (r) * EQ24B_DIGITBITS - EQ24B_BUCKBITS)
#define EQ24B_PAYLOAD(r)   ((EQ24B_HASHBITS(r) + 7) / 8)
// Record: [8 B ref][payload]. A layer whose payload fits 16 B only without the
// ref's 26 unused high bits is PACKED instead: [ref bits 0-31][payload][ref bits
// 32-37 in byte 15], so it takes one 16 B sector slot instead of a 32 B one.
// Layer 0 holds the blake index (4 B) and keeps the plain layout.
#define EQ24B_PACKED(r)    ((r) > 0u && 8u + EQ24B_PAYLOAD(r) > 16u && 4u + EQ24B_PAYLOAD(r) <= 15u)
#define EQ24B_POFF_W(r)    (EQ24B_PACKED(r) ? 1u : 2u)          // payload word offset
#define EQ24B_RECBYTES(r)  (EQ24B_PACKED(r) ? 16u : 8u + EQ24B_PAYLOAD(r))
// Payload bytes valid in its last word (readers mask the rest: a packed layer
// keeps ref bits there, and the zero-XOR test must not see them).
#define EQ24B_PMASK(r)     ((EQ24B_PAYLOAD(r) & 3u) ? ((1u << (8u * (EQ24B_PAYLOAD(r) & 3u))) - 1u) : 0xffffffffu)
#define EQ24B_STRIDE(r) \
	(EQ24B_RECBYTES(r) <= 8u ? 8u : (EQ24B_RECBYTES(r) <= 16u ? 16u : 32u))
#define EQ24B_LAYER_BYTES(r) \
	((eq_u64)EQ24B_NBUCKETS * EQ24B_NSLOTS * EQ24B_STRIDE(r))

#ifdef __cplusplus
#include <cstdint>
typedef std::uint64_t eq_u64;

static_assert(EQ24B_DIGITBITS == 24,
	"equi24b is the DIGITBITS=24 family only (144/5, 192/7)");
static_assert(EQ24B_BUCKBITS + EQ24B_RESTBITS == EQ24B_DIGITBITS,
	"bucket and rest bits must partition a digit");
static_assert(EQ24B_CANTOR_MAX < (((eq_u64)1) << EQ24B_CANTORBITS),
	"Cantor code overflows its field at this NSLOTS");
static_assert(EQ24B_REFBITS <= 64,
	"reference word must fit 64 bits");
static_assert(EQ24B_REFBITS > 32,
	"reference exceeds 32 bits by design -- layers must NOT be staggered");
static_assert(EQ24B_REFBITS <= 40,
	"a packed record keeps ref bits 32+ in one byte");
#endif

// ---- digitR / digitK staging ------------------------------------------------
// One block per (bucket, chunk). Per staged item: the round's input payload and
// a 2-byte list link. The ORIGIN SLOT the reference names (the staging index is
// an append order) replaces payload bytes 0-1, the rest, which is equal within
// a list and never emitted. Heads cover only the chunk's rest range and are u32
// because there is no 16-bit atomicExch. The heads division rounds UP, or the
// top rests would select a chunk never visited.
#define EQ24B_STAGE_W(r)     (((EQ24B_PAYLOAD((r) - 1u) + 3u) / 4u))

// Chunks per round (digitK uses CHUNKS_R(EQ_WK - 1)).
#if EQ_WN == 144
#define EQ24B_CHUNKS_R(r)    ((r) == 1u ? 5u : ((r) <= 3u ? 4u : 3u))
#else
#define EQ24B_CHUNKS_R(r)    ((r) == 1u ? 6u : ((r) <= 3u ? 5u : ((r) <= 5u ? 4u : 3u)))
#endif
#define EQ24B_HEADS(r)       ((EQ24B_NRESTS + EQ24B_CHUNKS_R(r) - 1u) / EQ24B_CHUNKS_R(r))

// 48 KB minus 512 B for the static counters: a dynamic request of exactly
// 49152 B fails the launch with cudaErrorInvalidValue.
#define EQ24B_SHARED_BUDGET  (49152u - 512u)
#define EQ24B_LIST_END       0xffffu
#define EQ24B_PROOFSIZE      (1u << EQ_WK)
#define EQ24B_STAGE_SLOTS(r) ((EQ24B_SHARED_BUDGET - EQ24B_HEADS(r) * 4u - 4u) / (EQ24B_STAGE_W(r) * 4u + 2u))
// Word offset of the head array: payloads, then the u16 links padded to a word.
#define EQ24B_HEAD_OFF_W(r)  (EQ24B_STAGE_SLOTS(r) * EQ24B_STAGE_W(r) + (EQ24B_STAGE_SLOTS(r) + 1u) / 2u)
#define EQ24B_SHARED_BYTES(r) ((EQ24B_HEAD_OFF_W(r) + EQ24B_HEADS(r)) * 4u)

#ifdef __cplusplus
// Per round: staging fits the budget, one block holds a full bucket's share of
// its chunk + 6 sigma (binomial), and its indices fit the u16 links
// (EQ24B_LIST_END is reserved).
#define EQ24B_SHARE(r)       (EQ24B_NSLOTS * EQ24B_HEADS(r) / EQ24B_NRESTS)
#define EQ24B_ROUND_OK(r) \
	(EQ24B_SHARED_BYTES(r) <= EQ24B_SHARED_BUDGET \
	 && EQ24B_STAGE_SLOTS(r) > EQ24B_SHARE(r) \
	 && (EQ24B_STAGE_SLOTS(r) - EQ24B_SHARE(r)) * (EQ24B_STAGE_SLOTS(r) - EQ24B_SHARE(r)) >= 36u * EQ24B_SHARE(r) \
	 && EQ24B_STAGE_SLOTS(r) < EQ24B_LIST_END)
static_assert(EQ24B_ROUND_OK(1u) && EQ24B_ROUND_OK(2u) && EQ24B_ROUND_OK(3u) && EQ24B_ROUND_OK(4u),
	"digitR/digitK staging out of budget or capacity");
static_assert(EQ24B_NSLOTS <= 0x10000u, "the staged origin slot is 16 bits");
#if EQ_WK > 5
static_assert(EQ24B_ROUND_OK(5u) && EQ24B_ROUND_OK(6u),
	"digitR/digitK staging out of budget or capacity");
#endif
#endif

#endif // EQUI24B_PARAMS_H
