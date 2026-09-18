// ---------------------------------------------------------------------------
// Geometry for the DIGITBITS=24 solver (144/5 and 192/7): FEW LARGE BUCKETS.
//
// The scatter is the cost, and it is bound by the write frontier -- one open
// 32 B sector per destination bucket. With 4096 buckets that is 128 KB and
// stays resident in L2; a many-bucket layout puts it in the megabytes and
// thrashes. Everything else follows from choosing 4096:
//
//   * a bucket is NEVER fully staged in shared memory; it is processed in
//     EQ24B_CHUNKS passes split by REST PREFIX, which is lossless because two
//     items whose rest values differ in the high bits can never collide
//   * collisions are found with a per-rest LINKED LIST in shared memory
//     (atomicExch on a head array), not a counting sort, so no order array
//   * layers are INDEPENDENT arrays, not a staggered heap, so nothing is
//     overwritten and the reference may be 8 bytes
//
// Only 8/16/32-byte strides are used: a scattered record costs one sector
// transaction regardless of its size, and only those three divide a 32 B
// sector without straddling.
// ---------------------------------------------------------------------------
#ifndef EQUI24B_PARAMS_H
#define EQUI24B_PARAMS_H

#if !defined(EQ_WN) || !defined(EQ_WK)
#error "equi24b_params.h needs -DEQ_WN and -DEQ_WK (144/5 or 192/7)"
#endif

#define EQ24B_NDIGITS      (EQ_WK + 1)
#define EQ24B_DIGITBITS    (EQ_WN / EQ24B_NDIGITS)

// The whole point of this architecture: few, large buckets.
#define EQ24B_BUCKBITS     12u
#define EQ24B_RESTBITS     (EQ24B_DIGITBITS - EQ24B_BUCKBITS)
#define EQ24B_NBUCKETS     (1u << EQ24B_BUCKBITS)
#define EQ24B_NRESTS       (1u << EQ24B_RESTBITS)
#define EQ24B_BUCKMASK     (EQ24B_NBUCKETS - 1u)
#define EQ24B_RESTMASK     (EQ24B_NRESTS - 1u)

// Capacity per bucket. Mean is 2^25 / 4096 = 8192 with sigma 90.5, so 8704 is
// mean + 5.7 sigma, i.e. 1.06x over-provisioned. Large buckets need
// proportionally less slack because relative spread falls as
// 1/sqrt(n); that is half of why this arena is smaller despite bigger records.
#define EQ24B_NSLOTS       8704u

// Chunks per bucket, split by rest prefix. 3 fits 39.0 KB of shared and 4 fits
// 31.5 KB; each chunk costs one extra streaming pass over the bucket, measured
// at 1.70x (3 passes) and 1.53x (4) against the current single-pass round.
#ifndef EQ24B_CHUNKS
#define EQ24B_CHUNKS       3u
#endif

// ---- reference word: 8 bytes, because no layer overwrites another ----------
#define EQ24B_CANTOR_MAX   (((eq_u64)EQ24B_NSLOTS * (EQ24B_NSLOTS - 1u)) / 2u)
#define EQ24B_CANTORBITS   26u
#define EQ24B_CANTORMASK   ((((eq_u64)1) << EQ24B_CANTORBITS) - 1u)
#define EQ24B_REFBITS      (EQ24B_BUCKBITS + EQ24B_CANTORBITS)

// ---- payload schedule ------------------------------------------------------
// Bits still to cancel after round r, less the bucket index that position
// already encodes.
#define EQ24B_HASHBITS(r)  (EQ_WN - (r) * EQ24B_DIGITBITS - EQ24B_BUCKBITS)
#define EQ24B_PAYLOAD(r)   ((EQ24B_HASHBITS(r) + 7) / 8)
#define EQ24B_RECBYTES(r)  (8u + EQ24B_PAYLOAD(r))

// Only 8/16/32 avoid straddling a 32 B sector; anything between costs MORE
// transactions than the 32 B record it replaces (measured: 20 B and 24 B both
// touch 1.50 sectors per record against 1.00 for 32 B).
#define EQ24B_STRIDE(r) \
	(EQ24B_RECBYTES(r) <= 8u ? 8u : (EQ24B_RECBYTES(r) <= 16u ? 16u : 32u))

#define EQ24B_LAYER_BYTES(r) \
	((eq_u64)EQ24B_NBUCKETS * EQ24B_NSLOTS * EQ24B_STRIDE(r))

// Bytes staged per slot while a chunk is resident: 8 B of payload (all the
// collision test and the XOR need) plus a 2-byte list link.
#define EQ24B_STAGE_BYTES  10u

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
// The constraint that killed three earlier designs, now checked rather than argued.
static_assert(EQ24B_REFBITS > 32,
	"reference exceeds 32 bits by design -- layers must NOT be staggered");
#endif


// ---- digitR staging -------------------------------------------------------
// A bucket is never fully staged. Chunk c handles the rests whose PREFIX is c,
// which is lossless: two items whose rest values differ in the high bits cannot
// collide, so no pair spans a chunk.
//
// Per staged item: the layer r-1 payload, a 2-byte list link, and the 2-byte
// ORIGIN SLOT -- the reference must name slots within the source bucket, and the
// staging index is an atomic append order that says nothing about them.
//
// Heads cover only the chunk's rest range, not all 4096; sizing them per chunk
// cut the total passes over a bucket from 18 to 15. The division must round
// UP: with chunks*heads < EQ24B_NRESTS the top rests select a chunk index
// that is never visited and their items are staged by no pass at all.
#define EQ24B_STAGE_W(r)     (((EQ24B_PAYLOAD((r) - 1u) + 3u) / 4u))
#define EQ24B_STAGE_B(r)     (EQ24B_STAGE_W(r) * 4u + 4u)

// Heads are u32: CUDA has no 16-bit atomicExch, and the list head must be
// swapped atomically. (The reference miner's head array is 4-byte too.)
//
// Chunks per round, derived from the 48 KB budget at mean + 5 sigma occupancy.
// 144/5: 6,5,4,3   192/7: 6,6,6,5,4,3
#if EQ_WN == 144
#define EQ24B_CHUNKS_R(r)    ((r) == 1u ? 6u : ((r) == 2u ? 5u : ((r) == 3u ? 4u : 3u)))
#else
#define EQ24B_CHUNKS_R(r)    ((r) <= 3u ? 6u : ((r) == 4u ? 5u : ((r) == 5u ? 4u : 3u)))
#endif

#define EQ24B_HEADS(r)       ((EQ24B_NRESTS + EQ24B_CHUNKS_R(r) - 1u) / EQ24B_CHUNKS_R(r))
// Budget 48 KB MINUS 512 B: the block also carries a static counter, and a
// dynamic request that lands on 49152 exactly makes the launch fail with
// cudaErrorInvalidValue -- silently, if nobody checks (it did).
#define EQ24B_SHARED_BUDGET  (49152u - 512u)
#define EQ24B_STAGE_SLOTS(r) ((EQ24B_SHARED_BUDGET - EQ24B_HEADS(r) * 4u) / EQ24B_STAGE_B(r))
#define EQ24B_LIST_END       0xffffu
#define EQ24B_PROOFSIZE      (1u << EQ_WK)

#endif // EQUI24B_PARAMS_H
