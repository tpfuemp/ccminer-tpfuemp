/* yespower 1.0 family -- host driver and mining kernel.
 *
 * The hash body lives in yespower_hash.cuh and is shared verbatim with an
 * out-of-tree KAT that checks it against sph/yespower_ref.c for every variant
 * shape.  The reference stays compiled in as the host-side re-verify for every
 * GPU candidate, so a kernel bug can only cost a local reject, never a bad share.
 *
 * ARCH GATE: S is 96 KiB of shared memory per instance, so this needs a device
 * that can opt in to >= 98304 B per block -- sm_80/sm_86 only.  Turing caps at
 * 64 KiB and Pascal at 48 KiB, and the global-memory alternative is slower per
 * access.  Checked at init.
 *
 * Version selection: yespower 1.0 only.  yespower 0.5 is bit-identical to the
 * yescrypt 0.5 this miner already ships in algos/yescrypt/, so a 0.5 request
 * must be routed there instead of duplicating a consensus path.
 */

#include "miner.h"
#include "cuda_helper.h"
#include "algos.h"     /* ALGO_* and `opt_algo`; miner.h declares neither */

extern "C" {
#include "sph/yespower.h"
}

#include <string.h>
#include <stdlib.h>

#include "yespower_hash.cuh"
#include "cuda/selftest_gate.cuh"

extern char *yescrypt_key;          /* shared with --yescrypt-key / --yespower-key */
extern size_t yescrypt_key_len;
extern uint32_t yescrypt_param_N;   /* shared with --yescrypt-param / --yespower-param */
extern uint32_t yescrypt_param_r;

/* ---------------------------------------------------------------------------
 * Verified parameter table.
 *
 * `pers` is hashed, so one wrong byte gives a valid-looking hash that a pool
 * rejects 100% of the time while the miner looks healthy.  Every string below is
 * cross-checked against two independent sources and its length is written out
 * beside it as a checksum.  Do not retype one from rendered text: a quote that
 * passes through a filter can lose short words silently, and only the length
 * catches it.  See algos/yespower/README.md for the sources and pool status.
 * ------------------------------------------------------------------------- */
static const char PERS_POWER2B[] = "Now I am become Death, the destroyer of worlds";
static const char PERS_SUGAR[]   =
	"Satoshi Nakamoto 31/Oct/2008 Proof-of-work is essentially one-CPU-one-vote";
static const char PERS_URX[]     = "UraniumX";
static const char PERS_LTNCG[]   = "LTNCGYES";
static const char PERS_MGPC[]    = "Magpies are birds of the Corvidae family.";
static const char PERS_ARWN[]    = "ARWN";
static const char PERS_ADVC[]    = "Let the quest begin";
/* EqPay: 88 bytes, and the trailing '.' is part of it -- perslen is exactly
 * strlen(), with no NUL hashed. */
static const char PERS_EQPAY[]   =
	"The gods had gone away, and the ritual of the religion continued "
	"senselessly, uselessly.";
static const char PERS_IC[]      = "IsotopeC";
static const char PERS_IOTS[]    = "Iots is committed to the development of IOT";
static const char PERS_LITB[]    =
	"LITBpower: The number of LITB working or available for proof-of-work mini";
static const char PERS_CPUPOWER[] =
	"CPUpower: The number of CPU working or available for proof-of-work mining";

/* Coin variants: `-a yespower` with a fixed (N, r, pers) triple, so each is an
 * alias rather than an algo. `algo_to_int()` (algos.h) maps the name; this table
 * supplies the parameters. The two lists must agree, and they are checked against
 * each other: yespower_set_variant() rejects a name algos.h accepted but that has
 * no entry here, because the silent failure is worse than no alias -- generic
 * r=32/keyless parameters mine perfectly healthy-looking hashes that the pool
 * rejects 100% of the time.
 *
 * `perslen` is written out rather than taken from strlen() so that it acts as a
 * checksum on the literal beside it; the setter verifies the two agree.
 *
 * yespowerEQPAY is deliberately not here: it hashes a 181-byte header, which no
 * (N, r, pers) triple can express, so it is its own algo (ALGO_YESPOWEREQPAY)
 * and --yespower-param/-key cannot reach it either. */
struct yespower_variant {
	const char *name;
	uint32_t    N;
	uint32_t    r;
	const char *pers;      /* NULL = keyless */
	size_t      perslen;   /* 0 iff pers == NULL */
};

static const struct yespower_variant yespower_variants[] = {
	/* name             N     r   pers            perslen */
	{ "yespowersugar",  2048, 32, PERS_SUGAR,     74 },  /* Sugarchain (SUGAR) */
	{ "sugarchain",     2048, 32, PERS_SUGAR,     74 },
	{ "yespowerurx",    2048, 32, PERS_URX,       8  },  /* UraniumX (URX)     */
	{ "yespowerltncg",  2048, 32, PERS_LTNCG,     8  },  /* LightningCash-Gold */
	{ "yespowermgpc",   2048, 32, PERS_MGPC,      41 },  /* MagpieCoin (MGPC)  */
	{ "yespowertide",   2048,  8, NULL,           0  },  /* Tidecoin (TDC)     */
	{ "yespowerarwn",   2048, 32, PERS_ARWN,      4  },  /* Arowana (ARWN)     */
	{ "yespoweradvc",   2048, 32, PERS_ADVC,      19 },  /* AdventureCoin ADVC */
	{ "yespoweric",     2048, 32, PERS_IC,        8  },  /* IsotopeC           */
	{ "yespoweriots",   2048, 32, PERS_IOTS,      43 },  /* IOTS               */
	{ "yespowerlitb",   2048, 32, PERS_LITB,      73 },  /* LightBit (LITB)    */
	{ "cpupower",       2048, 32, PERS_CPUPOWER,  73 },  /* CPUchain (CPU)     */

	/* power2b is absent deliberately: it is its own algo (ALGO_POWER2B, blake2b
	 * head/tail) and takes its parameters from yespower_params_for() below. */
};

/* The names that carry no preset -- they mine whatever --yespower-param/-key say,
 * which is the behaviour they have always had. Listed so that an alias present in
 * algos.h but missing from the table above is still an error rather than silence. */
static bool yespower_name_is_generic(const char *name)
{
	return !strcasecmp(name, "yespower")     || !strcasecmp(name, "yespowerr16") ||
	       !strcasecmp(name, "yenten")       || !strcasecmp(name, "power2b")     ||
	       !strcasecmp(name, "yespower-b2b");
}

/* The variant `-a` (or a pool's "algo") selected, NULL for the generic names.
 *
 * Deliberately NOT written into yescrypt_param_N/_r/yescrypt_key: those are the
 * CLI's own state and are shared with the yescrypt 0.5 algos, so presetting them
 * would leak a coin's key across a pool switch into a different algo's job. The
 * selection is consulted in yespower_params_for() instead, and an explicit
 * --yespower-param/-key clears it (see yespower_clear_variant). */
static const struct yespower_variant *yespower_selected = NULL;

extern "C" void yespower_clear_variant(void)
{
	yespower_selected = NULL;
}

/* Select the variant named by an `-a` alias or a pool's "algo". False = the name
 * has no entry, which is a bug in the alias list, not user error: `-a` treats it
 * as fatal, and a pool switch refuses the switch. */
extern "C" bool yespower_set_variant(const char *name)
{
	if (!name || !*name || yespower_name_is_generic(name)) {
		yespower_selected = NULL;
		return true;
	}
	for (size_t i = 0; i < ARRAY_SIZE(yespower_variants); i++) {
		const struct yespower_variant *v = &yespower_variants[i];
		if (strcasecmp(name, v->name))
			continue;
		if (v->pers ? (strlen(v->pers) != v->perslen) : (v->perslen != 0)) {
			applog(LOG_ERR, "yespower: variant '%s' pers is %u bytes, table says %u"
			       " -- the literal has been damaged, refusing to mine",
			       v->name, (uint32_t) (v->pers ? strlen(v->pers) : 0),
			       (uint32_t) v->perslen);
			return false;
		}
		yespower_selected = v;
		return true;
	}
	applog(LOG_ERR, "yespower: '%s' is an accepted algo name with no parameter entry"
	       " -- it would mine generic yespower and every share would be rejected",
	       name);
	return false;
}

/* "" when no variant is selected. Lets a caller that switches algo put the old
 * selection back if the switch it was part of fails. */
extern "C" const char *yespower_variant_name(void)
{
	return yespower_selected ? yespower_selected->name : "";
}

/* Fill `p` for the given algo. Returns false if the algo is not a yespower one. */
static bool yespower_params_for(int algo, yespower_params_t *p)
{
	memset(p, 0, sizeof(*p));
	p->version = YESPOWER_1_0;

	switch (algo) {
	case ALGO_YESPOWER:
		/* A coin alias (`-a yespowerSUGAR`) if one was named, else whatever
		 * --yespower-param/-key say, else the 1.0 defaults. */
		if (yespower_selected) {
			p->N = yespower_selected->N;
			p->r = yespower_selected->r;
			p->pers = (const uint8_t *) yespower_selected->pers;
			p->perslen = yespower_selected->perslen;
			return true;
		}
		p->N = yescrypt_param_N ? yescrypt_param_N : 2048;
		p->r = yescrypt_param_r ? yescrypt_param_r : 32;
		p->pers = (const uint8_t *) yescrypt_key;
		p->perslen = yescrypt_key ? yescrypt_key_len : 0;
		return true;
	case ALGO_YESPOWERR16:       /* Yenten (YTN) */
		p->N = 4096; p->r = 16;
		p->pers = NULL; p->perslen = 0;
		return true;
	case ALGO_POWER2B:           /* MicroBitcoin -- b2b wrappers not implemented yet */
		p->N = 2048; p->r = 32;
		p->pers = (const uint8_t *) PERS_POWER2B;
		p->perslen = sizeof(PERS_POWER2B) - 1;   /* 46 */
		return true;
	case ALGO_YESPOWEREQPAY:     /* EqPay (EQPAY) -- ordinary 1.0 params, 181-byte header */
		p->N = 2048; p->r = 32;
		p->pers = (const uint8_t *) PERS_EQPAY;
		p->perslen = sizeof(PERS_EQPAY) - 1;     /* 88 */
		return true;
	default:
		return false;
	}
}

/* EqPay (Qtum-derived) hashes the 181-byte extended header, not 80: the usual
 * fields, then hashStateRoot, hashUTXORoot, a null prevoutStake and an empty
 * signature length.  Full layout in README.md.
 *
 * prevoutStake.n MUST be COutPoint::NULL_INDEX; zero gives a well-formed digest
 * that every local check accepts and the pool rejects. */
#define YP_EQPAY_HDR_BYTES     181u
#define YP_EQPAY_HDR_WORDS     46u       /* ceil(181/4), the work-buffer span */
#define YP_EQPAY_STATE_ROOT     80u
#define YP_EQPAY_UTXO_ROOT     112u
#define YP_EQPAY_STAKE         144u
#define YP_EQPAY_STAKE_N       176u
#define YP_EQPAY_SIGLEN        180u

/* Bytes of block header this algo's proof-of-work covers. */
static inline size_t yp_header_bytes(int algo)
{
	return (algo == ALGO_YESPOWEREQPAY) ? YP_EQPAY_HDR_BYTES : 80u;
}

/* One header in, 32-byte digest out.
 *
 * NOTE the return convention: the reference returns 1 on success and -1 on
 * error, the opposite of what its own upstream header documents. On failure we
 * fail CLOSED -- an all-ones digest can never be <= any target, so a broken
 * hash can never be mistaken for a share. */
static bool yespower_hash_hdr(void *state, const void *input, size_t len,
                              const yespower_params_t *params)
{
	yespower_binary_t out;
	/*  The re-verify MUST follow the same primitive as the kernel. It called
	 * yespower_tls_ref() (SHA-256) for every algo including ALGO_POWER2B, so the
	 * GPU and the CPU agreed with each other and both disagreed with the network:
	 * 9/9 pool submissions were rejected while 0 of them tripped "does not
	 * validate". A re-verify that cannot disagree with the kernel is not a gate. */
	const int rc = (opt_algo == ALGO_POWER2B)
	             ? yespower_b2b_tls_ref((const uint8_t *) input, len, params, &out)
	             : yespower_tls_ref((const uint8_t *) input, len, params, &out);
	if (rc != 1) {
		memset(state, 0xff, 32);
		return false;
	}
	memcpy(state, out.uc, 32);
	return true;
}

/* Exposed for the GPU path's host-side re-verification. */
extern "C" void yespower_hash(void *state, const void *input, int algo)
{
	yespower_params_t p;
	if (!yespower_params_for(algo, &p)) { memset(state, 0xff, 32); return; }
	(void) yespower_hash_hdr(state, input, yp_header_bytes(algo), &p);
}

/* ---------------------------------------------------------------------------
 * GPU path.
 * ------------------------------------------------------------------------- */

/* X goes in shared on every arch that takes the global-S path.  It was briefly
 * gated to sm_80+ out of caution, which was wrong: Pascal gains more from it,
 * not less.  Only the instance count differs per arch. */
#define YP_X_SHARED(PL)  ((PL) == PWX_S_GLOBAL)

/* minBlocks is a register cap (ptxas divides the register file by
 * minBlocks x maxThreads) and `__launch_bounds__` supersedes -maxrregcount.
 * Left uncapped: it cannot be made arch-conditional, because with more than
 * one -gencode the bounds are fixed by the arch-agnostic front-end pass, and
 * capping every arch costs spill on the ones that do not need it. */
#ifndef YP_MINBLK
#define YP_MINBLK 1
#endif

/* Block WIDTH.  Four lanes do pwxform -- forced by the spec (`PWXgather = 4`)
 * -- and the spare threads only widen the `V` transfer, which is what this algo
 * is starved on: it is latency-bound, not bandwidth-bound.  It is also the
 * ceiling, enforced by the static_assert in yespower_hash.cuh: every barrier
 * here is a `__syncwarp`, which orders one warp and nothing more.  Does not
 * compose with -DYP_VBATCH, which recovers the same starvation. */
#ifndef YP_WIDTH
#define YP_WIDTH 32
#endif

/* Full participation mask for the block.  Written this way because `1u << 32`
 * is undefined behaviour, which is the shape a future WIDTH bump would hit. */
#define YP_WMASK ((YP_WIDTH >= 32) ? 0xffffffffu : (uint32_t)((1u << (YP_WIDTH & 31)) - 1u))

/* Candidate slots: res[0] is the count, res[1..YP_MAX_RES] the nonces.  Only
 * one block per nonce can report, so a real pool target never approaches the
 * bound; it is reachable only under a deliberately loosened benchmark target. */
#define YP_MAX_RES 8u

/* Instances per SM, and where X lives.
 *
 * Ampere shares one L1/shared pool per SM, so shared taken for X comes out of
 * the L1 the global S-box and V depend on; it runs fewer, larger instances.
 * Pascal has separate shared and L1 and runs twice as many.  Fewer instances
 * also halve VRAM and launch duration, which sets the stale-share rate.
 *
 * sm_75 takes the Volta+ value untested: its shared budget would put the higher
 * count at or over the per-SM cap.  Clamped by VRAM at init. */
#define YP_IPB_AMPERE 8u
#define YP_IPB_OTHER  16u

/* Header words in SHA-256 input order; [19] is the nonce and is supplied to the
 * hash separately.  46 words rather than 20 because EqPay's extended header is
 * 181 bytes; the other variants upload only the first 20 and never read past. */
#define YP_HDR_WORDS      20u                   /* yespower 1.0, 80-byte header */
#define YP_HDR_WORDS_181  46u                   /* EqPay, 181-byte header       */
__constant__ static uint32_t c_yp_hdr[YP_HDR_WORDS_181];
__constant__ static uint32_t c_yp_target[8];

/* Exact 256-bit compare, MSW first -- the ordering fulltest() uses. Exact rather
 * than a `hash[7] <= target[7]` screen on purpose: such a screen silently drops
 * shares below diff 1, which has cost other algos in this tree real shares.
 * Candidates are rare enough here that the full compare is free. */
__device__ __forceinline__ bool yp_below_target(const uint32_t h[8])
{
#pragma unroll
	for (int i = 7; i >= 0; i--) {
		if (h[i] > c_yp_target[i]) return false;
		if (h[i] < c_yp_target[i]) return true;
	}
	return true;                     /* equal counts as a hit, as fulltest does */
}

/* One block = one hash instance = 4 pwxform lanes. A wider block whose extra
 * threads exist only to move V is a possible future shape; not built. */
template<uint32_t R, int PLACE, int HEAD>
__global__ __launch_bounds__(YP_WIDTH, YP_MINBLK)
void yespower_gpu_hash(const uint32_t startNonce, const uint32_t N,
                       uint32_t *__restrict__ Vs, uint32_t *__restrict__ Bs,
                       uint32_t *__restrict__ Xs, uint4 *__restrict__ Sg,
                       uint32_t *__restrict__ resNonces)
{
	/* Declared unconditionally; unused (and launched with 0 bytes) when the S-box
	 * lives in the global arena.  PLACE is a template constant, so the select below
	 * folds and the shared path keeps its exact previous code. */
	extern __shared__ uint4 s_S[];

	/* The smix X buffer lives in shared, but only on the global-S path: on the
	 * shared-S path S already takes 98304 B of the 101376 B a block may opt into,
	 * so another 4096 B would fail the launch.  PLACE is a template constant, so
	 * the array folds to one element there. */
	__shared__ uint32_t s_X[YP_X_SHARED(PLACE) ? 32u * R : 1u];

	const int j = threadIdx.x & 3;
	const uint32_t inst = blockIdx.x;
	const uint32_t nonce = startNonce + inst;
	const uint32_t bw = 32u * R;
	uint32_t out[8];

	uint4 *S = (PLACE == PWX_S_SHARED) ? s_S
	                                   : (Sg + (size_t)inst * YP_SBOX_UINT4);
	uint32_t *Xp = YP_X_SHARED(PLACE) ? s_X : (Xs + (size_t)inst * bw);

	yespower_hash_1_0<R, PLACE, HEAD, YP_WIDTH>(c_yp_hdr, nonce, N, S,
	                     Bs + (size_t)inst * bw,
	                     Xp,
	                     Vs + (size_t)inst * bw * N,
	                     out, j, 0xfu, (int)threadIdx.x, YP_WMASK);

	/* Candidate report, in the shape of the tree's reference implementation
 * (`algos/blake2s/blake2s.cu`): an atomic COUNT in slot 0, a bounded write into
 * the slots after it, and the count -- not an in-band nonce -- as the sentinel,
 * so "0 is a legal nonce" cannot bite.  Ordering is the host's job.
 *
 * The previous form (`atomicExch` into slot 0, previous value into slot 1) lost
 * a candidate whenever two blocks reported at once, dropped the earliest of
 * three, and left slot 0 holding the last reporter rather than the lowest --
 * which made the host cursor skip every nonce below it. */

	/*  `threadIdx.x == 0`, NOT `j == 0`.  `j = threadIdx.x & 3`, so at any
	 * WIDTH > 4 EIGHT threads satisfy `j == 0` while only `tid < 4` ever computes
	 * `out[]` -- the rest return early from the tail.  The old guard let seven
	 * threads read an UNINITIALISED `out[]` and race on the result slot.
	 * Identical to `j == 0` at the shipped WIDTH=4, which is why it hid. */
	if (threadIdx.x == 0 && yp_below_target(out)) {
		const uint32_t pos = atomicAdd(&resNonces[0], 1u);
		if (pos < YP_MAX_RES) resNonces[1u + pos] = nonce;
		/* overflow is counted by the atomic and reported host-side, never
		 * swallowed -- see the `candidates flood` log below. */
	}
}

/* Two instruments, deliberately different: the self-test is a fail-closed gate
 * that runs at init on the user's card, the `-D` differential runs once per job
 * on real stratum headers.  Both inline the same `__device__ __forceinline__`
 * hash body the mining kernel uses, so neither can drift from what ships.
 *
 * Adding another `__global__` here invalidates any per-kernel `ptxas -v` recipe
 * written when this TU had one: scan forward from the entry-function line. */

/* The digest folds into two order-independent accumulators: a plain XOR and a
 * nonce-weighted one.  Both are load-bearing.  The plain sum is permutation
 * blind, and it is also blind to any perturbation applied uniformly across an
 * even span, because the constant cancels with itself; only the weighted term
 * sees either.  The weight must be `2*nonce+1`, never `nonce|1`, which gives
 * the aligned pair 2k/2k+1 the same weight. */
template<uint32_t R, int PLACE, int HEAD>
__global__ __launch_bounds__(YP_WIDTH, YP_MINBLK)
void yespower_gpu_checksum(const uint32_t startNonce, const uint32_t N,
                           uint32_t *__restrict__ Vs, uint32_t *__restrict__ Bs,
                           uint32_t *__restrict__ Xs, uint4 *__restrict__ Sg,
                           unsigned long long *__restrict__ acc)
{
	extern __shared__ uint4 s_S[];

	/* The smix X buffer in SHARED, but ONLY on the global-S path.
	 *
	 *  The guard is not cosmetic: on the shared-S path S already takes 98304 B and
	 * sm_86 allows 101376 B opt-in per block, so adding 4096 B here would push the
	 * launch past the limit and fail at runtime.  PLACE is a template constant, so
	 * the array folds to one element on that path.
	 *
	 * The sign of this flips with the instance count on Ampere, where L1 and
	 * shared come out of one pool per SM: shared taken for X is taken from the
	 * L1 the now-global S-box and V depend on.  */
	__shared__ uint32_t s_X[YP_X_SHARED(PLACE) ? 32u * R : 1u];

	const int j = threadIdx.x & 3;
	const uint32_t inst = blockIdx.x;
	const uint32_t nonce = startNonce + inst;
	const uint32_t bw = 32u * R;
	uint32_t out[8];

	uint4 *S = (PLACE == PWX_S_SHARED) ? s_S
	                                   : (Sg + (size_t)inst * YP_SBOX_UINT4);
	uint32_t *Xp = YP_X_SHARED(PLACE) ? s_X : (Xs + (size_t)inst * bw);

	yespower_hash_1_0<R, PLACE, HEAD, YP_WIDTH>(c_yp_hdr, nonce, N, S,
	                     Bs + (size_t)inst * bw,
	                     Xp,
	                     Vs + (size_t)inst * bw * N,
	                     out, j, 0xfu, (int)threadIdx.x, YP_WMASK);

	/*  `threadIdx.x == 0`, NOT `j == 0`.  `j = threadIdx.x & 3`, so at any
	 * WIDTH > 4 EIGHT threads satisfy `j == 0` while only `tid < 4` ever computes
	 * `out[]` -- the rest return early from the tail.  The old guard let seven
	 * threads read an UNINITIALISED `out[]` and race on the result slot.
	 * Identical to `j == 0` at the shipped WIDTH=4, which is why it hid. */
	if (threadIdx.x == 0) {
		unsigned long long q = ((unsigned long long) out[7] << 32) | out[6];
		unsigned long long w = 2ull * (unsigned long long) nonce + 1ull;
#ifdef YP_FAULT_DIFF_CONST
		/* Fault injection: perturb EVERY digest identically.  Over an even span
		 * acc[0] cannot see this; acc[1] must. */
		q ^= 1ull;
#endif
#ifdef YP_FAULT_DIFF_PERM
		/* Fault injection: pair each digest with the wrong nonce's weight.
		 * acc[0] cannot see this by construction; acc[1] must. */
		w = 2ull * (unsigned long long) (nonce ^ 1u) + 1ull;
#endif
		atomicXor(&acc[0], q);
		atomicXor(&acc[1], q * w);
	}
}

static THREAD uint32_t *d_V[MAX_GPUS]  = { 0 };
static THREAD uint32_t *d_B[MAX_GPUS]  = { 0 };
static THREAD uint32_t *d_X[MAX_GPUS]  = { 0 };
static THREAD uint4    *d_S[MAX_GPUS]  = { 0 };   /* global S arena; NULL on the shared path */
static THREAD bool      yp_sglobal[MAX_GPUS] = { false };
static THREAD int       yp_head[MAX_GPUS]    = { YP_HEAD_SHA256 };  /* head/tail primitive */
static THREAD uint32_t *d_res[MAX_GPUS] = { 0 };
static THREAD uint32_t yp_instances[MAX_GPUS] = { 0 };
static bool init[MAX_GPUS] = { false };

/* Dispatch on r. Only the three shapes the variants actually use are instantiated;
 * a generic runtime r would put the 2R block loop out of the compiler's reach.
 * r=8 exists for Tidecoin and is the one shape no KAT has covered yet. */
static bool yp_launch(uint32_t r, uint32_t instances, uint32_t startNonce, uint32_t N,
                      int dev, bool sglobal, int head)
{
	const uint32_t shbytes = sglobal ? 0u : (YP_SBOX_UINT4 * 16u);

#define YP_LAUNCH_1(RR, PL, HD, SH)                                              	yespower_gpu_hash<RR, PL, HD><<<instances, YP_WIDTH, (SH)>>>(startNonce, N,         		d_V[dev], d_B[dev], d_X[dev], d_S[dev], d_res[dev])

#define YP_LAUNCH(RR)                                                            	do {                                                                         		if (sglobal && head == YP_HEAD_B2B) YP_LAUNCH_1(RR, PWX_S_GLOBAL, YP_HEAD_B2B, 0); 		else if (sglobal)          YP_LAUNCH_1(RR, PWX_S_GLOBAL, YP_HEAD_SHA256, 0); 		else if (head == YP_HEAD_B2B)       YP_LAUNCH_1(RR, PWX_S_SHARED, YP_HEAD_B2B, shbytes); 		else                       YP_LAUNCH_1(RR, PWX_S_SHARED, YP_HEAD_SHA256, shbytes); 	} while (0)

	/* EqPay: r is fixed at 32, so the 181-byte head is instantiated only here
	 * rather than for every r. */
	if (head == YP_HEAD_SHA256_181) {
		if (r != 32) return false;
		if (sglobal) YP_LAUNCH_1(32, PWX_S_GLOBAL, YP_HEAD_SHA256_181, 0);
		else         YP_LAUNCH_1(32, PWX_S_SHARED, YP_HEAD_SHA256_181, shbytes);
		return true;
	}

	switch (r) {
	case 8:  YP_LAUNCH(8);  return true;
	case 16: YP_LAUNCH(16); return true;
	case 32: YP_LAUNCH(32); return true;
	default: return false;
	}
#undef YP_LAUNCH
#undef YP_LAUNCH_1
}

static bool yp_set_shared_limit(uint32_t r, int head)
{
	const uint32_t shbytes = YP_SBOX_UINT4 * 16u;
	cudaError_t e;

/*  The opt-in is PER KERNEL. The checksum kernel asks for the same 98304 B, so	 * it needs its own cudaFuncSetAttribute or every instrument launch fails with	 * `invalid argument` on sm_86 while mining works perfectly. */	#define YP_OPTIN(RR)                                                                 	((head == YP_HEAD_B2B) ? (cudaFuncSetAttribute(yespower_gpu_hash<RR, PWX_S_SHARED, YP_HEAD_B2B>,    	                            cudaFuncAttributeMaxDynamicSharedMemorySize, shbytes)	        , cudaFuncSetAttribute(yespower_gpu_checksum<RR, PWX_S_SHARED, YP_HEAD_B2B>,	                            cudaFuncAttributeMaxDynamicSharedMemorySize, shbytes))	     : (cudaFuncSetAttribute(yespower_gpu_hash<RR, PWX_S_SHARED, YP_HEAD_SHA256>, 	                            cudaFuncAttributeMaxDynamicSharedMemorySize, shbytes)	        , cudaFuncSetAttribute(yespower_gpu_checksum<RR, PWX_S_SHARED, YP_HEAD_SHA256>,	                            cudaFuncAttributeMaxDynamicSharedMemorySize, shbytes)))

	if (head == YP_HEAD_SHA256_181) {
		if (r != 32) return false;
		e = cudaFuncSetAttribute(yespower_gpu_hash<32, PWX_S_SHARED, YP_HEAD_SHA256_181>,
		                         cudaFuncAttributeMaxDynamicSharedMemorySize, shbytes);
		if (e == cudaSuccess)
			e = cudaFuncSetAttribute(yespower_gpu_checksum<32, PWX_S_SHARED, YP_HEAD_SHA256_181>,
			                         cudaFuncAttributeMaxDynamicSharedMemorySize, shbytes);
		return e == cudaSuccess;
	}

	switch (r) {
	case 8:  e = YP_OPTIN(8);  break;
	case 16: e = YP_OPTIN(16); break;
	case 32: e = YP_OPTIN(32); break;
	default: return false;
	}
#undef YP_OPTIN
	return e == cudaSuccess;
}


/* ---------------------------------------------------------------------------
 * Instrument plumbing.  A span that is a multiple of the launch width, or that
 * starts at 0, leaves the ragged tail and the startNonce arithmetic untested.
 *
 * 53 and 13 are prime, so neither divides any instance count this algo uses
 * (28 on the shared path, 448 on the global one); 0x51 puts the first nonce
 * off zero.  With 28 instances a 53-span is 28 + 25, i.e. a deliberate short
 * final launch.
 * ------------------------------------------------------------------------- */
#define YP_DIFF_SPAN 53u
#define YP_DIFF_OFF  0x51u
#define YP_ST_SPAN   13u
#define YP_ST_OFF    0x51u
#define YP_ST_EVEN   12u    /* even sub-span: see the acc[0] blindness leg */

static bool yp_checksum_launch(uint32_t r, uint32_t instances, uint32_t startNonce,
                               uint32_t N, int dev, bool sglobal, int head,
                               unsigned long long *d_acc)
{
	const uint32_t shbytes = sglobal ? 0u : (YP_SBOX_UINT4 * 16u);

#define YP_CK_1(RR, PL, HD, SH)                                                  \
	yespower_gpu_checksum<RR, PL, HD><<<instances, YP_WIDTH, (SH)>>>(startNonce, N,     \
		d_V[dev], d_B[dev], d_X[dev], d_S[dev], d_acc)

#define YP_CK(RR)                                                                \
	do {                                                                         \
		if (sglobal && head == YP_HEAD_B2B) YP_CK_1(RR, PWX_S_GLOBAL, YP_HEAD_B2B, 0); \
		else if (sglobal)          YP_CK_1(RR, PWX_S_GLOBAL, YP_HEAD_SHA256, 0); \
		else if (head == YP_HEAD_B2B)       YP_CK_1(RR, PWX_S_SHARED, YP_HEAD_B2B, shbytes); \
		else                       YP_CK_1(RR, PWX_S_SHARED, YP_HEAD_SHA256, shbytes); \
	} while (0)

	/* EqPay, same r=32 restriction as yp_launch. Without this branch the
	 * instruments would fall through to the 80-byte head and compare the GPU
	 * against a hash the miner does not compute -- which is what a green
	 * self-test over the wrong function looks like. */
	if (head == YP_HEAD_SHA256_181) {
		if (r != 32) return false;
		if (sglobal) YP_CK_1(32, PWX_S_GLOBAL, YP_HEAD_SHA256_181, 0);
		else         YP_CK_1(32, PWX_S_SHARED, YP_HEAD_SHA256_181, shbytes);
		return true;
	}

	switch (r) {
	case 8:  YP_CK(8);  return true;
	case 16: YP_CK(16); return true;
	case 32: YP_CK(32); return true;
	default: return false;
	}
#undef YP_CK
#undef YP_CK_1
}

/* Accumulate over [start, start+span) on the GPU, in chunks no wider than the
 * allocated instance count -- d_V/d_B/d_X are sized for that and nothing else. */
static bool yp_gpu_acc(int dev, const yespower_params_t *p,
                       uint32_t start, uint32_t span, unsigned long long acc[2])
{
	unsigned long long *d_acc = NULL;
	unsigned long long zero[2] = { 0ull, 0ull };
	bool ok = true;

	if (cudaMalloc(&d_acc, sizeof(zero)) != cudaSuccess) return false;
	if (cudaMemcpy(d_acc, zero, sizeof(zero), cudaMemcpyHostToDevice) != cudaSuccess) {
		cudaFree(d_acc);
		return false;
	}
	for (uint32_t done = 0; done < span && ok; ) {
		uint32_t chunk = span - done;
		if (chunk > yp_instances[dev]) chunk = yp_instances[dev];
		ok = yp_checksum_launch(p->r, chunk, start + done, p->N, dev,
		                        yp_sglobal[dev], yp_head[dev], d_acc)
		     && cudaDeviceSynchronize() == cudaSuccess;
		done += chunk;
	}
	ok = ok && cudaMemcpy(acc, d_acc, sizeof(zero), cudaMemcpyDeviceToHost) == cudaSuccess;
	cudaFree(d_acc);
	return ok;
}

/* One CPU pass over the span, producing every accumulator the legs need.
 * `flip` perturbs the HEADER (a different job, so the digests must differ);
 * the perm/const variants perturb the accumulation instead, which is what makes
 * them demonstrations of the two blindnesses rather than of the hash. */
static bool yp_cpu_acc(const uint32_t *pdata, const yespower_params_t *p,
                       uint32_t start, uint32_t span, bool flip,
                       unsigned long long plain[2], unsigned long long perm[2],
                       unsigned long long even_plain[2], unsigned long long even_const[2])
{
	uint32_t endian[YP_EQPAY_HDR_WORDS], vhash[8];
	const size_t hlen = yp_header_bytes(opt_algo);
	const uint32_t hwords = (uint32_t)((hlen + 3u) / 4u);

	if (plain)      plain[0]      = plain[1]      = 0ull;
	if (perm)       perm[0]       = perm[1]       = 0ull;
	if (even_plain) even_plain[0] = even_plain[1] = 0ull;
	if (even_const) even_const[0] = even_const[1] = 0ull;

	for (uint32_t k = 0; k < hwords; k++) be32enc(&endian[k], pdata[k]);
	if (flip) endian[3] ^= 0x00000001u;

	for (uint32_t i = 0; i < span; i++) {
		const uint32_t nonce = start + i;
		unsigned long long q, w;

		be32enc(&endian[19], nonce);
		if (!yespower_hash_hdr(vhash, endian, hlen, p)) return false;

		q = ((unsigned long long) vhash[7] << 32) | vhash[6];
		w = 2ull * (unsigned long long) nonce + 1ull;

		if (plain) { plain[0] ^= q; plain[1] ^= q * w; }
		if (perm)  { perm[0]  ^= q; perm[1]  ^= q * (2ull * (unsigned long long)(nonce ^ 1u) + 1ull); }
		if (i < YP_ST_EVEN) {
			if (even_plain) { even_plain[0] ^= q;        even_plain[1] ^= q * w; }
			if (even_const) { even_const[0] ^= (q ^ 1ull); even_const[1] ^= (q ^ 1ull) * w; }
		}
	}
	return true;
}

/* EqPay genesis gate: the digest of the chain's own genesis header, so unlike
 * every other check here it is not satisfied by an in-tree mistake that the GPU
 * and the CPU reference share.  The two perturbation legs are length probes --
 * bytes 80 and 180 are unreachable if the head hashed 80 or 180 bytes. */
static bool yespower_eqpay_genesis_kat(int thr_id, const yespower_params_t *p)
{
	static const uint8_t kat_hdr[181] = {
		0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0xe1, 0xa7, 0xff, 0x0f, 0x99, 0xf5, 0xe6, 0x92, 0xa1, 0x6f, 0x69, 0x69,
		0xe0, 0x09, 0xca, 0xf9, 0x51, 0x86, 0x57, 0xe6, 0x7b, 0x12, 0xa5, 0xac,
		0x15, 0xf8, 0x53, 0x39, 0x5c, 0x7f, 0xed, 0xaa, 0x25, 0xc6, 0x55, 0x61,
		0xff, 0xff, 0x3f, 0x1f, 0x53, 0x01, 0x00, 0x00, 0xe9, 0x65, 0xff, 0xd0,
		0x02, 0xcd, 0x6a, 0xd0, 0xe2, 0xdc, 0x40, 0x2b, 0x80, 0x44, 0xde, 0x83,
		0x3e, 0x06, 0xb2, 0x31, 0x27, 0xea, 0x8c, 0x3d, 0x80, 0xae, 0xc9, 0x14,
		0x10, 0x77, 0x14, 0x95, 0x56, 0xe8, 0x1f, 0x17, 0x1b, 0xcc, 0x55, 0xa6,
		0xff, 0x83, 0x45, 0xe6, 0x92, 0xc0, 0xf8, 0x6e, 0x5b, 0x48, 0xe0, 0x1b,
		0x99, 0x6c, 0xad, 0xc0, 0x01, 0x62, 0x2f, 0xb5, 0xe3, 0x63, 0xb4, 0x21,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff,
		0x00,
	};

	static const uint8_t kat_digest[32] = {
		0x56, 0xda, 0x66, 0x12, 0x94, 0xe4, 0xac, 0xf5, 0x60, 0xea, 0x4d, 0x73,
		0x12, 0x1a, 0xc9, 0xd8, 0xd8, 0xf8, 0xd3, 0x09, 0xf0, 0x6e, 0xe3, 0x32,
		0x4e, 0x7b, 0xe3, 0x36, 0x01, 0x56, 0x04, 0x00,
	};

	uint8_t out[32], alt[YP_EQPAY_HDR_BYTES], tmp[32];
	bool digest_ok, roots_ok, siglen_ok, passed;

	if (!yespower_hash_hdr(out, kat_hdr, YP_EQPAY_HDR_BYTES, p)) {
		gpulog(LOG_ERR, thr_id, "eqpay self-test: the CPU reference refused the "
		                        "parameters (N=%u r=%u)", p->N, (uint32_t) p->r);
		return selftest_gate(thr_id, "eqpay", false);
	}
	digest_ok = (memcmp(out, kat_digest, 32) == 0);

	memcpy(alt, kat_hdr, YP_EQPAY_HDR_BYTES);
	alt[YP_EQPAY_STATE_ROOT] ^= 0x01;
	roots_ok = yespower_hash_hdr(tmp, alt, YP_EQPAY_HDR_BYTES, p) &&
	           memcmp(tmp, out, 32) != 0;

	memcpy(alt, kat_hdr, YP_EQPAY_HDR_BYTES);
	alt[YP_EQPAY_SIGLEN] ^= 0x01;
	siglen_ok = yespower_hash_hdr(tmp, alt, YP_EQPAY_HDR_BYTES, p) &&
	            memcmp(tmp, out, 32) != 0;

	passed = digest_ok && roots_ok && siglen_ok;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "eqpay self-test FAILED: digest=%d roots=%d siglen=%d "
		                        "(%u-byte header)",
		       (int) digest_ok, (int) roots_ok, (int) siglen_ok,
		       (unsigned) YP_EQPAY_HDR_BYTES);

	return selftest_gate(thr_id, "eqpay", passed);
}

/* Fail-closed init gate.  Four legs; the last two need no GPU and are the ones
 * that keep the instrument honest, because they PROVE the two documented
 * blindnesses instead of citing them. */
static bool yespower_device_selftest(int thr_id, int dev, const yespower_params_t *p)
{
	/* EqPay first: it is the only leg here with an external oracle, and if the
	 * parameters or the header length are wrong there is no point reporting a
	 * GPU-vs-CPU agreement built on them. */
	if (opt_algo == ALGO_YESPOWEREQPAY && !yespower_eqpay_genesis_kat(thr_id, p))
		return false;

	/* A fixed header, so the gate is reproducible run to run.  c_yp_hdr is
	 * re-uploaded with the real job on every scanhash call, so borrowing it
	 * here cannot leak into mining. */
	static const uint32_t testhdr[YP_EQPAY_HDR_WORDS] = {
		0x20000000u, 0x0f1e2d3cu, 0x4b5a6978u, 0x8796a5b4u, 0xc3d2e1f0u,
		0x11223344u, 0x55667788u, 0x99aabbccu, 0xddeeff00u, 0x0badc0deu,
		0xfeedfaceu, 0xcafebabeu, 0x13579bdfu, 0x2468ace0u, 0xa5a5a5a5u,
		0x5a5a5a5au, 0x00ff00ffu, 0xff00ff00u, 0x1a2b3c4du, 0x00000000u,
		/* EqPay tail in the shape stratum_gen_work builds. Ignored unless the
		 * algo hashes 181 bytes, but it must exist: otherwise that head reads
		 * uninitialised constant memory and fails for the wrong reason. */
		0x01234567u, 0x89abcdefu, 0x1a2b3c4du, 0x5e6f7a8bu,
		0x00112233u, 0x44556677u, 0x8899aabbu, 0xccddeeffu,
		0xdeadbeefu, 0xfeedfaceu, 0x0badf00du, 0xcafed00du,
		0x13579bdfu, 0x02468aceu, 0xfdb97531u, 0xeca86420u,
		0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
		0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
		0xffffffffu,   /* prevoutStake.n */
		0x00000000u    /* signature length byte, then padding */
	};
	const uint32_t st_words = (uint32_t)((yp_header_bytes(opt_algo) + 3u) / 4u);
	unsigned long long g[2], cpu[2], cneg[2], perm[2], ev[2], evc[2];
	bool kat, neg, permleg, constleg, passed;

	if (cudaMemcpyToSymbol(c_yp_hdr, testhdr, st_words * sizeof(uint32_t)) != cudaSuccess)
		return selftest_gate(thr_id, "yespower", selftest_cuda_fault());

	if (!yp_gpu_acc(dev, p, YP_ST_OFF, YP_ST_SPAN, g))
		return selftest_gate(thr_id, "yespower", selftest_cuda_fault());

	if (!yp_cpu_acc(testhdr, p, YP_ST_OFF, YP_ST_SPAN, false, cpu, perm, ev, evc) ||
	    !yp_cpu_acc(testhdr, p, YP_ST_OFF, YP_ST_SPAN, true,  cneg, NULL, NULL, NULL)) {
		gpulog(LOG_ERR, thr_id, "yespower self-test: the CPU reference refused the "
		                        "parameters (N=%u r=%u)", p->N, (uint32_t) p->r);
		return selftest_gate(thr_id, "yespower", false);
	}

	/* 1. the gate proper: this card reproduces the in-tree reference. */
	kat = (g[0] == cpu[0] && g[1] == cpu[1]);
	/* 2. and it does NOT reproduce a different job -- without this, an
	 *    instrument that returned a constant would pass leg 1. */
	neg = (g[0] != cneg[0] || g[1] != cneg[1]);
	/* 3. acc[0] is permutation-blind; acc[1] is not. */
	permleg = (perm[0] == cpu[0] && perm[1] != cpu[1]);
	/* 4. over an EVEN span a uniform perturbation cancels in acc[0]; acc[1]
	 *    still moves.  Predicting which accumulator moves is the point: get
	 *    that prediction wrong and the blindness is what surfaces. */
	constleg = (evc[0] == ev[0] && evc[1] != ev[1]);

	passed = kat && neg && permleg && constleg;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "yespower self-test FAILED: kat=%d neg=%d perm=%d const=%d "
		                        "(gpu %016llx/%016llx vs cpu %016llx/%016llx)",
		       (int) kat, (int) neg, (int) permleg, (int) constleg,
		       (unsigned long long) g[0], (unsigned long long) g[1],
		       (unsigned long long) cpu[0], (unsigned long long) cpu[1]);

	return selftest_gate(thr_id, "yespower", passed);
}

/* `-D` differential: once per JOB, over real stratum headers.  Keyed off an
 * FNV-1a of pdata[0..18] -- the nonce word is excluded, or this would fire on
 * every scanhash call instead of every job. */
static THREAD uint32_t yp_diff_sig[MAX_GPUS] = { 0 };

static uint32_t yp_job_sig(const uint32_t pdata[20])
{
	uint32_t h = 2166136261u;
	for (int i = 0; i < 19; i++) {
		const uint32_t w = pdata[i];
		for (int b = 0; b < 4; b++) { h ^= (w >> (b * 8)) & 0xffu; h *= 16777619u; }
	}
	return h ? h : 1u;
}

static void yespower_debug_differential(int thr_id, int dev, const uint32_t pdata[20],
                                        const yespower_params_t *p, uint32_t base)
{
	unsigned long long g[2], c[2];
	const uint32_t sig = yp_job_sig(pdata);
	const uint32_t start = base + YP_DIFF_OFF;

	if (yp_diff_sig[dev] == sig) return;
	yp_diff_sig[dev] = sig;

	if (!yp_gpu_acc(dev, p, start, YP_DIFF_SPAN, g)) {
		gpulog(LOG_WARNING, thr_id, "yespower differential: could not run (CUDA resource "
		                            "failure) -- not evidence of a wrong hash");
		return;
	}
	if (!yp_cpu_acc(pdata, p, start, YP_DIFF_SPAN, false, c, NULL, NULL, NULL)) {
		gpulog(LOG_WARNING, thr_id, "yespower differential: CPU reference refused the parameters");
		return;
	}

	if (g[0] == c[0] && g[1] == c[1])
		gpulog(LOG_DEBUG, thr_id, "yespower differential ok: %u nonces from %08x, "
		                          "acc %016llx/%016llx",
		       YP_DIFF_SPAN, start, (unsigned long long) g[0], (unsigned long long) g[1]);
	else
		gpulog(LOG_ERR, thr_id, "yespower DIFFERENTIAL MISMATCH over %u nonces from %08x: "
		                        "gpu %016llx/%016llx != cpu %016llx/%016llx",
		       YP_DIFF_SPAN, start,
		       (unsigned long long) g[0], (unsigned long long) g[1],
		       (unsigned long long) c[0], (unsigned long long) c[1]);
}

extern "C" int scanhash_yespower(int thr_id, struct work *work, uint32_t max_nonce,
                                 unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	uint32_t endiandata[YP_EQPAY_HDR_WORDS];   /* 46: sized for the longest header */
	uint32_t vhash[8];
	yespower_params_t params;
	uint32_t n = first_nonce;
	const int dev = device_map[thr_id];
	static THREAD bool announced = false;
	time_t t_start;

	if (!yespower_params_for(opt_algo, &params)) {
		applog(LOG_ERR, "yespower: algo %d is not a yespower variant", opt_algo);
		proper_exit(EXIT_CODE_USAGE);   /* permanent */
	}

	/* Reject up front exactly what the reference rejects (yespower_ref.c:495):
	 * N a power of two in [1024, 512K], r in [8, 32].  Without this the reference
	 * returns -1 for every nonce, the fail-closed digest below is all-ones, no
	 * share is ever found -- and the scan loop spins at memset speed and reports
	 * a completely fictitious hashrate.  `--yespower-param 128,2` read as
	 * fast on a CPU. A validity check that only runs per-hash is not enough
	 * when the failure path is cheaper than the success path. */
	if (params.N < 1024 || params.N > 512 * 1024 || (params.N & (params.N - 1))) {
		applog(LOG_ERR, "yespower: N=%u invalid (need a power of two in [1024, 524288])",
		       params.N);
		proper_exit(EXIT_CODE_USAGE);   /* permanent: the next call would fail identically */
	}
	if (params.r < 8 || params.r > 32) {
		applog(LOG_ERR, "yespower: r=%u invalid (need 8..32)", params.r);
		proper_exit(EXIT_CODE_USAGE);   /* permanent */
	}

	/* r is a template parameter of the kernel, so only the shapes the variants
	 * use exist. Reject anything else here rather than at launch, where the
	 * failure would look like a dead GPU. */
	if (params.r != 8 && params.r != 16 && params.r != 32) {
		applog(LOG_ERR, "yespower: r=%u has no GPU kernel (only 8, 16 and 32 are built)",
		       params.r);
		proper_exit(EXIT_CODE_USAGE);   /* permanent */
	}

	if (!announced) {
		announced = true;
		applog(LOG_NOTICE, "yespower 1.0: N=%u r=%u key=%s (%u bytes)",
		       params.N, params.r,
		       params.perslen ? (const char *) params.pers : "<none>",
		       (uint32_t) params.perslen);
	}

	/* NOT the tree's usual 0x0000ff, and worth reading before "tightening" back.
	 * Benchmark work is regenerated every pass with the same header and data[19]
	 * reset to 0, so at a few hundred H/s the miner rescans one FIXED ~250-nonce
	 * window rather than sampling fresh nonces. What matters is therefore whether
	 * that single window contains a hit: 0x0000ff gives p=1/16.7M (empty nearly
	 * always), 0x00ffffff p=1/256 (often empty), 0x07ffffff p=1/32.
	 * Loosening this is not a correctness risk -- it only affects benchmarks, and
	 * every candidate is still re-hashed on the host. */
	if (opt_benchmark)
		ptarget[7] = 0x07ffffff;

	/* Follow ccminer's OWN yescrypt (algos/yescrypt/yescrypt.cu), not cpuminer's:
	 * cpuminer-opt leaves the nonce raw, while ccminer's yescrypt be32enc's all 20
	 * words and submits with the switch's default le32enc.  The two miners keep
	 * work->data[19] in different conventions, so the sibling's form does not
	 * transfer. */
	const size_t hdr_bytes  = yp_header_bytes(opt_algo);
	const uint32_t hdr_words = (uint32_t)((hdr_bytes + 3u) / 4u);

	/* One rule for every word: endiandata is the wire byte stream, and the GPU
	 * reads the same words as SHA-256 input. stratum_gen_work stores EqPay's
	 * extended tail as the big-endian decode of the wire for exactly this reason,
	 * so no part of the header needs a special case here. */
	for (uint32_t k = 0; k < hdr_words; k++)
		be32enc(&endiandata[k], pdata[k]);

	if (!init[dev]) {
		const uint32_t shneed = YP_SBOX_UINT4 * 16u;
		int optin = 0;
		CUDA_CALL_OR_RET_X(cudaSetDevice(device_map[thr_id]), -1);

		cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin,
		                       device_map[thr_id]);
		/* S-box placement.
 *
 * yespower 1.0 fixes Swidth = 11, so S is 98304 B per instance.  A card that
 * can opt a block in to that could keep S in shared, but doing so pins the SM
 * to one block; the global arena gives that up and buys many more instances,
 * which is far better.  So the shared path is a last resort, taken only
 * when the global arena does not fit in VRAM. */

		/* Chosen by VRAM, not by capability: the global arena wins on every card that
	 * has room for a useful number of instances.  Instance count saturates well
	 * before VRAM does, so the threshold is deliberately low. */
		const size_t bw_q      = 32u * (size_t) params.r;
		const size_t v_inst_q  = bw_q * 4 * (size_t) params.N;
		const size_t per_inst_q = v_inst_q + bw_q * 4 * 2 + (size_t) shneed;
		size_t avail_q = (size_t) cuda_available_memory(thr_id) * 1024u * 1024u;
		avail_q = (avail_q > (256u << 20)) ? (avail_q - (256u << 20)) : 0u;
		const uint32_t fit_q = (uint32_t) (avail_q / per_inst_q);

			/* Even the narrowest global arm beats the shared path, so that is the
	 * whole condition. */
		const bool global_fits = fit_q >= (uint32_t) device_mpcount[dev] * 4u;   /* 4/SM already beats shared 1.30x */
		yp_sglobal[dev] = global_fits || ((uint32_t) optin < shneed);

		if (!yp_sglobal[dev] && (uint32_t) optin >= shneed)
			applog(LOG_WARNING, "yespower: GPU #%d has room for only %u instances; "
			                    "falling back to the shared S-box, which is ~2.1x slower",
			       device_map[thr_id], fit_q);
		/* ALGO_POWER2B is yespower-b2b: same (N, r, pers) machinery, BLAKE2b head
		 * and tail instead of SHA-256.  `-a yespower-b2b` aliases to it in algos.h. */
		yp_head[dev] = (opt_algo == ALGO_POWER2B)    ? YP_HEAD_B2B
	             : (opt_algo == ALGO_YESPOWEREQPAY) ? YP_HEAD_SHA256_181
	                                                : YP_HEAD_SHA256;
		if (yp_sglobal[dev] && (uint32_t) optin < shneed)
			applog(LOG_INFO, "yespower: GPU #%d offers %d B of shared memory per block "
			                 "(needs %u), using the global S-box path",
			       device_map[thr_id], optin, shneed);
		else if (yp_sglobal[dev])
			applog(LOG_INFO, "yespower: GPU #%d using the global S-box path "
			                 "(~2.1x the shared path -- occupancy beats access cost)",
			       device_map[thr_id]);

		if (!yp_sglobal[dev] && !yp_set_shared_limit(params.r, yp_head[dev])) {
			applog(LOG_ERR, "yespower: could not raise the dynamic shared limit to %u B",
			       shneed);
			proper_exit(EXIT_CODE_CUDA_ERROR);   /* permanent; see the arch gate above */
		}

		const size_t bw      = 32u * (size_t) params.r;
		const size_t sbytes  = (size_t) YP_SBOX_UINT4 * 16u;
		const size_t v_inst  = bw * 4 * (size_t) params.N;      /* V per instance */
		const size_t bx_inst = bw * 4 * 2;                      /* B + X          */

		if (!yp_sglobal[dev]) {
			/* S fills all of shared memory, so exactly one block is resident per SM;
			 * more blocks than that only queue up behind each other. */
			yp_instances[dev] = (uint32_t) device_mpcount[dev];
		} else {
			/* The whole point of the global path: with no 96 KiB shared allocation the
			 * SM is no longer capped at one block, so run many instances and let the
			 * occupancy hide the slower S access.  That is what makes this viable at
			 * all -- the per-access cost goes UP and the throughput still goes up.
			 * V scales with the instance count, so clamp to what actually fits rather
			 * than trusting the multiplier. */
			const size_t per_inst = v_inst + bx_inst + sbytes;
			size_t avail = (size_t) cuda_available_memory(thr_id) * 1024u * 1024u;
			const size_t reserve = 256u * 1024u * 1024u;
			avail = (avail > reserve) ? (avail - reserve) : 0u;

			uint32_t fit  = (uint32_t) (avail / per_inst);
			/* device_sm[] is the compute capability x10 (860 = sm_86), read from the
			 * device at init -- guideline, and the same mechanism the rest of
			 * the tree uses for arch-dependent host decisions. */
			const uint32_t ipb = (device_sm[dev] >= 700) ? YP_IPB_AMPERE : YP_IPB_OTHER;
			uint32_t want = (uint32_t) device_mpcount[dev] * ipb;
			fit -= fit % 4u;                       /* whole quads of instances */
			yp_instances[dev] = (want < fit) ? want : fit;

			if (yp_instances[dev] < 4u) {
				applog(LOG_ERR, "yespower: GPU #%d needs %.0f MB per instance and has "
				                "%d MB free -- not enough for the global S-box path",
				       device_map[thr_id], (double) per_inst / (1024.0 * 1024.0),
				       cuda_available_memory(thr_id));
				proper_exit(EXIT_CODE_CUDA_ERROR);
			}
			if (yp_instances[dev] < want)
				applog(LOG_WARNING, "yespower: GPU #%d VRAM caps this at %u instances "
				                    "(wanted %u) -- expect a lower rate",
				       device_map[thr_id], yp_instances[dev], want);
		}

		CUDA_CALL_OR_RET_X(cudaMalloc(&d_V[dev], v_inst * yp_instances[dev]), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_B[dev], bw * 4 * yp_instances[dev]), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_X[dev], bw * 4 * yp_instances[dev]), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_res[dev], (1u + YP_MAX_RES) * sizeof(uint32_t)), -1);
		if (yp_sglobal[dev])
			CUDA_CALL_OR_RET_X(cudaMalloc(&d_S[dev], sbytes * yp_instances[dev]), -1);

		/* pers is constant per VARIANT, not per run -- a pool-side or API algo
		 * switch can select a different coin.  Correct only because
		 * `free_yespower()` now re-arms `init[dev]`, which re-runs this upload. */
		if (params.perslen > YP_PERS_MAX) {
			applog(LOG_ERR, "yespower: pers is %u bytes, the kernel holds %u",
			       (uint32_t) params.perslen, (uint32_t) YP_PERS_MAX);
			proper_exit(EXIT_CODE_USAGE);   /* permanent */
		}
		{
			uint8_t persbuf[YP_PERS_MAX] = { 0 };
			uint32_t plen = (uint32_t) params.perslen;
			if (plen) memcpy(persbuf, params.pers, plen);
			cudaMemcpyToSymbol(c_yp_pers, persbuf, YP_PERS_MAX);
			cudaMemcpyToSymbol(c_yp_perslen, &plen, sizeof(plen));
		}

		applog(LOG_INFO, "GPU #%d: yespower %u instances, %.0f MB of V, S in %s",
		       device_map[thr_id], yp_instances[dev],
		       (double) (v_inst * (double) yp_instances[dev]) / (1024.0 * 1024.0),
		       yp_sglobal[dev] ? "global" : "shared");
		/* Fail-closed: a card that cannot reproduce the consensus hash otherwise
		 * mines a whole session producing nothing but local rejects, silently.
		 * Runs after the buffers exist, because it uses them. */
		yespower_device_selftest(thr_id, dev, &params);
		init[dev] = true;
	}

	/* Upload pdata, NOT endiandata. The kernel wants the SHA-256 input words, i.e.
	 * the big-endian decode of the header byte stream. endiandata holds that stream
	 * as bytes, so reading it as host uint32 byte-reverses every word -- and since
	 * be32enc(&endiandata[i], pdata[i]), the correct decode is just pdata[i] again.
	 * Uploading endiandata makes the GPU hash a different header than the host
	 * re-verify, which surfaces only as "does not validate". Word 19 is ignored --
	 * the nonce is passed to the kernel separately. */
	cudaMemcpyToSymbol(c_yp_hdr, pdata, hdr_words * sizeof(uint32_t));
	cudaMemcpyToSymbol(c_yp_target, ptarget, 32);

	/* -D only: one differential per JOB, against the header just uploaded. The
	 * host re-verify is one-sided -- it catches a bad candidate and is blind to a
	 * MISSED one -- so this is the only instrument here that would see a kernel
	 * silently skipping nonces. */
	if (opt_debug)
		yespower_debug_differential(thr_id, dev, pdata, &params, first_nonce);

	/* Return periodically instead of scanning to max_nonce, so the miner reports
	 * a rate and notices new work; the caller re-enters with the updated nonce. */
	t_start = time(NULL);

	do {
		const uint32_t batch = yp_instances[dev];
		/*  Arm size == readback size == allocation size, all three.  Arming
		 * fewer bytes than are read back is  (vanilla.cu armed 4 of 8 and
		 * produced 3228 initcheck errors). */
		uint32_t res[1u + YP_MAX_RES] = { 0 };

		cudaMemcpy(d_res[dev], res, sizeof(res), cudaMemcpyHostToDevice);

		if (!yp_launch(params.r, batch, n, params.N, dev, yp_sglobal[dev], yp_head[dev])) {
			applog(LOG_ERR, "yespower: no kernel for r=%u", params.r);
			return -1;
		}
		if (cudaGetLastError() != cudaSuccess ||
		    cudaMemcpy(res, d_res[dev], sizeof(res), cudaMemcpyDeviceToHost) != cudaSuccess) {
			applog(LOG_ERR, "yespower: GPU #%d launch failed: %s",
			       device_map[thr_id], cudaGetErrorString(cudaGetLastError()));
			return -1;
		}

		*hashes_done = n - first_nonce + batch;

		/* Every candidate is re-hashed on the host before it can become a share,
		 * and BOTH slots are checked -- submitting the second nonce without its
		 * own fulltest is a recurring defect elsewhere in this tree. */
		if (res[0] != 0u) {
			uint32_t cand[YP_MAX_RES];
			uint32_t ncand = res[0];
			int found = 0;

			if (ncand > YP_MAX_RES) {
				/* Logged and clamped, never swallowed -- the blake2s shape. */
				applog(LOG_WARNING, "GPU #%d: yespower candidates flood: %u (keeping %u)",
				       device_map[thr_id], ncand, (uint32_t) YP_MAX_RES);
				ncand = YP_MAX_RES;
			}
			for (uint32_t s = 0; s < ncand; s++) cand[s] = res[1u + s];

			/* Ascending, so the two LOWEST are examined first.  The kernel
			 * reports in completion order, which is not nonce order, and the
			 * cursor below depends on this. */
			for (uint32_t a = 1; a < ncand; a++) {
				const uint32_t v = cand[a];
				uint32_t b = a;
				while (b > 0 && cand[b - 1] > v) { cand[b] = cand[b - 1]; b--; }
				cand[b] = v;
			}

			for (uint32_t s = 0; s < ncand && found < 2; s++) {
				be32enc(&endiandata[19], cand[s]);
				if (!yespower_hash_hdr(vhash, endiandata, hdr_bytes, &params)) {
					applog(LOG_ERR, "yespower: host re-verify failed at nonce %08x", cand[s]);
					return -1;
				}
				if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
					work->nonces[found] = cand[s];
					if (found == 0) work_set_target_ratio(work, vhash);
					found++;
				} else {
					gpu_increment_reject(thr_id);
					applog(LOG_WARNING, "GPU #%d: yespower result %08x does not validate",
					       device_map[thr_id], cand[s]);
				}
			}
			if (found) {
				/* work->nonces[0] is what is actually submitted: miner_thread does
				 * `nonceptr[0] = work.nonces[0]` right before submit_work. Setting
				 * only pdata[19] leaves nonces[0] zeroed and every share goes out as
				 * nonce 00000000 -- "Invalid share" while the miner looks healthy. */
				work->valid_nonces = found;
				/* Resume past the HIGHEST nonce actually submitted.  `+1` so a
				 * submitted nonce is never re-scanned ; `max` because
				 * resuming at the lowest would re-find and re-submit the second.
				 * Candidates above this in the same window are given up, which is
				 * the standard trade and is why the report keeps the LOWEST. */
				pdata[19] = ((found > 1 && work->nonces[1] > work->nonces[0])
				             ? work->nonces[1] : work->nonces[0]) + 1;
				return found;
			}
		}

		n += batch;
		if (time(NULL) - t_start >= 1)
			break;                       /* yield: report the rate, poll for work */
	} while ((uint64_t) n + yp_instances[dev] < (uint64_t) max_nonce &&
	         !work_restart[thr_id].restart);

	*hashes_done = n - first_nonce;
	pdata[19] = n;
	return 0;
}

/* Releases the device buffers and re-arms `init[dev]`.
 *
 * This must stay registered in `algo_free_all()`: a pool-side or API algo
 * switch bumps `algo_switch_gen`, every miner thread calls that, and only this
 * re-arms the init guard.  Everything derived from the variant is latched under
 * that guard -- the buffer sizes (`bw = 32 * r`), the `pers` upload and the
 * head/tail primitive -- so without it a switch would run the new coin against
 * the previous coin's buffers. */
extern "C" void free_yespower(int thr_id)
{
	const int dev = device_map[thr_id];

	if (!init[dev])
		return;

	cudaSetDevice(dev);
	cudaDeviceSynchronize();

	if (d_V[dev])   cudaFree(d_V[dev]);
	if (d_B[dev])   cudaFree(d_B[dev]);
	if (d_X[dev])   cudaFree(d_X[dev]);
	if (d_S[dev])   cudaFree(d_S[dev]);
	if (d_res[dev]) cudaFree(d_res[dev]);

	d_V[dev] = NULL; d_B[dev] = NULL; d_X[dev] = NULL;
	d_S[dev] = NULL; d_res[dev] = NULL;

	yp_instances[dev] = 0;
	yp_sglobal[dev]   = false;
	yp_head[dev]      = YP_HEAD_SHA256;
	yp_diff_sig[dev]  = 0;    /* so the `-D` differential re-runs on the new params */

	init[dev] = false;
}
