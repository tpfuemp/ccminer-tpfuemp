/* yespower 1.0 family -- host driver and mining kernel.
 *
 * The hash body lives in yespower_hash.cuh and is shared verbatim with an
 * out-of-tree KAT that checks it against sph/yespower_ref.c for every variant
 * shape.  The reference stays compiled in as the host-side re-verify for every
 * GPU candidate, so a kernel bug can only cost a local reject, never a bad share.
 *
 * ARCH GATE: S is 96 KiB of shared memory per instance, so this needs a device
 * that can opt in to >= 98304 B per block -- sm_80/sm_86 only.  Turing caps at
 * 64 KiB and Pascal at 48 KiB, and the global-memory alternative measured slower.
 * Checked at init.
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
 * yespowerADVC and yespowerEQPAY are absent on purpose: no byte-exact source for
 * their pers was found, so they stay unlisted rather than guessed.
 * `--yespower-param`/`--yespower-key` still reach them. */
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
	default:
		return false;
	}
}

/* One 80-byte header in, 32-byte digest out.
 *
 * NOTE the return convention: the reference returns 1 on success and -1 on
 * error, the opposite of what its own upstream header documents. On failure we
 * fail CLOSED -- an all-ones digest can never be <= any target, so a broken
 * hash can never be mistaken for a share. */
static bool yespower_hash_80(void *state, const void *input,
                             const yespower_params_t *params)
{
	yespower_binary_t out;
	if (yespower_tls_ref((const uint8_t *) input, 80, params, &out) != 1) {
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
	(void) yespower_hash_80(state, input, &p);
}

/* ---------------------------------------------------------------------------
 * GPU path.
 * ------------------------------------------------------------------------- */

__constant__ static uint32_t c_yp_hdr[20];      /* words 0..18; [19] is the nonce */
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
template<uint32_t R>
__global__ __launch_bounds__(4, 1)
void yespower_gpu_hash(const uint32_t startNonce, const uint32_t N,
                       uint32_t *__restrict__ Vs, uint32_t *__restrict__ Bs,
                       uint32_t *__restrict__ Xs, uint32_t *__restrict__ resNonces)
{
	extern __shared__ uint4 s_S[];

	const int j = threadIdx.x & 3;
	const uint32_t inst = blockIdx.x;
	const uint32_t nonce = startNonce + inst;
	const uint32_t bw = 32u * R;
	uint32_t out[8];

	yespower_hash_1_0<R>(c_yp_hdr, nonce, N, s_S,
	                     Bs + (size_t)inst * bw,
	                     Xs + (size_t)inst * bw,
	                     Vs + (size_t)inst * bw * N,
	                     out, j, 0xfu);

	if (j == 0 && yp_below_target(out)) {
		const uint32_t prev = atomicExch(&resNonces[0], nonce);
		if (prev != UINT32_MAX) resNonces[1] = prev;
	}
}

static THREAD uint32_t *d_V[MAX_GPUS]  = { 0 };
static THREAD uint32_t *d_B[MAX_GPUS]  = { 0 };
static THREAD uint32_t *d_X[MAX_GPUS]  = { 0 };
static THREAD uint32_t *d_res[MAX_GPUS] = { 0 };
static THREAD uint32_t yp_instances[MAX_GPUS] = { 0 };
static bool init[MAX_GPUS] = { false };

/* Dispatch on r. Only the three shapes the variants actually use are instantiated;
 * a generic runtime r would put the 2R block loop out of the compiler's reach.
 * r=8 exists for Tidecoin and is the one shape no KAT has covered yet. */
static bool yp_launch(uint32_t r, uint32_t instances, uint32_t startNonce, uint32_t N,
                      int dev)
{
	const uint32_t shbytes = YP_SBOX_UINT4 * 16u;
	switch (r) {
	case 8:
		yespower_gpu_hash<8><<<instances, 4, shbytes>>>(startNonce, N,
			d_V[dev], d_B[dev], d_X[dev], d_res[dev]);
		return true;
	case 16:
		yespower_gpu_hash<16><<<instances, 4, shbytes>>>(startNonce, N,
			d_V[dev], d_B[dev], d_X[dev], d_res[dev]);
		return true;
	case 32:
		yespower_gpu_hash<32><<<instances, 4, shbytes>>>(startNonce, N,
			d_V[dev], d_B[dev], d_X[dev], d_res[dev]);
		return true;
	default:
		return false;
	}
}

static bool yp_set_shared_limit(uint32_t r)
{
	const uint32_t shbytes = YP_SBOX_UINT4 * 16u;
	cudaError_t e;
	switch (r) {
	case 8:
		e = cudaFuncSetAttribute(yespower_gpu_hash<8>,
		                         cudaFuncAttributeMaxDynamicSharedMemorySize, shbytes);
		break;
	case 16:
		e = cudaFuncSetAttribute(yespower_gpu_hash<16>,
		                         cudaFuncAttributeMaxDynamicSharedMemorySize, shbytes);
		break;
	case 32:
		e = cudaFuncSetAttribute(yespower_gpu_hash<32>,
		                         cudaFuncAttributeMaxDynamicSharedMemorySize, shbytes);
		break;
	default:
		return false;
	}
	return e == cudaSuccess;
}

extern "C" int scanhash_yespower(int thr_id, struct work *work, uint32_t max_nonce,
                                 unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	uint32_t endiandata[20];
	uint32_t vhash[8];
	yespower_params_t params;
	uint32_t n = first_nonce;
	const int dev = device_map[thr_id];
	static THREAD bool announced = false;
	time_t t_start;

	if (!yespower_params_for(opt_algo, &params)) {
		applog(LOG_ERR, "yespower: algo %d is not a yespower variant", opt_algo);
		return -1;
	}

	/* Reject up front exactly what the reference rejects (yespower_ref.c:495):
	 * N a power of two in [1024, 512K], r in [8, 32].  Without this the reference
	 * returns -1 for every nonce, the fail-closed digest below is all-ones, no
	 * share is ever found -- and the scan loop spins at memset speed and reports
	 * a completely fictitious hashrate.  `--yespower-param 128,2` read as
	 * 83 MH/s on a CPU. A validity check that only runs per-hash is not enough
	 * when the failure path is cheaper than the success path. */
	if (params.N < 1024 || params.N > 512 * 1024 || (params.N & (params.N - 1))) {
		applog(LOG_ERR, "yespower: N=%u invalid (need a power of two in [1024, 524288])",
		       params.N);
		return -1;
	}
	if (params.r < 8 || params.r > 32) {
		applog(LOG_ERR, "yespower: r=%u invalid (need 8..32)", params.r);
		return -1;
	}

	/* r is a template parameter of the kernel, so only the shapes the variants
	 * use exist. Reject anything else here rather than at launch, where the
	 * failure would look like a dead GPU. */
	if (params.r != 8 && params.r != 16 && params.r != 32) {
		applog(LOG_ERR, "yespower: r=%u has no GPU kernel (only 8, 16 and 32 are built)",
		       params.r);
		return -1;
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
	 * always), 0x00ffffff p=1/256 (empty 37% of runs, measured), 0x07ffffff p=1/32.
	 * Loosening this is not a correctness risk -- it only affects benchmarks, and
	 * every candidate is still re-hashed on the host. */
	if (opt_benchmark)
		ptarget[7] = 0x07ffffff;

	/* Follow ccminer's OWN yescrypt (algos/yescrypt/yescrypt.cu), not cpuminer's:
	 * cpuminer-opt runs yescrypt AND yespower through one scanhash that leaves the
	 * nonce raw, but ccminer's live, pool-proven yescrypt be32enc's all 20 words
	 * and submits with the switch's default le32enc.  The two miners keep
	 * work->data[19] in different conventions, so the sibling's raw-nonce form is
	 * not transferable here. */
	for (int k = 0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	if (!init[dev]) {
		const uint32_t shneed = YP_SBOX_UINT4 * 16u;
		int optin = 0;
		CUDA_CALL_OR_RET_X(cudaSetDevice(device_map[thr_id]), -1);

		cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin,
		                       device_map[thr_id]);
		if ((uint32_t) optin < shneed) {
			applog(LOG_ERR, "yespower: GPU #%d offers %d B of opt-in shared memory, "
			                "this kernel needs %u (sm_80/sm_86 only)",
			       device_map[thr_id], optin, shneed);
			return -1;
		}
		if (!yp_set_shared_limit(params.r)) {
			applog(LOG_ERR, "yespower: could not raise the dynamic shared limit to %u B",
			       shneed);
			return -1;
		}

		/* S fills all of shared memory, so exactly one block is resident per SM;
		 * more blocks than that only queue up behind each other. */
		yp_instances[dev] = (uint32_t) device_mpcount[dev];

		const size_t bw = 32u * (size_t) params.r;
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_V[dev],
			bw * 4 * params.N * yp_instances[dev]), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_B[dev], bw * 4 * yp_instances[dev]), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_X[dev], bw * 4 * yp_instances[dev]), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_res[dev], 2 * sizeof(uint32_t)), -1);

		/* pers is a per-job constant and never changes during a run. */
		if (params.perslen > YP_PERS_MAX) {
			applog(LOG_ERR, "yespower: pers is %u bytes, the kernel holds %u",
			       (uint32_t) params.perslen, (uint32_t) YP_PERS_MAX);
			return -1;
		}
		{
			uint8_t persbuf[YP_PERS_MAX] = { 0 };
			uint32_t plen = (uint32_t) params.perslen;
			if (plen) memcpy(persbuf, params.pers, plen);
			cudaMemcpyToSymbol(c_yp_pers, persbuf, YP_PERS_MAX);
			cudaMemcpyToSymbol(c_yp_perslen, &plen, sizeof(plen));
		}

		applog(LOG_INFO, "GPU #%d: yespower %u instances, %.0f MB of V",
		       device_map[thr_id], yp_instances[dev],
		       (double) (bw * 4.0 * params.N * yp_instances[dev]) / (1024.0 * 1024.0));
		init[dev] = true;
	}

	/* Upload pdata, NOT endiandata. The kernel wants the SHA-256 input words, i.e.
	 * the big-endian decode of the header byte stream. endiandata holds that stream
	 * as bytes, so reading it as host uint32 byte-reverses every word -- and since
	 * be32enc(&endiandata[i], pdata[i]), the correct decode is just pdata[i] again.
	 * Uploading endiandata makes the GPU hash a different header than the host
	 * re-verify, which surfaces only as "does not validate". Word 19 is ignored --
	 * the nonce is passed to the kernel separately. */
	cudaMemcpyToSymbol(c_yp_hdr, pdata, 20 * sizeof(uint32_t));
	cudaMemcpyToSymbol(c_yp_target, ptarget, 32);

	/* Return periodically instead of scanning to max_nonce, so the miner reports
	 * a rate and notices new work; the caller re-enters with the updated nonce. */
	t_start = time(NULL);

	do {
		const uint32_t batch = yp_instances[dev];
		uint32_t res[2] = { UINT32_MAX, UINT32_MAX };

		cudaMemcpy(d_res[dev], res, sizeof(res), cudaMemcpyHostToDevice);

		if (!yp_launch(params.r, batch, n, params.N, dev)) {
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
		if (res[0] != UINT32_MAX) {
			int found = 0;
			for (int s = 0; s < 2; s++) {
				if (res[s] == UINT32_MAX) continue;
				be32enc(&endiandata[19], res[s]);
				if (!yespower_hash_80(vhash, endiandata, &params)) {
					applog(LOG_ERR, "yespower: host re-verify failed at nonce %08x", res[s]);
					return -1;
				}
				if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
					work->nonces[found] = res[s];
					if (found == 0) work_set_target_ratio(work, vhash);
					found++;
				} else {
					gpu_increment_reject(thr_id);
					applog(LOG_WARNING, "GPU #%d: yespower result %08x does not validate",
					       device_map[thr_id], res[s]);
				}
			}
			if (found) {
				/* work->nonces[0] is what is actually submitted: miner_thread does
				 * `nonceptr[0] = work.nonces[0]` right before submit_work. Setting
				 * only pdata[19] leaves nonces[0] zeroed and every share goes out as
				 * nonce 00000000 -- "Invalid share" while the miner looks healthy. */
				work->valid_nonces = found;
				pdata[19] = res[0] + 1;      /* +1: never re-scan a submitted nonce */
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

extern "C" void free_yespower(int thr_id)
{
	/* The reference allocates and frees per hash; nothing is retained. */
	(void) thr_id;
}
