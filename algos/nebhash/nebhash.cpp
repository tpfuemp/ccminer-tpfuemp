/* Host glue for nebhash v2 (NEB mainnet, KEP1) mining.  The consensus code and
 * the CUDA scan are the Apache-2.0 reference from SilasEigen/neb, see
 * LICENSE.NEB.  This file owns the dataset, the per-job models and one scan
 * handle per (mining thread, dataset slot). */
#include "algos/nebhash/nebhash.h"

#include "miner.h"
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
/* The copied reference files are compiled as C; their public headers do not
 * all carry extern "C" guards, so include them under C linkage here. */
extern "C" {
#include "algos/nebhash/shares.h"
#include "algos/nebhash/chacha8.h"
#include "algos/nebhash/nebhash2.h"
#include "algos/nebhash/nebhash2_table_cuda.h"
}
#include "cuda/blake3_device.cuh"
#include "cuda/selftest_gate.cuh"

#include <cuda_runtime.h>
#ifndef _WIN32
#include <unistd.h>   /* usleep; Windows gets it from compat via miner.h */
#endif

#include <vector>
#include <memory>
#include <mutex>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <thread>

#define NEB_MAX_SLOTS   16
/* Nonces per scan_run call. Each call pays a few uploads and a synchronisation; a
 * 64k window instead of 8k measured +2-3 % on Linux and +11-12 % on Windows (WDDM).
 * A call can report at most MAX_NONCES shares, so the window adapts to the target:
 * about targetdiff / 8 attempts (an expected 1/8 share per call), clamped to
 * [NEB_SCAN_MIN, NEB_SCAN_CAP]; low vardiff starts fall back to the 8k window. */
#define NEB_SCAN_MIN    8192
#define NEB_SCAN_CAP    65536
#define NEB_MAX_HITS    (MAX_NONCES + 1)
/* In-miner self-test: CPU tickets of this many nonces of a fixed header. */
#define NEB_ST_N        32
/* --benchmark: one share per 2^22 attempts (a few per second), each re-verified on the CPU. */
#define NEB_BENCH_BITS  22

namespace {

struct Dataset {
	uint32_t n = 0;
	uint32_t nin = 0;
	int arch = 0;
	std::vector<uint8_t> values;   /* n * nin, values 0..7 */
	std::vector<uint8_t> labels;   /* n, values 0..1 */
};

struct Model {
	uint8_t hash[32];
	int slot = 0;
	int arch = 0;
	std::vector<int8_t> weights;
	std::shared_ptr<neb2_model> ref;   /* 130 KB; keep off the stack */
};

struct ScanHandle {
	neb2t_scan *h = nullptr;
	int arch = 0;
	uint8_t model_hash[32];
	int logged_threads = 0;  /* launch shape last reported for this model (-D) */
};

std::vector<Dataset> g_datasets;
/* Mining threads read the current model while the stratum thread installs the
 * next one. Publish whole shared_ptrs so an in-flight verification keeps its
 * model alive instead of racing with a vector reallocation. */
std::mutex g_state_lock;
std::vector<std::shared_ptr<Model>> g_models;
uint8_t g_dataset_hash[32] = { 0 };
ScanHandle *g_scans[MAX_GPUS][NEB_MAX_SLOTS];
char g_dataset_path[512] = { 0 };

/* --benchmark: a synthetic job per mining thread (seeded model of slot 0, own header and target)
 * with a private cursor, because the benchmark loop rewrites the work and its nonce on every call. */
struct BenchJob {
	bool ready = false;
	uint8_t header[80];
	uint32_t cursor = 0;
	uint64_t hits = 0, verified = 0;
	time_t last_log = 0;
};
BenchJob g_bench[MAX_GPUS];
uint8_t g_bench_model[32];
bool g_bench_ready = false;

uint32_t read_le32(const uint8_t *p)
{
	return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
	       ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

void write_le32(uint8_t *p, uint32_t v)
{
	p[0] = (uint8_t)v;
	p[1] = (uint8_t)(v >> 8);
	p[2] = (uint8_t)(v >> 16);
	p[3] = (uint8_t)(v >> 24);
}

void append_bytes(std::vector<uint8_t> &v, const void *p, size_t n)
{
	const uint8_t *b = (const uint8_t *)p;
	if (n)
		v.insert(v.end(), b, b + n);
}

/* H(tag, parts...) is BLAKE3 over the tag bytes followed by the parts, with no
 * separators (spec/node.md section 3). */
void hash_parts(const char *tag, size_t taglen,
                const uint8_t *a, size_t alen,
                const uint8_t *b, size_t blen,
                const uint8_t *c, size_t clen, uint8_t out[32])
{
	std::vector<uint8_t> buf;
	buf.reserve(taglen + alen + blen + clen);
	append_bytes(buf, tag, taglen);
	append_bytes(buf, a, alen);
	append_bytes(buf, b, blen);
	append_bytes(buf, c, clen);
	blake3_hash(buf.data(), buf.size(), NULL, out);
}

int dataset_get(void *ctx, uint32_t index, uint8_t *x)
{
	Dataset *ds = (Dataset *)ctx;
	if (index >= ds->n)
		return 0;
	memcpy(x, ds->values.data() + (size_t)index * ds->nin, ds->nin);
	return ds->labels[index];
}

std::shared_ptr<Model> find_model(const uint8_t hash[32])
{
	std::lock_guard<std::mutex> lock(g_state_lock);
	for (size_t i = 0; i < g_models.size(); i++)
		if (!memcmp(g_models[i]->hash, hash, 32))
			return g_models[i];
	return std::shared_ptr<Model>();
}

/* Compare two 256-bit little-endian integers: return true when a <= b. */
bool le_le(const uint8_t a[32], const uint8_t b[32])
{
	for (int i = 31; i >= 0; i--) {
		if (a[i] != b[i])
			return a[i] < b[i];
	}
	return true;
}

/* Re-check a GPU hit with the portable reference before submitting it. */
int verify_ticket(const Model &m, int slot, const uint8_t header[80], uint8_t ticket[32])
{
	if (slot < 0 || slot >= (int)g_datasets.size() || !m.ref)
		return -1;
	Dataset &ds = g_datasets[slot];
	const neb2_arch *a = neb2_arch_get(ds.arch);
	if (!a || a->id != m.arch)
		return -1;

	uint8_t key[32];
	blake3_hash(header, 80, NULL, key);

	neb2_dataset nd;
	nd.n = ds.n;
	nd.get = dataset_get;
	nd.ctx = &ds;

	std::vector<int32_t> grad((size_t)a->nweights);
	uint8_t digest[32];
	return neb2_share_eval(m.ref.get(), &nd, key, grad.data(), digest, ticket,
	                       NULL, NULL);
}

/* Little-endian 256-bit compare, strict. */
bool le_lt(const uint8_t a[32], const uint8_t b[32])
{
	return le_le(a, b) && memcmp(a, b, 32) != 0;
}

/* In-miner check of every table this thread uploads (V-01/V-02/V-12): the CPU reference
 * tickets of NEB_ST_N nonces of a fixed header, then the GPU hit set through the shipping
 * launcher at bound = the 2nd lowest ticket (exactly the two lowest nonces) and at that
 * ticket - 1 (exactly the lowest). A mismatch refuses to mine (selftest_gate); a failure to
 * run (CUDA resource, host memory) only warns. NEBHASH_SELFTEST_FAULT=1 gives the GPU the
 * 3rd lowest ticket for the first leg, =2 the 2nd lowest for the second leg: both must fail. */
bool scan_selftest(int thr_id, neb2t_scan *h, const Model &m, int slot)
{
	static bool logged[MAX_GPUS];
	const char *fenv = getenv("NEBHASH_SELFTEST_FAULT");
	const int fault = fenv ? atoi(fenv) : 0;
	if (fault)
		gpulog(LOG_WARNING, thr_id, "nebhash: SELF-TEST FAULT INJECTION ACTIVE (%d) -- for testing only", fault);
	struct timeval t0, t1;
	gettimeofday(&t0, NULL);

	uint8_t header[80];
	for (int i = 0; i < 80; i++)
		header[i] = (uint8_t)(i * 37 + 11);
	const uint32_t start = 0x104d;   /* not aligned to anything */
	struct Tk { uint8_t t[32]; uint32_t nonce; } v[NEB_ST_N];
	std::vector<std::thread> th;
	int bad = 0;
	std::mutex bad_lock;
	unsigned nt = std::thread::hardware_concurrency();
	if (nt == 0) nt = 4;
	if (nt > NEB_ST_N) nt = NEB_ST_N;
	for (unsigned t = 0; t < nt; t++)
		th.emplace_back([&, t] {
			uint8_t hd[80];
			memcpy(hd, header, 80);
			for (unsigned i = t; i < NEB_ST_N; i += nt) {
				v[i].nonce = start + i;
				write_le32(hd + 76, v[i].nonce);
				if (verify_ticket(m, slot, hd, v[i].t) != 0) {
					std::lock_guard<std::mutex> lock(bad_lock);
					bad++;
				}
			}
		});
	for (auto &x : th) x.join();
	if (bad) {
		gpulog(LOG_WARNING, thr_id, "nebhash self-test could not run (CPU reference failed), continuing");
		return true;
	}
	std::sort(v, v + NEB_ST_N, [](const Tk &a, const Tk &b) { return le_lt(a.t, b.t); });

	uint8_t bound[32];
	uint32_t hits[NEB_MAX_HITS + 1], nh = 0;
	bool ran = true, leg_a = true, leg_b = true;
	/* every launch shape the scan may pick for this model (two on sm_6x), each pinned in turn */
	const int shapes = neb2t_scan_shapes(h);
	int bad_shape = -1;

	for (int sh = 0; sh < shapes && ran && leg_a && leg_b; sh++) {
		neb2t_scan_pin_shape(h, sh);
		memcpy(bound, v[fault == 1 ? 2 : 1].t, 32);
		if (neb2t_scan_run(h, header, start, NEB_ST_N, bound, hits, &nh) != 0)
			ran = leg_a = leg_b = selftest_cuda_fault();   /* false: neither leg ran */
		else
			leg_a = nh == 2 && ((hits[0] == v[0].nonce && hits[1] == v[1].nonce) ||
			                    (hits[0] == v[1].nonce && hits[1] == v[0].nonce));

		memcpy(bound, v[1].t, 32);
		if (fault != 2)
			for (int i = 0; i < 32 && bound[i]-- == 0; i++) {}   /* bound - 1 */
		if (ran) {
			if (neb2t_scan_run(h, header, start, NEB_ST_N, bound, hits, &nh) != 0)
				ran = leg_b = selftest_cuda_fault();
			else
				leg_b = nh == 1 && hits[0] == v[0].nonce;
		}
		if (!ran || !leg_a || !leg_b)
			bad_shape = sh;
	}
	neb2t_scan_pin_shape(h, -1);

	gettimeofday(&t1, NULL);
	const double ms = 1e3 * (t1.tv_sec - t0.tv_sec) + 1e-3 * (t1.tv_usec - t0.tv_usec);
	const bool passed = ran && leg_a && leg_b;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "nebhash self-test FAILED (slot %d, launch shape %d of %d: GPU hits at rank 2 %s, at rank 2 - 1 %s)",
			slot, bad_shape + 1, shapes,
			leg_a ? "ok" : (ran ? "WRONG" : "not run"), leg_b ? "ok" : (ran ? "WRONG" : "not run"));
	else if (!logged[thr_id] || opt_debug) {
		gpulog(logged[thr_id] ? LOG_DEBUG : LOG_INFO, thr_id,
			"nebhash self-test passed (slot %d, %d launch shape%s, GPU == CPU hit sets at rank 2 and rank 2 - 1 of %d nonces, %.0f ms)",
			slot, shapes, shapes > 1 ? "s" : "", NEB_ST_N, ms);
		logged[thr_id] = true;
	}
	return selftest_gate(thr_id, "nebhash", passed);
}

bool ensure_scan(int thr_id, int slot, const uint8_t model_hash[32])
{
	if (thr_id < 0 || thr_id >= MAX_GPUS || slot < 0 || slot >= NEB_MAX_SLOTS)
		return false;
	int dev = device_map[thr_id];
	if (dev < 0 || slot >= (int)g_datasets.size())
		return false;

	std::shared_ptr<Model> m = find_model(model_hash);
	if (!m)
		return false;
	Dataset &ds = g_datasets[slot];
	const neb2_arch *a = neb2_arch_get(ds.arch);
	if (!a || a->id != m->arch)
		return false;

	ScanHandle *&ref = g_scans[thr_id][slot];
	if (!ref)
		ref = new ScanHandle();

	if (!ref->h) {
		int rc = neb2t_scan_open(dev, a->id, ds.values.data(), ds.labels.data(),
		                        ds.n, m->weights.data(), NEB_SCAN_CAP,
		                        NEB_MAX_HITS, &ref->h);
		if (rc != 0) {
			applog(LOG_ERR, "nebhash: neb2t_scan_open failed on GPU %d (%d)", dev, rc);
			return false;
		}
		ref->arch = a->id;
		/* opening the handle already built and uploaded this model (upstream 43ff9d6) */
		memcpy(ref->model_hash, model_hash, 32);
		scan_selftest(thr_id, ref->h, *m, slot);
	} else if (ref->arch != a->id) {
		neb2t_scan_close(ref->h);
		ref->h = NULL;
		return ensure_scan(thr_id, slot, model_hash);
	}

	if (memcmp(ref->model_hash, model_hash, 32)) {
		int rc = neb2t_scan_set_model(ref->h, m->weights.data());
		if (rc != 0) {
			applog(LOG_ERR, "nebhash: model upload failed on GPU %d (%d)", dev, rc);
			return false;
		}
		memcpy(ref->model_hash, model_hash, 32);
		ref->logged_threads = 0;
		scan_selftest(thr_id, ref->h, *m, slot);
	}
	return true;
}

void close_all_scans()
{
	for (int t = 0; t < MAX_GPUS; t++) {
		for (int s = 0; s < NEB_MAX_SLOTS; s++) {
			if (g_scans[t][s]) {
				neb2t_scan_close(g_scans[t][s]->h);
				delete g_scans[t][s];
				g_scans[t][s] = NULL;
			}
		}
	}
}

} /* namespace */

void nebhash_set_dataset_option(const char *path)
{
	if (!path)
		return;
	snprintf(g_dataset_path, sizeof(g_dataset_path), "%s", path);
}

const char *nebhash_dataset_option(void)
{
	return g_dataset_path[0] ? g_dataset_path : NULL;
}

int nebhash_dataset_loaded(void)
{
	return !g_datasets.empty();
}

void nebhash_dataset_hash(uint8_t out[32])
{
	memcpy(out, g_dataset_hash, 32);
}

int nebhash_load_dataset_file(const char *path)
{
	FILE *f = fopen(path, "rb");
	if (!f) {
		applog(LOG_ERR, "nebhash: cannot open dataset %s", path);
		return -1;
	}
	if (fseek(f, 0, SEEK_END) != 0) {
		fclose(f);
		return -1;
	}
	long filelen = ftell(f);
	if (filelen <= 8 || fseek(f, 0, SEEK_SET) != 0) {
		fclose(f);
		return -1;
	}
	std::vector<uint8_t> bytes((size_t)filelen);
	if (fread(bytes.data(), 1, bytes.size(), f) != bytes.size()) {
		fclose(f);
		applog(LOG_ERR, "nebhash: short read on dataset %s", path);
		return -1;
	}
	fclose(f);

	if (memcmp(bytes.data(), "NMDS", 4)) {
		applog(LOG_ERR, "nebhash: %s has no NMDS header", path);
		return -1;
	}
	uint32_t count = read_le32(bytes.data() + 4);
	if (!count || count > NEB_MAX_SLOTS) {
		applog(LOG_ERR, "nebhash: dataset count %u out of range", count);
		return -1;
	}

	size_t pos = 8;
	std::vector<Dataset> loaded;
	for (uint32_t i = 0; i < count; i++) {
		if (pos + 12 > bytes.size() || memcmp(bytes.data() + pos, "NMD2", 4)) {
			applog(LOG_ERR, "nebhash: dataset %u has no NMD2 header", i);
			return -1;
		}
		uint32_t arch_id = read_le32(bytes.data() + pos + 4);
		uint32_t n = read_le32(bytes.data() + pos + 8);
		const neb2_arch *a = neb2_arch_get((int)arch_id);
		if (!a || !n) {
			applog(LOG_ERR, "nebhash: dataset %u has unsupported arch %u", i, arch_id);
			return -1;
		}
		size_t need = (size_t)a->nin + 1;
		size_t total = 12 + (size_t)n * need;
		if (pos + total > bytes.size()) {
			applog(LOG_ERR, "nebhash: dataset %u is truncated", i);
			return -1;
		}

		Dataset ds;
		ds.arch = (int)arch_id;
		ds.nin = (uint32_t)a->nin;
		ds.n = n;
		ds.values.resize((size_t)n * a->nin);
		ds.labels.resize(n);
		for (uint32_t j = 0; j < n; j++) {
			const uint8_t *rec = bytes.data() + pos + 12 + (size_t)j * need;
			memcpy(ds.values.data() + (size_t)j * a->nin, rec, a->nin);
			ds.labels[j] = rec[a->nin];
			if (ds.labels[j] >= NEB2_NOUT) {
				applog(LOG_ERR, "nebhash: dataset %u sample %u has bad label", i, j);
				return -1;
			}
		}
		pos += total;
		loaded.push_back(std::move(ds));
	}
	if (pos != bytes.size()) {
		applog(LOG_ERR, "nebhash: trailing bytes in dataset file");
		return -1;
	}

	close_all_scans();
	g_models.clear();
	g_datasets = std::move(loaded);
	blake3_hash(bytes.data(), bytes.size(), NULL, g_dataset_hash);
	return 0;
}

int nebhash_have_model(const uint8_t hash[32])
{
	return find_model(hash) != NULL;
}

int nebhash_set_model(int slot, const uint8_t hash[32],
                      const uint8_t *packed, size_t packed_len)
{
	if (slot < 0 || slot >= (int)g_datasets.size())
		return -1;
	const Dataset &ds = g_datasets[slot];
	const neb2_arch *a = neb2_arch_get(ds.arch);
	if (!a || !packed || packed_len != (size_t)a->nweights / 2)
		return -1;

	Model m;
	memcpy(m.hash, hash, 32);
	m.slot = slot;
	m.arch = a->id;
	m.weights.resize((size_t)a->nweights);
	for (int i = 0; i < a->nweights; i++) {
		uint8_t nibble = (i & 1) ? (packed[i / 2] >> 4) : (packed[i / 2] & 15);
		m.weights[i] = (int8_t)((int8_t)(nibble << 4) >> 4);
	}

	m.ref.reset(new neb2_model);
	memset(m.ref.get(), 0, sizeof(neb2_model));
	m.ref->arch = a->id;
	memcpy(m.ref->w, m.weights.data(), (size_t)a->nweights);

	uint8_t wh[32], rh[32];
	neb2_model_hashes(m.ref.get(), wh, rh);
	if (memcmp(wh, hash, 32)) {
		applog(LOG_ERR, "nebhash: set_model weights_hash mismatch");
		return -1;
	}

	std::lock_guard<std::mutex> lock(g_state_lock);
	std::shared_ptr<Model> published = std::make_shared<Model>(std::move(m));
	for (size_t i = 0; i < g_models.size(); i++) {
		if (!memcmp(g_models[i]->hash, hash, 32)) {
			g_models[i] = published;
			return 0;
		}
	}
	/* Models arrive every block; only a few are needed for hit verification. */
	if (g_models.size() >= 4)
		g_models.erase(g_models.begin());
	g_models.push_back(published);
	return 0;
}

double nebhash_target_to_diff(const uint8_t le[32])
{
	/* Match yiimp's nebhash_le_to_diff(): 2^256 / (target + 1). The inverse
	 * nebhash_diff_to_target() there uses 2^256 / diff - 1, so dropping the
	 * +1 here would report a slightly different difficulty than the pool. */
	long double t = 0.0L;
	for (int i = 31; i >= 0; i--)
		t = t * 256.0L + le[i];
	return (double)(ldexpl(1.0L, 256) / (t + 1.0L));
}

int nebhash_build_header(uint32_t version, const uint8_t parent[32],
                         const uint8_t *cb_prefix, size_t cb_prefix_len,
                         const uint8_t *xn1, size_t xn1_len,
                         const uint8_t *xn2, size_t xn2_len,
                         const uint8_t *cb_suffix, size_t cb_suffix_len,
                         const uint8_t *branch, size_t branch_count,
                         uint32_t tx_count, const uint8_t shares_root[32],
                         uint32_t ntime, uint32_t bits, uint8_t header[80])
{
	std::vector<uint8_t> coinbase;
	coinbase.reserve(cb_prefix_len + xn1_len + xn2_len + cb_suffix_len);
	append_bytes(coinbase, cb_prefix, cb_prefix_len);
	append_bytes(coinbase, xn1, xn1_len);
	append_bytes(coinbase, xn2, xn2_len);
	append_bytes(coinbase, cb_suffix, cb_suffix_len);
	if (coinbase.empty())
		return -1;

	uint8_t coinhash[32], root[32], le[4];
	blake3_hash(coinbase.data(), coinbase.size(), NULL, coinhash);
	hash_parts("leaf", 4, coinhash, 32, NULL, 0, NULL, 0, root);
	for (size_t i = 0; i < branch_count; i++)
		hash_parts("node", 4, root, 32, branch + i * 32, 32, NULL, 0, root);
	write_le32(le, tx_count);
	hash_parts("body", 4, le, 4, root, 32, shares_root, 32, root);

	write_le32(header, version);
	memcpy(header + 4, parent, 32);
	memcpy(header + 36, root, 32);
	write_le32(header + 68, ntime);
	write_le32(header + 72, bits);
	write_le32(header + 76, 0);   /* nonce, rolled by the scan */
	return 0;
}

int nebhash_benchmark_init(void)
{
	if (g_datasets.empty())
		return -1;
	const neb2_arch *a = neb2_arch_get(g_datasets[0].arch);
	if (!a)
		return -1;
	static const uint8_t seed[33] = "ccminer-nebhash-benchmark-seed-0";
	Model m;
	m.slot = 0;
	m.arch = a->id;
	m.ref.reset(new neb2_model);
	memset(m.ref.get(), 0, sizeof(neb2_model));
	if (neb2_model_init(a->id, seed, m.ref.get()) != 0)
		return -1;
	m.weights.assign(m.ref->w, m.ref->w + a->nweights);
	uint8_t rh[32];
	neb2_model_hashes(m.ref.get(), m.hash, rh);
	memcpy(g_bench_model, m.hash, 32);
	std::lock_guard<std::mutex> lock(g_state_lock);
	g_models.clear();
	g_models.push_back(std::make_shared<Model>(std::move(m)));
	g_bench_ready = true;
	return 0;
}

/* --benchmark: the seeded model of slot 0 (an untrained model: its rate is not a live rate,
 * compare benchmarks only with each other), a per-thread header, one share per 2^NEB_BENCH_BITS
 * attempts. Every GPU hit is re-verified on the CPU exactly as in the pool path. */
static int bench_scan(int thr_id, struct work *work, unsigned long *hashes_done)
{
	BenchJob &b = g_bench[thr_id];
	if (!b.ready) {
		for (int i = 0; i < 80; i++)
			b.header[i] = (uint8_t)(i * 13 + 5 + thr_id);
		b.cursor = 0;
		b.ready = true;
	}
	if (!g_bench_ready || !ensure_scan(thr_id, 0, g_bench_model)) {
		usleep(1000 * 1000);
		return -1;
	}
	uint8_t target[32];
	memset(target, 0xff, 32);
	target[31] = target[30] = 0;
	target[29] = (uint8_t)((1u << (256 - NEB_BENCH_BITS - 232)) - 1);   /* 2^(256 - bits) - 1 */
	const uint32_t start = b.cursor, count = NEB_SCAN_CAP;
	uint32_t hits[NEB_MAX_HITS + 1], nhits = 0;
	int rc = neb2t_scan_run(g_scans[thr_id][0]->h, b.header, start, count, target, hits, &nhits);
	if (rc != 0) {
		applog(LOG_ERR, "nebhash: scan failed on GPU %d (%d)", device_map[thr_id], rc);
		usleep(1000 * 1000);
		return -1;
	}
	b.cursor = start + count;
	if (b.cursor < start) {   /* range exhausted: next header */
		b.header[0]++;
		b.cursor = 0;
	}
	*hashes_done = count;
	std::shared_ptr<Model> m = find_model(g_bench_model);
	uint8_t header[80];
	memcpy(header, b.header, 80);
	int found = 0;
	for (uint32_t i = 0; i < nhits && i < NEB_MAX_HITS; i++) {
		write_le32(header + 76, hits[i]);
		uint8_t ticket[32];
		b.hits++;
		if (!m || verify_ticket(*m, 0, header, ticket) != 0 || !le_le(ticket, target)) {
			gpulog(LOG_WARNING, thr_id, "nebhash benchmark: GPU hit %08x does not validate on the CPU", hits[i]);
			continue;
		}
		b.verified++;
		if (found < MAX_NONCES) {
			work->nonces[found] = hits[i];
			work->sharediff[found] = nebhash_target_to_diff(ticket);
			found++;
		}
	}
	if (time(NULL) - b.last_log >= 30 && b.hits) {
		gpulog(LOG_INFO, thr_id, "nebhash benchmark: %llu GPU hits, %llu CPU-verified",
			(unsigned long long)b.hits, (unsigned long long)b.verified);
		b.last_log = time(NULL);
	}
	work->valid_nonces = found;
	return found;
}

int nebhash_scanhash(int thr_id, struct work *work, uint32_t max_nonce,
                     unsigned long *hashes_done)
{
	*hashes_done = 0;
	work->valid_nonces = 0;
	if (opt_benchmark)
		return bench_scan(thr_id, work, hashes_done);
	if (!work->is_nebhash || !nebhash_dataset_loaded())
		return -1;

	/* miner_thread retries a -1 at once: back off so a missing model or a
	 * failing GPU does not turn into a busy loop flooding the log */
	int slot = (int)work->neb_slot;
	if (!ensure_scan(thr_id, slot, work->neb_model_hash)) {
		usleep(1000 * 1000);
		return -1;
	}

	/* max_nonce is the END of this thread's nonce slice (miner_thread splits
	 * [0, 2^32) between GPUs that hash the same header), not a count: scanning
	 * past it would duplicate the next GPU's nonces. */
	uint32_t start = work->data[19];
	if (start >= max_nonce)
		return 0;
	uint32_t window = NEB_SCAN_MIN;
	if (work->targetdiff / 8.0 > (double)NEB_SCAN_MIN)
		window = work->targetdiff / 8.0 >= (double)NEB_SCAN_CAP ? NEB_SCAN_CAP
		                                                       : (uint32_t)(work->targetdiff / 8.0);
	uint32_t count = max_nonce - start;
	if (count > window)
		count = window;

	uint8_t header[80];
	memcpy(header, work->neb_header, 80);
	uint32_t hits[NEB_MAX_HITS + 1];
	uint32_t nhits = 0;
	int rc = neb2t_scan_run(g_scans[thr_id][slot]->h, header, start, count,
	                       (const uint8_t *)work->target, hits, &nhits);
	if (rc != 0) {
		applog(LOG_ERR, "nebhash: scan failed on GPU %d (%d)", device_map[thr_id], rc);
		usleep(1000 * 1000);
		return -1;
	}

	if (opt_debug) {
		ScanHandle *sh = g_scans[thr_id][slot];
		const int nt = neb2t_scan_threads(sh->h);
		if (nt && nt != sh->logged_threads) {
			applog(LOG_DEBUG, "GPU #%d: nebhash slot %d launch shape %d threads per attempt", device_map[thr_id], slot, nt);
			sh->logged_threads = nt;
		}
	}

	*hashes_done = count;
	/* Leave the LAST scanned nonce, not the next one: on unchanged work the miner loop
	 * adds 1 before the next call (ccminer.cpp, nonceptr[0]++), so start + count here
	 * skipped one nonce per call. start + count - 1 < max_nonce, no wrap (count >= 1). */
	work->data[19] = start + count - 1;

	std::shared_ptr<Model> m = find_model(work->neb_model_hash);
	int found = 0;
	for (uint32_t i = 0; i < nhits && found < MAX_NONCES; i++) {
		uint32_t nonce = hits[i];
		write_le32(header + 76, nonce);
		uint8_t ticket[32];
		if (!m || verify_ticket(*m, slot, header, ticket) != 0)
			continue;
		if (!le_le(ticket, (const uint8_t *)work->target))
			continue;
		work->nonces[found] = nonce;
		bn_set_target_ratio(work, (uint32_t *)ticket, found);
		found++;
	}
	work->valid_nonces = found;
	return found;
}

void nebhash_free(int thr_id)
{
	if (thr_id < 0 || thr_id >= MAX_GPUS)
		return;
	for (int s = 0; s < NEB_MAX_SLOTS; s++) {
		if (g_scans[thr_id][s]) {
			neb2t_scan_close(g_scans[thr_id][s]->h);
			delete g_scans[thr_id][s];
			g_scans[thr_id][s] = NULL;
		}
	}
}
