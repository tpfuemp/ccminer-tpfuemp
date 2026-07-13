#include "x11/cuda_x11.h"

extern void x13_hamsi512_cpu_init(int thr_id, uint32_t threads);
extern void x13_hamsi512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void x13_fugue512_cpu_init(int thr_id, uint32_t threads);
extern void x13_fugue512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);
extern void x13_fugue512_cpu_free(int thr_id);

extern void x14_shabal512_cpu_init(int thr_id, uint32_t threads);
extern void x14_shabal512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void x15_whirlpool_cpu_init(int thr_id, uint32_t threads, int flag);
extern void x15_whirlpool_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);
extern void x15_whirlpool_cpu_free(int thr_id);

extern void x17_sha512_cpu_init(int thr_id, uint32_t threads);
extern void x17_sha512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_hash);

extern void x17_haval256_cpu_init(int thr_id, uint32_t threads);
extern void x17_haval256_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_hash, const int outlen);

void quark_blake512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_nonceVector, uint32_t *d_outputHash, int order);

extern void tiger192_cpu_hash_64(int thr_id, int threads, int zero_pad_64, uint32_t *d_hash);

/* ---- stage-launcher name bridge --------------------------------------------
 * The 64-byte stage launchers still carry their originating family's prefix
 * (quark_/x11_/x13_/x14_/x15_/x17_) because the TUs that define them have not
 * moved out of their legacy folders yet. Migrated x16 sources call the stages
 * by their bare <prim>512 names; each alias below forwards the bare name to the
 * current real symbol. When a family migrates its launcher is renamed to the
 * bare form and the matching line here is deleted, at which point the prefixed
 * name has no remaining users and can be removed. The _cpu_ infix keeps these
 * host launchers distinct from the register-resident device primitives
 * (blake512_hash_64, ...) and the already-bare 80-byte launchers
 * (keccak512_cuda_hash_80, ...). */
#define blake512_cpu_init        quark_blake512_cpu_init
#define blake512_cpu_setBlock_80 quark_blake512_cpu_setBlock_80
#define blake512_cpu_hash_80     quark_blake512_cpu_hash_80
#define blake512_cpu_hash_64     quark_blake512_cpu_hash_64
#define blake512_cpu_free        quark_blake512_cpu_free
#define bmw512_cpu_init          quark_bmw512_cpu_init
#define bmw512_cpu_setBlock_80   quark_bmw512_cpu_setBlock_80
#define bmw512_cpu_hash_80       quark_bmw512_cpu_hash_80
#define bmw512_cpu_hash_64       quark_bmw512_cpu_hash_64
#define groestl512_cpu_init      quark_groestl512_cpu_init
#define groestl512_cpu_hash_64   quark_groestl512_cpu_hash_64
#define groestl512_cpu_free      quark_groestl512_cpu_free
#define skein512_cpu_init        quark_skein512_cpu_init
#define skein512_cpu_hash_64     quark_skein512_cpu_hash_64
#define jh512_cpu_init           quark_jh512_cpu_init
#define jh512_cpu_hash_64        quark_jh512_cpu_hash_64
#define keccak512_cpu_init       quark_keccak512_cpu_init
#define keccak512_cpu_hash_64    quark_keccak512_cpu_hash_64
#define luffa512_cpu_init        x11_luffa512_cpu_init
#define luffa512_cpu_hash_64     x11_luffa512_cpu_hash_64
#define cubehash512_cpu_init     x11_cubehash512_cpu_init
#define cubehash512_cpu_hash_64  x11_cubehash512_cpu_hash_64
#define shavite512_cpu_init      x11_shavite512_cpu_init
#define shavite512_cpu_hash_64   x11_shavite512_cpu_hash_64
#define simd512_cpu_init         x11_simd512_cpu_init
#define simd512_cpu_hash_64      x11_simd512_cpu_hash_64
#define simd512_cpu_free         x11_simd512_cpu_free
/* echo: the bare name is the optimised alexis 64-byte launcher
 * (cuda_x16_echo512_64.cu, real symbol below); the tpruvot compat variant used
 * only on arch < 500 (below the sm_61 build floor) is demoted to *_compat */
#define echo512_cpu_init_compat  x11_echo512_cpu_init
#define echo512_cpu_hash_64_compat x11_echo512_cpu_hash_64
#define hamsi512_cpu_init        x13_hamsi512_cpu_init
#define hamsi512_cpu_hash_64     x13_hamsi512_cpu_hash_64
#define fugue512_cpu_init        x13_fugue512_cpu_init
#define fugue512_cpu_hash_64     x13_fugue512_cpu_hash_64
#define fugue512_cpu_free        x13_fugue512_cpu_free
#define shabal512_cpu_init       x14_shabal512_cpu_init
#define shabal512_cpu_hash_64    x14_shabal512_cpu_hash_64
#define whirlpool512_cpu_init    x15_whirlpool_cpu_init
#define whirlpool512_cpu_hash_64 x15_whirlpool_cpu_hash_64
#define whirlpool512_cpu_free    x15_whirlpool_cpu_free
#define sha512_cpu_init          x17_sha512_cpu_init
#define sha512_cpu_hash_64       x17_sha512_cpu_hash_64

/* fused multi-stage runs (cuda_x16_fused.cu); stage ids = enum Algo, 16 = tiger192 */
void x16_fused_setOrder(const uint8_t *ids, int count);
void x16_fused_cpu_hash_64(int thr_id, uint32_t threads, int start, int len, int has_tiger, uint32_t *d_hash);
bool x16_fused_device_selftest(int thr_id);

/* stages whose device-library primitive runs register-resident (no shared
 * table fill, no quad-lane interface, no multi-kernel pipeline) — the set
 * the fused kernel can chain; [16] = tiger192 */
static const bool x16_fusible[17] = {
	true,  true,  false, true,  true,  true,  true,  true,
	false, false, false, true,  false, true,  false, true,
	true
};

// ---- optimised but non compatible kernels

/* optimised alexis 64-byte echo (cuda_x16_echo512_64.cu). Bare name is
 * canonical; x16_echo512_cpu_hash_64 is a legacy forwarder kept until the other
 * consumers (x17, skydoge, x21s, ghostrider) migrate to the bare name. */
void echo512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
void x16_echo512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);

// ---- 80 bytes kernels

void quark_bmw512_cpu_setBlock_80(void *pdata);
void quark_bmw512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_hash, int order);

void groestl512_setBlock_80(int thr_id, uint32_t *endiandata);
void groestl512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);

void skein512_cpu_setBlock_80(void *pdata);
void skein512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_hash, int swap);

void qubit_luffa512_cpu_init(int thr_id, uint32_t threads);
void qubit_luffa512_cpu_setBlock_80(void *pdata);
void qubit_luffa512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_hash, int order);

void jh512_setBlock_80(int thr_id, uint32_t *endiandata);
void jh512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);

void keccak512_setBlock_80(int thr_id, uint32_t *endiandata);
void keccak512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);

void cubehash512_setBlock_80(int thr_id, uint32_t* endiandata);
void cubehash512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);

/* generic 80-byte shavite launcher (cuda_x16_shavite512.cu); x16_shavite512_*
 * are legacy forwarders kept until ghostrider/x21s migrate to the bare names */
void shavite512_setBlock_80(void *pdata);
void shavite512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_hash, int order);
void x16_shavite512_setBlock_80(void *pdata);
void x16_shavite512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_hash, int order);


/* generic 80-byte shabal launcher (cuda_x16_shabal512.cu); x16_shabal512_* are
 * legacy forwarders kept until ghostrider/x21s migrate to the bare names */
void shabal512_setBlock_80(void *pdata);
void shabal512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);
void x16_shabal512_setBlock_80(void *pdata);
void x16_shabal512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);

/* generic 80-byte launchers (cuda_x16_*.cu); x16_* are legacy forwarders kept
 * until ghostrider/x21s migrate to the bare names */
void simd512_setBlock_80(void *pdata);
void simd512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);
void x16_simd512_setBlock_80(void *pdata);
void x16_simd512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);

void echo512_cuda_init(int thr_id, const uint32_t threads);
void echo512_setBlock_80(void *pdata);
void echo512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);
void x16_echo512_cuda_init(int thr_id, const uint32_t threads);
void x16_echo512_setBlock_80(void *pdata);
void x16_echo512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);

void x16_hamsi512_setBlock_80(void *pdata);
void x16_hamsi512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);

/* fugue: setBlock_80/cuda_hash_80 de-branded; cpu_init/cpu_free stay x16_ (bare
 * fugue512_cpu_init/free are the 64-byte x13 fugue bridge above) */
void fugue512_setBlock_80(void *pdata);
void fugue512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);
void x16_fugue512_cpu_init(int thr_id, uint32_t threads);
void x16_fugue512_cpu_free(int thr_id);
void x16_fugue512_setBlock_80(void *pdata);
void x16_fugue512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);

void x16_whirlpool512_init(int thr_id, uint32_t threads);
void x16_whirlpool512_setBlock_80(void* endiandata);
void x16_whirlpool512_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);

void x16_sha512_setBlock_80(void *pdata);
void x16_sha512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);

void tiger192_setBlock_80(void *pdata);
void tiger192_cpu_hash_80(int thr_id, int threads, uint32_t startNonce, uint32_t *d_hash);
