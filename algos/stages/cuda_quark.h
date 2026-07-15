#include "cuda_helper.h"

/* commonly used cuda quark kernels prototypes */

extern void blake512_cpu_init(int thr_id, uint32_t threads);
extern void blake512_cpu_free(int thr_id);
extern void blake512_cpu_setBlock_80(int thr_id, uint32_t *pdata);
extern void blake512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash);
extern void blake512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void bmw512_cpu_init(int thr_id, uint32_t threads);
extern void bmw512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);
extern void bmw512_cpu_setBlock_80(void *pdata);
extern void bmw512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_hash, int order);

extern void groestl512_cpu_init(int thr_id, uint32_t threads);
extern void groestl512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);
extern void groestl512_cpu_free(int thr_id);

extern void skein512_cpu_init(int thr_id, uint32_t threads);
extern void skein512_cpu_hash_64(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void keccak512_cpu_init(int thr_id, uint32_t threads);
//extern void keccak512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

/* startNounce is only needed when d_nonceVector != NULL (quark branch vectors
 * store absolute nonces); it defaults to 0 for the many NULL-vector callers. */
extern void keccak512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_nonceVector, uint32_t *d_hash, uint32_t startNounce = 0);
extern void keccak512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t *d_nonceVector, uint32_t *d_hash, uint64_t target, uint32_t *d_resNonce);


extern void jh512_cpu_init(int thr_id, uint32_t threads);
extern void jh512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void quark_compactTest_cpu_init(int thr_id, uint32_t threads);
extern void quark_compactTest_cpu_free(int thr_id);
extern void quark_compactTest_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *inpHashes, uint32_t *d_validNonceTable,
											uint32_t *d_nonces1, uint32_t *nrm1, uint32_t *d_nonces2, uint32_t *nrm2, int order);
extern void quark_compactTest_single_false_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *inpHashes, uint32_t *d_validNonceTable,
											uint32_t *d_nonces1, uint32_t *nrm1, int order);

extern uint32_t cuda_check_hash_branch(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_inputHash, int order);

/* Legacy-name compatibility: the core primitive launchers were de-branded to
 * their bare <prim>512_cpu_* names (real symbols in algos/stages/). The
 * quark_<prim>512_cpu_* thin forwarders below (defined in the same TUs) remain
 * for the not-yet-migrated callers (x13/x15/x17/tribus/JHA + standalone
 * zr5/skein/skein2/pentablake/bastion/Algo256-bmw512 + evohash/ghostrider/x21s);
 * each drops out as its family switches to the bare name. (compactTest stays
 * quark_-named.) Real forwarder symbols so callers link whether or not they
 * include this header. */
extern void quark_blake512_cpu_init(int thr_id, uint32_t threads);
extern void quark_blake512_cpu_free(int thr_id);
extern void quark_blake512_cpu_setBlock_80(int thr_id, uint32_t *pdata);
extern void quark_blake512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash);
extern void quark_blake512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);
extern void quark_bmw512_cpu_init(int thr_id, uint32_t threads);
extern void quark_bmw512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);
extern void quark_bmw512_cpu_setBlock_80(void *pdata);
extern void quark_bmw512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_hash, int order);
extern void quark_groestl512_cpu_init(int thr_id, uint32_t threads);
extern void quark_groestl512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);
extern void quark_groestl512_cpu_free(int thr_id);
extern void quark_skein512_cpu_init(int thr_id, uint32_t threads);
extern void quark_skein512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);
extern void quark_keccak512_cpu_init(int thr_id, uint32_t threads);
extern void quark_keccak512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_nonceVector, uint32_t *d_hash, uint32_t startNounce = 0);
extern void quark_jh512_cpu_init(int thr_id, uint32_t threads);
extern void quark_jh512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);
