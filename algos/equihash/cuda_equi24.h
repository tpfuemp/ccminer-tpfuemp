// equi24 - round-templated Equihash solver for the DIGITBITS=24 family
// (144/5 and 192/7). C entry points for ccminer; mirrors the tromp144_* API
// so the two solvers are swappable at the dispatch site.
#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Allocate a persistent solver context; NULL if the device cannot provide the
// arena (see equi24_arena_needed), in which case the caller must fall back.
// One solver, two instantiations keyed on WN.
void *equi24_144_init(void);
void  equi24_144_free(void *ctx);
void *equi24_192_init(void);
void  equi24_192_free(void *ctx);

// Device bytes the context requires: ~4.3 GiB, against the tromp path's
// ~2.5 GiB. Check before switching solvers on a small card.
size_t equi24_144_arena_needed(void);
size_t equi24_192_arena_needed(void);

// Solve one 140-byte header+nonce under `personal`. Emits each distinct,
// non-trivial proof (duplicate-leaf candidates are filtered) and returns the
// count emitted, or -1 on error.
//
// Proofs are NOT verified here; the caller re-verifies before submit, which is
// what keeps a solver bug to a local reject.
int equi24_144_solve(void *ctx, const char *headernonce, const char *personal,
                     void (*emit)(void *, const uint32_t *, uint32_t), void *ud);
int equi24_192_solve(void *ctx, const char *headernonce, const char *personal,
                     void (*emit)(void *, const uint32_t *, uint32_t), void *ud);

// Consensus re-verify for 192/7. tromp144_verify is compiled -DWN=144 and
// cannot check a 192/7 proof: its PROOFSIZE is 32 and its personalization
// carries the wrong (n,k). Pick by variant at the call site.
int equi24_192_verify(const char *headernonce, const char *personal,
                      const uint32_t *indices);

#ifdef __cplusplus
}
#endif
