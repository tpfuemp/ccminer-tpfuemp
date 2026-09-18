// WS3 — Equihash 144/5 solver (tromp/equihash) C entry points for ccminer.
// Kept separate from djeZo's cuda_equi.cu (200/9). See cuda_equi_tromp.cu.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Callback invoked once per (de-duplicated) solution found.
//   ud        : opaque user data passed to tromp144_solve
//   indices   : PROOFSIZE (=2^WK = 32 for 144/5) solution indices
//   proofsize : number of indices (32)
typedef void (*tromp144_emit_fn)(void *ud, const uint32_t *indices, uint32_t proofsize);

// Allocate a persistent solver context (device buffers live for its lifetime).
//   nthreads : GPU threads (e.g. 8192);  tpb : threads/block (0 = auto ~sqrt)
// Returns NULL on allocation failure.
void *tromp144_init(unsigned nthreads, unsigned tpb);

// Solve one 140-byte header+nonce blob under the given 8-byte personalization
// (e.g. "BitcoinZ"). Emits each distinct solution via emit(). Returns the number
// of distinct solutions found, or -1 on error.
int tromp144_solve(void *ctx, const char *headernonce, const char *personal,
                   tromp144_emit_fn emit, void *ud);

// Solutions the last tromp144_solve() found and then threw away because the
// device-side buffer holds only MAXSOLS of them. Normally 0: Wagner gives ~2
// solutions per instance against a cap of 10. A non-zero value is real lost
// share value, and it is NOT the same thing as the solver's by-design discard
// of colliding pairs on bucket overflow -- that one happens earlier, and the
// two are indistinguishable from the solution rate alone.
unsigned tromp144_sols_dropped(void *ctx);

// Free a context created by tromp144_init.
void tromp144_free(void *ctx);

// Host-verify a solution (PROOFSIZE unpacked indices) for a 140-byte header
// under `personal`. Returns 0 (POW_OK) if valid, else a nonzero verify_code.
int tromp144_verify(const char *headernonce, const char *personal,
                    const uint32_t *indices);

#ifdef __cplusplus
}
#endif
