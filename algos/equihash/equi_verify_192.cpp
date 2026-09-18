// ---------------------------------------------------------------------------
// Host-side consensus verifier for Equihash 192/7.
//
// The equi24 solver serves both DIGITBITS=24 variants but the re-verify does
// not: tromp144_verify is compiled -DWN=144, so its PROOFSIZE is 32 and its
// BLAKE2b personalization carries 144/5. Handed a 192/7 proof it reads 32 of
// the 128 indices and checks them against the wrong parameters, so every
// solution fails.
//
// The host re-verify is what keeps a solver defect to a local reject rather
// than a bad share, so a variant without one has no safety net.
//
// equi_tromp.h derives everything from WN/WK, so this is the same pool-proven
// verifier in a second TU with different defines and a distinct namespace.
// ---------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "blake2/blake2.h"      // blake2b_state/param + eq_blake2b_* (extern "C")
#include "cuda_equi24.h"

#define WN 192
#define WK 7

namespace tromp192 {
#include "equi_tromp.h"         // params, setheader, verifyrec, verify
}

// Verify a 192/7 proof: PROOFSIZE (=128) unpacked indices against a 140-byte
// header+nonce under `personal`. Returns 0 (POW_OK) if valid.
extern "C" int equi24_192_verify(const char *headernonce, const char *personal,
                                 const uint32_t *indices)
{
	// verify() takes a non-const array; it does not modify it, but duped()
	// sorts a COPY and verifyrec walks it read-only.
	uint32_t idx[1 << WK];
	memcpy(idx, indices, sizeof(idx));
	return tromp192::verify(idx, headernonce, HEADERNONCELEN, personal);
}
