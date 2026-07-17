/*
 * Unity build of the vendored libsecp256k1 for the curvehash PoW.
 *
 * curvehash's proof-of-work runs 8 rounds of secp256k1 fixed-base scalar
 * multiplication (secp256k1_ec_pubkey_create) interleaved with SHA-256; the
 * host CPU path (and, later, the GPU candidate re-verify) needs a bit-exact
 * secp256k1_ec_pubkey_create oracle. We vendor the same libsecp256k1 the
 * upstream reference miner used and compile it as a single translation unit.
 *
 * Config (portable, MSVC-safe — no __int128, no build-time gen_context):
 *   FIELD_10X26 + SCALAR_8X32     -> 32-bit limbs, pure C
 *   NUM_NONE                      -> no GMP dependency
 *   FIELD/SCALAR_INV_BUILTIN      -> self-contained modular inverse
 *   (USE_ECMULT_STATIC_PRECOMPUTATION intentionally left undefined ->
 *    the ecmult_gen table is built at runtime in secp256k1_context_create,
 *    so no precomputed table / codegen step is required.)
 */
#define USE_NUM_NONE 1
#define USE_FIELD_INV_BUILTIN 1
#define USE_SCALAR_INV_BUILTIN 1
#define USE_FIELD_10X26 1
#define USE_SCALAR_8X32 1

#include "secp256k1/src/secp256k1.c"
