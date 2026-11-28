/* SPDX-License-Identifier: GPL-3.0-or-later
 *
 * XHash host core: epoch/light-cache/DAG lifecycle and GPU search with host
 * re-verification. Plain C interface so it never mixes ccminer's miner.h
 * (bool macro) world with the C++ STL used by the reference xhash library.
 */
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void *xhash_core_create(void);
void xhash_core_destroy(void *handle);

/* Recover the epoch number from a pool-supplied seed hash (-1 if unknown). */
int xhash_core_epoch_from_seed(const unsigned char seed[32]);

/* Build/keep the light cache and GPU DAG for this epoch. */
int xhash_core_ensure(void *handle, int epoch, int *regenerated);

/* Compare one GPU-found nonce/mix against the CPU reference. */
/* Both GPU search variants vs the CPU reference on the live DAG: 1 match, 0 mismatch (*bad_variant gets the
 * variant), -1 could not run */
int xhash_core_selftest(void *handle, int epoch, int *bad_variant);

/* Times both search variants (standard, dedup) once and keeps the faster for this handle; rates get their
 * hashes per second. Returns the variant, or -1 on error (the standard variant stays). */
int xhash_core_tune(void *handle, uint32_t throughput, double rates[2]);

/* Search [start_nonce, start_nonce + throughput). Returns 1 on a share that
 * also validates on the CPU, 0 on no share, -1 on launch failure, -2 when the
 * GPU flagged candidates that the CPU rejected. */
int xhash_core_search(void *handle, const unsigned char header[32], uint64_t start_nonce,
	const unsigned char target[32], uint32_t throughput, uint64_t *nonce_out,
	unsigned char mix_out[32], unsigned char final_out[32]);

#ifdef __cplusplus
}
#endif
