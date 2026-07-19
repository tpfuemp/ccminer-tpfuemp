// SPDX-License-Identifier: GPL-3.0-or-later
//
// KawPoW ProgPoW per-period JIT. Generates a period-specialized CUDA kernel
// source string (the random ProgPoW program baked in as literals), compiles it
// with NVRTC, and caches the resulting CUmodule keyed on the period so a
// recompile happens at most once per PROGPOW_PERIOD (3) blocks.
//
// Provenance and design: algos/kawpow/README.md.

#pragma once

#include <cstdint>
#include <string>
#include <cuda.h>

// GPU search result. The kernel reports the first nonce whose final hash meets
// the target, plus its mix and final hashes (for host re-verification). Layout
// is mirrored verbatim in the generated device source.
struct kawpow_result
{
    uint32_t found;      // 0/1 flag, reset to 0 before each launch
    uint32_t nonce_lo;   // winning thread index (gid); nonce = start_nonce + gid
    uint32_t mix[8];     // ProgPoW mix_hash
    uint32_t final[8];   // final hash (little-endian words, as state[0..7])
};

// Returns a complete, self-contained CUDA source for the ProgPoW search kernel
// specialized to `period` (= block_number / 3) and `num_items` (256-byte DAG
// elements, baked as a compile-time constant so the DAG-index modulo folds to a
// multiply-shift). Entry point: kawpow_search.
std::string kawpow_progpow_source(uint64_t period, uint32_t num_items);

// Per-period compiled-module cache. Not thread-safe; the miner drives it from a
// single scan thread per GPU (a period only changes every 3 blocks).
class kawpow_jit
{
public:
    explicit kawpow_jit(int sm_arch) : sm_arch_(sm_arch) {}
    ~kawpow_jit();

    // Compile (or return cached) kernel for `period` with `num_items` (256-byte
    // DAG elements) baked in. On success stores the launchable function in *fn
    // and returns true.
    bool get(uint64_t period, uint32_t num_items, CUfunction* fn);

    // Number of actual NVRTC compilations performed (cache misses).
    int compiles() const { return compiles_; }

private:
    int sm_arch_;
    int compiles_ = 0;
    uint64_t cached_period_ = UINT64_MAX;
    CUmodule cached_module_ = nullptr;
    CUfunction cached_fn_ = nullptr;
};
