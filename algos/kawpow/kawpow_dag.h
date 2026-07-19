// SPDX-License-Identifier: GPL-3.0-or-later
//
// KawPoW epoch/DAG state machine. Holds the per-GPU Ethash DAG in VRAM and
// rebuilds it when the epoch changes (every EPOCH_LENGTH = 7500 blocks). This is
// the persistent per-GPU state that ccminer's per-scanhash model lacks; a job's
// height selects the epoch, and mining uses the resident DAG + L1 cache.
//
// Provenance and design: algos/kawpow/README.md.

#pragma once

#include <cstdint>

class kawpow_dag
{
public:
    kawpow_dag() = default;
    ~kawpow_dag();

    // Ensure the DAG for the epoch of `height` is resident in VRAM, building it
    // if the epoch changed (or on first call). Returns true on success. Sets
    // *regenerated to whether a (re)build happened this call (optional).
    bool ensure(int height, bool* regenerated = nullptr);

    void release();

    // Valid after a successful ensure().
    const uint32_t* dag() const { return d_dag_; }        // full DAG, flat 64-byte nodes
    const uint32_t* l1() const { return d_l1_; }          // 4096-word L1 cache
    uint32_t dataset_items_128() const { return items_; } // full_dataset_num_items
    int epoch() const { return epoch_; }

private:
    int epoch_ = -1;
    uint32_t* d_dag_ = nullptr;
    uint32_t* d_l1_ = nullptr;
    uint32_t* d_light_ = nullptr;
    uint32_t items_ = 0;
};
