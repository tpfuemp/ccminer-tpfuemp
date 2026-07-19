// SPDX-License-Identifier: GPL-3.0-or-later
//
// KawPoW epoch/DAG state machine (see kawpow_dag.h).
//
// Builds the Ethash light cache + L1 cache on the host with the vendored
// reference (algos/kawpow/ethash), uploads them, and generates the full DAG on
// the GPU via kawpow_generate_dag_cpu. Regeneration is synchronous on epoch
// change (correctness first; async double-buffering is a later optimization).

#include "kawpow_dag.h"

#include "ethash/ethash.hpp"
#include "ethash/ethash-internal.hpp"
#include "ethash/ethash.h"

#include <cstdio>
#include <cuda_runtime.h>

extern "C" cudaError_t kawpow_generate_dag_cpu(
    const uint32_t* d_light, uint32_t* d_dag, uint32_t dag_nodes, uint32_t light_nodes);

kawpow_dag::~kawpow_dag() { release(); }

void kawpow_dag::release()
{
    if (d_dag_)   { cudaFree(d_dag_);   d_dag_ = nullptr; }
    if (d_l1_)    { cudaFree(d_l1_);    d_l1_ = nullptr; }
    if (d_light_) { cudaFree(d_light_); d_light_ = nullptr; }
    epoch_ = -1;
    items_ = 0;
}

bool kawpow_dag::ensure(int height, bool* regenerated)
{
    const int epoch = ethash::get_epoch_number(height);
    if (regenerated) *regenerated = false;
    if (d_dag_ && epoch == epoch_)
        return true;  // already resident

    // Build the host epoch context (light cache + L1 cache).
    ethash::epoch_context_ptr ctx = ethash::create_epoch_context(epoch);
    if (!ctx)
    {
        fprintf(stderr, "kawpow_dag: create_epoch_context(%d) OOM\n", epoch);
        return false;
    }

    const uint32_t light_nodes = (uint32_t)ctx->light_cache_num_items;
    const uint32_t items = (uint32_t)ctx->full_dataset_num_items;  // 128-byte items
    const uint32_t dag_nodes = items * 2u;                         // 64-byte nodes
    const size_t light_bytes = (size_t)light_nodes * 64u;
    const size_t dag_bytes = (size_t)dag_nodes * 64u;

    // Free any previous epoch's buffers before allocating the new ones.
    release();

    cudaError_t e;
    if ((e = cudaMalloc(&d_light_, light_bytes)) != cudaSuccess) goto fail;
    if ((e = cudaMemcpy(d_light_, ctx->light_cache, light_bytes, cudaMemcpyHostToDevice)) != cudaSuccess) goto fail;
    if ((e = cudaMalloc(&d_dag_, dag_bytes)) != cudaSuccess) goto fail;
    if ((e = kawpow_generate_dag_cpu(d_light_, d_dag_, dag_nodes, light_nodes)) != cudaSuccess) goto fail;
    if ((e = cudaMalloc(&d_l1_, 16 * 1024)) != cudaSuccess) goto fail;
    if ((e = cudaMemcpy(d_l1_, ctx->l1_cache, 16 * 1024, cudaMemcpyHostToDevice)) != cudaSuccess) goto fail;

    // The light cache is only needed during DAG generation.
    cudaFree(d_light_);
    d_light_ = nullptr;

    epoch_ = epoch;
    items_ = items;
    if (regenerated) *regenerated = true;
    return true;

fail:
    fprintf(stderr, "kawpow_dag: epoch %d build failed: %s\n", epoch, cudaGetErrorString(e));
    release();
    return false;
}
