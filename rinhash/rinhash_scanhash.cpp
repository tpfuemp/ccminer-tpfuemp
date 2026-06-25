#include <stdio.h>
#include <cstdint>
#include <memory.h>
#include <miner.h>
#include "cuda_helper.h" //

using namespace std;

// External reference to RinHash CUDA functions
//
extern "C" void RinHash_mine_persistent(
    const uint32_t* work_data,
    uint32_t nonce_offset,
    uint32_t start_nonce,
    uint32_t num_nonces,
    uint32_t* target,
    uint32_t* found_nonce,
    uint8_t* target_hash,
    uint8_t* best_hash,
    uint32_t* solution_found
);

extern "C" void rinhash_persistent_init(uint32_t max_blocks);
extern "C" void rinhash_persistent_cleanup();

// Thread-local variables
thread_local uint32_t *d_hash = NULL;
thread_local uint8_t *d_rinhash_out = NULL;

// Initialization function for RinHash algorithm
extern "C" void rinhash_init(int thr_id)
{
    cudaSetDevice(device_map[thr_id]);
    // Initialize VRAM for 32768 CUDA threads (not nonces)
    // 32768 threads * 128 waves/thread = 4,194,304 nonces/batch
    // VRAM = 32768 threads * 64KB/thread = 2 GB VRAM
    // FIX: let rinhash_persistent_init auto-adjust inside RinHash_mine_persistent
    // rinhash_persistent_init(32768); //
}

// Cleanup function for RinHash algorithm
extern "C" void rinhash_free(int thr_id)
{
    cudaSetDevice(device_map[thr_id]);

    rinhash_persistent_cleanup(); //
    
    cudaFree(d_hash);
    cudaFree(d_rinhash_out);

    d_hash = NULL;
    d_rinhash_out = NULL;
}

// Main scanning function that tries different nonces to find a valid hash
int scanhash_rinhash(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
    uint32_t *pdata = work->data;
    uint32_t *ptarget = work->target;
    const uint32_t first_nonce = pdata[19];
    uint32_t nonce = first_nonce;
    if (opt_benchmark)
        ptarget[7] = 0xff;

    // --- LOGIC BUG FIXED (5th attempt) ---

    // 1. TOTAL WORK (Total Batch):
    // Read from --batch-size. If unset, default is 2M (2097152).
    // This is large enough to keep the GPU from starving.
    uint32_t total_batch_size = (opt_batch_size > 0) ? opt_batch_size : 2097152;

    // 2. KERNEL CHUNK:
    // This is the size of EACH kernel call.
    // 128K (131072) is a safe value that won't trigger a TDR hang.
    const uint32_t kernel_chunk_size = 2097152;
    
    // Limit the total number of nonces for this loop
    max_nonce = min(first_nonce + total_batch_size, max_nonce);
    
    // --- END LOGIC BUG FIX ---

    uint32_t found_nonce = 0;
    uint32_t solution_found = 0;
    uint8_t best_hash[32];
    uint8_t target_hash[32]; // (unused, but the API requires it)
    uint32_t target[8];

    // Convert target (already little-endian)
    for (int i = 0; i < 8; i++) {
        target[i] = ptarget[i];
    }

    work->valid_nonces = 0;
    cudaSetDevice(device_map[thr_id]);

    do {
        // CORRECT LOGIC: compute the chunk for this kernel call
        uint32_t current_chunk_size = min(kernel_chunk_size, max_nonce - nonce);
        
        if (current_chunk_size <= 0) {
            *hashes_done = nonce - first_nonce;
            return 0; // completed total_batch_size
        }

        solution_found = 0;
        RinHash_mine_persistent(
            pdata,
            19, // nonce offset
            nonce,
            current_chunk_size, // call the kernel with this chunk
            target,
            &found_nonce,
            target_hash, // (unused)
            best_hash,
            &solution_found
        );

        *hashes_done = nonce - first_nonce + current_chunk_size;

        if (solution_found) {
            uint32_t _ALIGN(64) vhash[8];
            memcpy(vhash, best_hash, 32);
            
            const uint32_t Htarg = ptarget[7];
            if (vhash[7] <= Htarg) {    
                work->valid_nonces = 1;
                work_set_target_ratio(work, vhash);
                work->nonces[0] = found_nonce;
                pdata[19] = found_nonce;
                
                // Return 1 (found)
                // Update the last scanned nonce
                pdata[19] = nonce + current_chunk_size;
                *hashes_done = nonce - first_nonce + current_chunk_size;
                return 1;
            } else {
                gpu_increment_reject(thr_id);
            }
        }
        
        // CORRECT LOGIC: advance the nonce and repeat the 'do...while' loop
        // to process the next chunk
        nonce += current_chunk_size;

    } while (nonce < max_nonce && !work_restart[thr_id].restart);

    pdata[19] = nonce;
    *hashes_done = nonce - first_nonce;
    return 0; // finished the batch without finding anything
}

// Empty function to detect algorithm - needed by ccminer
// (This function is not called in the main mining loop)
extern "C" void rinhash_hash(const void *output, const void *input)
{
    // (Skipped; not needed for performance)
    //
}
