/*
 * curvehash (CurvehashCoin) — GPU scanhash.
 *
 * One thread per nonce runs the full curvehash (device secp256k1 fixed-base
 * k*G stack + SHA-256, see cuda/curvehash_device.cuh) and screens hash[7]
 * against the target high word. The host recomputes every candidate with the
 * libsecp256k1 oracle (curvehash_host_reverify) before submit, so a kernel bug
 * can only cause a local reject, never a bad share.
 *
 * The fixed-base G-table is built on the host (libsecp256k1) once per device at
 * init and uploaded. EC is compute-bound → expect a low (kH/s-class) rate.
 */

#include <miner.h>
#include <cuda_helper.h>

#include "cuda/curvehash_device.cuh"

#define CURVE_GTABLE_BYTES (32 * 256 * 64)

static bool      init_done[MAX_GPUS] = { 0 };
static uint8_t  *d_gtable[MAX_GPUS];
static uint8_t  *d_header[MAX_GPUS];
static uint32_t *d_resNonce[MAX_GPUS];

extern "C" void curvehash_build_gtable(unsigned char *out);
extern "C" int  curvehash_host_reverify(int thr_id, const uint32_t *pdata, uint32_t nonce,
                                        const uint32_t *ptarget, uint32_t *hash);
extern "C" void curvehash_host_free(int thr_id);

/* tpb=512 with a 128-register cap (__launch_bounds__(512,1)) lifts occupancy
 * ~23%->~33% on sm_86, which best hides the per-round field inversion (the
 * kernel is register/occupancy-bound, not inversion-count-bound — a warp
 * Montgomery batch inversion was measured slower). Worth ~+8-10% vs tpb 64. */
#define CURVE_TPB 512

__global__ void __launch_bounds__(CURVE_TPB, 1)
curvehash_scan_kernel(uint32_t threads, uint32_t startNonce,
    const uint8_t * __restrict__ header76, const uint8_t * __restrict__ gtable,
    uint32_t target7, uint32_t *resNonce)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= threads) return;
    uint32_t nonce = startNonce + idx;

    uint8_t hdr[80];
    #pragma unroll
    for (int i = 0; i < 76; i++) hdr[i] = header76[i];
    /* nonce hashed big-endian, matching the host swab32(nonce) buffer bytes */
    hdr[76] = (uint8_t)(nonce >> 24);
    hdr[77] = (uint8_t)(nonce >> 16);
    hdr[78] = (uint8_t)(nonce >> 8);
    hdr[79] = (uint8_t)(nonce);

    uint8_t h[32];
    if (!curvehash_full(h, hdr, gtable)) return; /* invalid-seckey nonce: skip */

    /* host compares hash[7] (uint32 at byte 28, little-endian read) <= target7 */
    uint32_t w7 = ((uint32_t)h[31] << 24) | ((uint32_t)h[30] << 16) |
                  ((uint32_t)h[29] << 8)  | (uint32_t)h[28];
    if (w7 <= target7) atomicMin(resNonce, nonce);
}

__global__ void curvehash_selftest_kernel(const uint8_t *gtable, uint8_t *out)
{
    uint8_t hdr[80];
    for (int i = 0; i < 80; i++) hdr[i] = (uint8_t)i;
    curvehash_full(out, hdr, gtable);
}

static void curvehash_init(int thr_id)
{
    cudaSetDevice(device_map[thr_id]);

    CUDA_SAFE_CALL(cudaMalloc(&d_gtable[thr_id], CURVE_GTABLE_BYTES));
    CUDA_SAFE_CALL(cudaMalloc(&d_header[thr_id], 76));
    CUDA_SAFE_CALL(cudaMalloc(&d_resNonce[thr_id], sizeof(uint32_t)));

    unsigned char *tbl = (unsigned char *)malloc(CURVE_GTABLE_BYTES);
    curvehash_build_gtable(tbl);
    CUDA_SAFE_CALL(cudaMemcpy(d_gtable[thr_id], tbl, CURVE_GTABLE_BYTES, cudaMemcpyHostToDevice));
    free(tbl);

    /* one-time device self-test vs the known KAT digest (logs on failure) */
    {
        uint8_t *d_out, h_out[32];
        CUDA_SAFE_CALL(cudaMalloc(&d_out, 32));
        curvehash_selftest_kernel <<< 1, 32 >>> (d_gtable[thr_id], d_out);
        CUDA_SAFE_CALL(cudaMemcpy(h_out, d_out, 32, cudaMemcpyDeviceToHost));
        cudaFree(d_out);
        if (memcmp(h_out, "\xb2\x64\x54\x16\xce\x97\xcf\x39\x35\x59\x2d\x82\xea\xeb\xf2\x52"
                          "\x12\x00\x8e\xbf\x04\xf6\x23\x73\x20\x3a\x71\x53\xfa\x1e\x14\x66", 32) != 0)
            gpulog(LOG_ERR, thr_id, "curvehash GPU self-test FAILED");
    }

    init_done[thr_id] = true;
}

extern "C" int scanhash_curvehash(int thr_id, struct work *work, uint32_t max_nonce,
                                  unsigned long *hashes_done)
{
    uint32_t *pdata = work->data;
    uint32_t *ptarget = work->target;
    const uint32_t first_nonce = pdata[19];
    uint32_t throughput = cuda_default_throughput(thr_id, 1U << 16); /* EC-heavy but no per-thread mem; big batch fills the GPU at tpb 512 (-i to tune) */
    if (init_done[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

    if (opt_benchmark && ptarget[7] < 0x0000ffffU)
        ptarget[7] = 0x0000ffffU;

    if (!init_done[thr_id])
        curvehash_init(thr_id);

    /* 76-byte base header = be32enc(pdata[0..18]); kernel appends the nonce */
    uint32_t _ALIGN(64) endiandata[19];
    for (int i = 0; i < 19; i++)
        be32enc(&endiandata[i], pdata[i]);
    CUDA_SAFE_CALL(cudaMemcpy(d_header[thr_id], endiandata, 76, cudaMemcpyHostToDevice));

    const uint32_t UMAX = UINT32_MAX;
    const uint32_t tpb = CURVE_TPB;

    do {
        CUDA_SAFE_CALL(cudaMemcpy(d_resNonce[thr_id], &UMAX, sizeof(uint32_t), cudaMemcpyHostToDevice));

        uint32_t grid = (throughput + tpb - 1) / tpb;
        curvehash_scan_kernel <<< grid, tpb >>> (throughput, pdata[19], d_header[thr_id],
                                                 d_gtable[thr_id], ptarget[7], d_resNonce[thr_id]);

        uint32_t win = UMAX;
        CUDA_SAFE_CALL(cudaMemcpy(&win, d_resNonce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost));

        *hashes_done = pdata[19] - first_nonce + throughput;

        if (win != UMAX) {
            uint32_t _ALIGN(64) vhash[8];
            if (curvehash_host_reverify(thr_id, pdata, win, ptarget, vhash)) {
                work_set_target_ratio(work, vhash);
                work->nonces[0] = win;
                work->valid_nonces = 1;
                pdata[19] = win;
                return 1;
            }
            gpu_increment_reject(thr_id);
            if (!opt_quiet)
                gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", win);
        }

        pdata[19] += throughput;

    } while (pdata[19] < max_nonce && !work_restart[thr_id].restart);

    *hashes_done = pdata[19] - first_nonce;
    return 0;
}

extern "C" void free_curvehash(int thr_id)
{
    if (!init_done[thr_id]) return;
    cudaSetDevice(device_map[thr_id]);
    cudaFree(d_gtable[thr_id]);
    cudaFree(d_header[thr_id]);
    cudaFree(d_resNonce[thr_id]);
    curvehash_host_free(thr_id);
    init_done[thr_id] = false;
}
