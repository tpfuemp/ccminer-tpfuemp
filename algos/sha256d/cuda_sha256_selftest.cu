// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Unit self-test for cuda/sha256_device.cuh (coding-guideline.md §7, layer 1):
 * exercises each header building block against the OpenSSL CPU reference.
 * Runs once per process at algo init; logs a warning and returns false on
 * mismatch (the scanhash CPU-verify path remains the per-share safety net).
 */

#include <string.h>

#include <miner.h>
#include <cuda_helper.h>
#include <openssl/sha.h>

#include "cuda/sha256_device.cuh"

/* GPU side: double SHA-256 of the fixed 80-byte message m[i] = i via
 * sha256_transform_full, plus sha256_final_to_target over the same second
 * hash. out[0..7] = full second-hash state, out[8..9] = truncated words 6/7. */
__global__ __launch_bounds__(1, 1)
void sha256_selftest_gpu(uint32_t *out)
{
	uint32_t in[16], st[8];

	// first block: bytes 0..63 of m as big-endian words
	#pragma unroll
	for (int i = 0; i < 16; i++)
		in[i] = ((4*i) << 24) | ((4*i+1) << 16) | ((4*i+2) << 8) | (4*i+3);
	#pragma unroll
	for (int i = 0; i < 8; i++) st[i] = c_sha256_H[i];
	sha256_transform_full(in, st, c_sha256_K);

	// second block: bytes 64..79, padding, length 0x280 bits
	#pragma unroll
	for (int i = 0; i < 4; i++)
		in[i] = ((64+4*i) << 24) | ((64+4*i+1) << 16) | ((64+4*i+2) << 8) | (64+4*i+3);
	in[4] = 0x80000000U;
	#pragma unroll
	for (int i = 5; i < 15; i++) in[i] = 0;
	in[15] = 0x280;
	sha256_transform_full(in, st, c_sha256_K);

	// second SHA-256 over the 32-byte digest: full transform...
	uint32_t in2[16], st2[8];
	#pragma unroll
	for (int i = 0; i < 8; i++) in2[i] = st[i];
	in2[8] = 0x80000000U;
	#pragma unroll
	for (int i = 9; i < 15; i++) in2[i] = 0;
	in2[15] = 0x100;
	#pragma unroll
	for (int i = 0; i < 8; i++) st2[i] = c_sha256_H[i];
	sha256_transform_full(in2, st2, c_sha256_K);
	#pragma unroll
	for (int i = 0; i < 8; i++) out[i] = st2[i];

	// ...and truncated: only words 6/7 are defined afterwards
	#pragma unroll
	for (int i = 0; i < 8; i++) in2[i] = st[i];
	in2[8] = 0x80000000U;
	#pragma unroll
	for (int i = 9; i < 15; i++) in2[i] = 0;
	in2[15] = 0x100;
	#pragma unroll
	for (int i = 0; i < 8; i++) st2[i] = c_sha256_H[i];
	sha256_final_to_target(in2, st2, c_sha256_K);
	out[8] = st2[6];
	out[9] = st2[7];
}

__host__
bool sha256_device_selftest(int thr_id)
{
	static bool tested = false, passed = false;
	if (tested) return passed;
	tested = true;

	uint8_t m[80];
	for (int i = 0; i < 80; i++) m[i] = (uint8_t) i;

	// CPU reference: OpenSSL double SHA-256
	uint8_t ref[32];
	{
		uint8_t h1[32];
		SHA256_CTX ctx;
		SHA256_Init(&ctx); SHA256_Update(&ctx, m, 80); SHA256_Final(h1, &ctx);
		SHA256_Init(&ctx); SHA256_Update(&ctx, h1, 32); SHA256_Final(ref, &ctx);
	}

	// host header path: sha256_midstate_host + sha256_transform_full
	uint32_t data20[20], in[16], st[8];
	for (int i = 0; i < 20; i++) data20[i] = be32dec(m + 4*i);
	sha256_midstate_host(data20, st);
	for (int i = 0; i < 4; i++) in[i] = data20[16 + i];
	in[4] = 0x80000000U;
	for (int i = 5; i < 15; i++) in[i] = 0;
	in[15] = 0x280;
	sha256_transform_full(in, st, h_sha256_K);

	uint32_t in2[16], st2[8];
	for (int i = 0; i < 8; i++) in2[i] = st[i];
	in2[8] = 0x80000000U;
	for (int i = 9; i < 15; i++) in2[i] = 0;
	in2[15] = 0x100;
	for (int i = 0; i < 8; i++) st2[i] = h_sha256_H[i];
	sha256_transform_full(in2, st2, h_sha256_K);

	uint8_t hostdig[32];
	for (int i = 0; i < 8; i++) be32enc(hostdig + 4*i, st2[i]);
	const bool host_ok = (memcmp(hostdig, ref, 32) == 0);

	// device path: full transform must match the host, truncated words 6/7
	// must match the full transform
	uint32_t out[10] = { 0 };
	uint32_t *d_out = NULL;
	bool gpu_ok = false;
	if (cudaMalloc(&d_out, sizeof(out)) == cudaSuccess) {
		sha256_selftest_gpu <<<1, 1>>> (d_out);
		gpu_ok = (cudaMemcpy(out, d_out, sizeof(out), cudaMemcpyDeviceToHost) == cudaSuccess);
		cudaFree(d_out);
	}
	gpu_ok = gpu_ok && (memcmp(out, st2, 32) == 0)
	                && out[8] == st2[6] && out[9] == st2[7];

	passed = host_ok && gpu_ok;
	if (!passed)
		gpulog(LOG_WARNING, thr_id, "SHA256 device-library self-test FAILED (host %d gpu %d)", (int) host_ok, (int) gpu_ok);
	return passed;
}
