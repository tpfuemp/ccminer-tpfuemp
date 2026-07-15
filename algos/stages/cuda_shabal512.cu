/*
 * Shabal-512 for X14/X15
 */
#include "cuda_helper.h"

/* $Id: shabal.c 175 2010-05-07 16:03:20Z tp $ */
/*
 * Shabal implementation.
 *
 * ==========================(LICENSE BEGIN)============================
 *
 * Copyright (c) 2007-2010 Projet RNRT SAPHIR
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ===========================(LICENSE END)=============================
 *
 * @author Thomas Pornin <thomas.pornin@cryptolog.com>
 */

/*
 * Part of this code was automatically generated (the part between
 * the "BEGIN" and "END" markers).
 */
#include "cuda/shabal512_device.cuh"

// The Shabal-512 device implementation (permutation macros, constants and
// shabal512_hash_64) lives in cuda/shabal512_device.cuh; the kernel below
// is a thin wrapper.

/***************************************************/
// GPU Hash Function
__global__ void shabal512_gpu_hash_64(uint32_t threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *g_nonceVector)
{
	__syncthreads();

	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads)
	{
		uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);
		int hashPosition = nounce - startNounce;
		uint32_t *Hash = (uint32_t*)&g_hash[hashPosition << 3]; // [8 * hashPosition];

		shabal512_hash_64(Hash);
	}
}

/* Unit self-test for cuda/shabal512_device.cuh (docs/coding-guideline.md §7
 * layer 1), defined in cuda/xfamily_selftest.cu. */
extern bool shabal512_device_selftest(int thr_id);

__host__ void shabal512_cpu_init(int thr_id, uint32_t threads)
{
	shabal512_device_selftest(thr_id);
}

__host__ void shabal512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
	const uint32_t threadsperblock = 256;

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	size_t shared_size = 0;

	shabal512_gpu_hash_64<<<grid, block, shared_size>>>(threads, startNounce, (uint64_t*)d_hash, d_nonceVector);
}

/* Legacy-name forwarders (x14 shabal) for the not-yet-migrated consumers
 * (x17/skydoge/hmq17, x21s, ghostrider, evohash, bastion); each drops out as
 * its family switches to the bare name. */
__host__ void x14_shabal512_cpu_init(int thr_id, uint32_t threads)
{
	shabal512_cpu_init(thr_id, threads);
}

__host__ void x14_shabal512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
	shabal512_cpu_hash_64(thr_id, threads, startNounce, d_nonceVector, d_hash, order);
}
