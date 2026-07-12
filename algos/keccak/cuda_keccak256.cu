/**
 * KECCAK-256 CUDA optimised implementation, based on ccminer-alexis code
 *
 * 2026-07: round body / truncated final round moved to the shared
 * cuda/keccak_device.cuh (bit-identical extraction — this kernel was the
 * donor); sub-sm_61 launch shapes deleted per arch floor.
 */

#include <miner.h>

extern "C" {
#include <stdint.h>
#include <memory.h>
}

#include <cuda_helper.h>
#include "cuda/keccak_device.cuh"

extern bool keccak_device_selftest(int thr_id);

#define TPB52 1024
#define NPT 2
#define NBN 2

static uint32_t *d_nonces[MAX_GPUS];
static uint32_t *h_nonces[MAX_GPUS];

__constant__ uint2 c_message48[6];
__constant__ uint2 c_mid[17];

__global__ __launch_bounds__(TPB52, 1)
void keccak256_gpu_hash_80(uint32_t threads, uint32_t startNonce, uint32_t *resNounce, const uint2 highTarget)
{
	uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;
	uint2 s[25], t[5], v, w, u[5];

	uint64_t step     = gridDim.x * blockDim.x;
	uint64_t maxNonce = startNonce + threads;
	for(uint64_t nounce = startNonce + thread; nounce<maxNonce;nounce+=step) {

		/* round 0 from the 72-byte midstate: s[9] carries the nonce,
		 * s[10] = 0x01 — Keccak-256 padding (NOT 0x06 NIST SHA3). */
		s[ 9] = make_uint2(c_message48[0].x,cuda_swab32(nounce));
		s[10] = make_uint2(1, 0);

		t[ 4] = c_message48[1]^s[ 9];
		/* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
		u[ 0] = t[4] ^ c_mid[ 0];
		u[ 1] = c_mid[ 1] ^ ROL2(t[4],1);
		u[ 2] = c_mid[ 2];
		/* thetarho pi: b[..] = rotl(a[..] ^ d[...], ..)*/
		s[ 7] = ROL2(s[10]^u[0], 3);
		s[10] = c_mid[ 3];
		    w = c_mid[ 4];
		s[20] = c_mid[ 5];
		s[ 6] = ROL2(s[ 9]^u[2],20);
		s[ 9] = c_mid[ 6];
		s[22] = c_mid[ 7];
		s[14] = ROL2(u[0],18);
		s[ 2] = c_mid[ 8];
		s[12] = ROL2(u[1],25);
		s[13] = c_mid[ 9];
		s[19] = ROR8(u[1]);
		s[23] = ROR2(u[0],23);
		s[15] = c_mid[10];
		s[ 4] = c_mid[11];
		s[24] = c_mid[12];
		s[21] = ROR2(c_message48[2]^u[1], 9);
		s[ 8] = c_mid[13];
		s[16] = ROR2(c_message48[3]^u[0],28);
		s[ 5] = ROL2(c_message48[4]^u[1],28);
		s[ 3] = ROL2(u[1],21);
		s[18] = c_mid[14];
		s[17] = c_mid[15];
		s[11] = c_mid[16];

		/* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
		v = c_message48[5]^u[0];
		s[ 0] = keccak_chi(v,w,s[ 2]);
		s[ 1] = keccak_chi(w,s[ 2],s[ 3]);
		s[ 2] = keccak_chi(s[ 2],s[ 3],s[ 4]);
		s[ 3] = keccak_chi(s[ 3],s[ 4],v);
		s[ 4] = keccak_chi(s[ 4],v,w);
		v = s[ 5]; w = s[ 6]; s[ 5] = keccak_chi(v,w,s[ 7]); s[ 6] = keccak_chi(w,s[ 7],s[ 8]); s[ 7] = keccak_chi(s[ 7],s[ 8],s[ 9]); s[ 8] = keccak_chi(s[ 8],s[ 9],v);s[ 9] = keccak_chi(s[ 9],v,w);
		v = s[10]; w = s[11]; s[10] = keccak_chi(v,w,s[12]); s[11] = keccak_chi(w,s[12],s[13]); s[12] = keccak_chi(s[12],s[13],s[14]); s[13] = keccak_chi(s[13],s[14],v);s[14] = keccak_chi(s[14],v,w);
		v = s[15]; w = s[16]; s[15] = keccak_chi(v,w,s[17]); s[16] = keccak_chi(w,s[17],s[18]); s[17] = keccak_chi(s[17],s[18],s[19]); s[18] = keccak_chi(s[18],s[19],v);s[19] = keccak_chi(s[19],v,w);
		v = s[20]; w = s[21]; s[20] = keccak_chi(v,w,s[22]); s[21] = keccak_chi(w,s[22],s[23]); s[22] = keccak_chi(s[22],s[23],s[24]); s[23] = keccak_chi(s[23],s[24],v);s[24] = keccak_chi(s[24],v,w);

		/* iota: a[0,0] ^= round constant */
		s[ 0] ^= c_keccak_rc[0];

		/* rounds 1-22: shared round body; round 23 is truncated to lane 3 */
		#pragma unroll 22
		for (int i = 1; i < 23; i++)
			keccak_round(s, c_keccak_rc[i]);

		if (devectorize(keccak_final_lane3(s)) <= devectorize(highTarget)) {
			const uint32_t tmp = atomicExch(&resNounce[0], nounce);
			if (tmp != UINT32_MAX)
				resNounce[1] = tmp;
		}
	}
}

__host__
void keccak256_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t* resNonces, const uint2 highTarget)
{
	const uint32_t tpb = TPB52;
	const dim3 grid((threads + (NPT*tpb)-1)/(NPT*tpb));
	const dim3 block(tpb);

	keccak256_gpu_hash_80<<<grid, block>>>(threads, startNonce, d_nonces[thr_id], highTarget);
	cudaMemcpy(h_nonces[thr_id], d_nonces[thr_id], NBN*sizeof(uint32_t), cudaMemcpyDeviceToHost);
	memcpy(resNonces, h_nonces[thr_id], NBN*sizeof(uint32_t));
}

__host__
void keccak256_setBlock_80(uint64_t *endiandata)
{
	uint64_t midstate[17], s[25];
	uint64_t t[5], u[5];

	s[10] = 1; //(uint64_t)make_uint2(1, 0);
	s[16] = ((uint64_t)1)<<63; //(uint64_t)make_uint2(0, 0x80000000);

	t[0] = endiandata[0] ^ endiandata[5] ^ s[10];
	t[1] = endiandata[1] ^ endiandata[6] ^ s[16];
	t[2] = endiandata[2] ^ endiandata[7];
	t[3] = endiandata[3] ^ endiandata[8];

	midstate[ 0] = ROTL64(t[1], 1);         //u[0] -partial
	       u[1] = t[ 0] ^ ROTL64(t[2], 1);  //u[1]
	       u[2] = t[ 1] ^ ROTL64(t[3], 1);  //u[2]
	midstate[ 1] = t[ 2];                   //u[3] -partial
	midstate[ 2] = t[ 3] ^ ROTL64(t[0], 1); //u[4]
	midstate[ 3] = ROTL64(endiandata[1]^u[1], 1); //v
	midstate[ 4] = ROTL64(endiandata[6]^u[1], 44);
	midstate[ 5] = ROTL64(endiandata[2]^u[2], 62);
	midstate[ 6] = ROTL64(u[2], 61);
	midstate[ 7] = ROTL64(midstate[2], 39);
	midstate[ 8] = ROTL64(u[2], 43);
	midstate[ 9] = ROTL64(midstate[2], 8);
	midstate[10] = ROTL64(endiandata[4]^midstate[ 2],27);
	midstate[11] = ROTL64(midstate[2], 14);
	midstate[12] = ROTL64(u[1], 2);
	midstate[13] = ROTL64(s[16] ^ u[1], 45);
	midstate[14] = ROTL64(u[2],15);
	midstate[15] = ROTL64(u[1],10);
	midstate[16] = ROTL64(endiandata[7]^u[2], 6);

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_mid, midstate,17*sizeof(uint64_t), 0, cudaMemcpyHostToDevice));

	// pass only what's needed
	uint64_t message48[6];
	message48[0] = endiandata[9];
	message48[1] = endiandata[4];
	message48[2] = endiandata[8];
	message48[3] = endiandata[5];
	message48[4] = endiandata[3];
	message48[5] = endiandata[0];
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_message48, message48, 6*sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
}

__host__
void keccak256_cpu_init(int thr_id)
{
	CUDA_SAFE_CALL(cudaMalloc(&d_nonces[thr_id], NBN*sizeof(uint32_t)));
	h_nonces[thr_id] = (uint32_t*) malloc(NBN * sizeof(uint32_t));
	if(h_nonces[thr_id] == NULL) {
		gpulog(LOG_ERR,thr_id,"Host memory allocation failed");
		exit(EXIT_FAILURE);
	}
	keccak_device_selftest(thr_id);
}

__host__
void keccak256_setOutput(int thr_id)
{
	CUDA_SAFE_CALL(cudaMemset(d_nonces[thr_id], 0xff, NBN*sizeof(uint32_t)));
}

__host__
void keccak256_cpu_free(int thr_id)
{
	cudaFree(d_nonces[thr_id]);
	free(h_nonces[thr_id]);
}
