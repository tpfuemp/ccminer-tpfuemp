/*
 * Groestl-512 shared device library (docs/coding-guideline.md §3).
 *
 * QUAD-THREAD bitslice formulation: ONE HASH NEEDS FOUR THREADS (lanes
 * threadIdx.x & 3 of the same warp-internal group of 4, warp shuffles with
 * width 4). Callers launch 4x threads and build the per-lane message[8]
 * exactly as quark/cuda_quark_groestl512.cu does.
 *
 * Assembled verbatim from quark/groestl_functions_quad.h and
 * quark/groestl_transf_quad.h (both now deleted; also used by
 * cuda_groestlcoin.cu and cuda_myriadgroestl.cu).
 */

#ifndef CUDA_GROESTL512_DEVICE_CUH
#define CUDA_GROESTL512_DEVICE_CUH

#include <cuda_helper.h>

__device__ __forceinline__
void G256_Mul2(uint32_t *regs)
{
    uint32_t tmp = regs[7];
    regs[7] = regs[6];
    regs[6] = regs[5];
    regs[5] = regs[4];
    regs[4] = regs[3] ^ tmp;
    regs[3] = regs[2] ^ tmp;
    regs[2] = regs[1];
    regs[1] = regs[0] ^ tmp;
    regs[0] = tmp;
}

__device__ __forceinline__
void G256_AddRoundConstantQ_quad(uint32_t &x7, uint32_t &x6, uint32_t &x5, uint32_t &x4, uint32_t &x3, uint32_t &x2, uint32_t &x1, uint32_t &x0, int rnd)
{
    x0 = ~x0;
    x1 = ~x1;
    x2 = ~x2;
    x3 = ~x3;
    x4 = ~x4;
    x5 = ~x5;
    x6 = ~x6;
    x7 = ~x7;

#if 0
    if ((threadIdx.x & 3) != 3)
        return;

    int andmask = 0xFFFF0000;
#else
    /* from sp: faster (branching problem with if ?) */
    uint32_t andmask = -((threadIdx.x & 3) == 3) & 0xFFFF0000U;
#endif

    x0 ^= ((- (rnd & 0x01)    ) & andmask);
    x1 ^= ((-((rnd & 0x02)>>1)) & andmask);
    x2 ^= ((-((rnd & 0x04)>>2)) & andmask);
    x3 ^= ((-((rnd & 0x08)>>3)) & andmask);

    x4 ^= (0xAAAA0000 & andmask);
    x5 ^= (0xCCCC0000 & andmask);
    x6 ^= (0xF0F00000 & andmask);
    x7 ^= (0xFF000000 & andmask);
}

__device__ __forceinline__
void G256_AddRoundConstantP_quad(uint32_t &x7, uint32_t &x6, uint32_t &x5, uint32_t &x4, uint32_t &x3, uint32_t &x2, uint32_t &x1, uint32_t &x0, int rnd)
{
    if (threadIdx.x & 3)
        return;

    int andmask = 0xFFFF;

    x0 ^= ((- (rnd & 0x01)    ) & andmask);
    x1 ^= ((-((rnd & 0x02)>>1)) & andmask);
    x2 ^= ((-((rnd & 0x04)>>2)) & andmask);
    x3 ^= ((-((rnd & 0x08)>>3)) & andmask);

    x4 ^= 0xAAAAU;
    x5 ^= 0xCCCCU;
    x6 ^= 0xF0F0U;
    x7 ^= 0xFF00U;
}

__device__ __forceinline__
void G16mul_quad(uint32_t &x3, uint32_t &x2, uint32_t &x1, uint32_t &x0,
                                       uint32_t &y3, uint32_t &y2, uint32_t &y1, uint32_t &y0)
{
    uint32_t t0,t1,t2;
    
    t0 = ((x2 ^ x0) ^ (x3 ^ x1)) & ((y2 ^ y0) ^ (y3 ^ y1));
    t1 = ((x2 ^ x0) & (y2 ^ y0)) ^ t0;
    t2 = ((x3 ^ x1) & (y3 ^ y1)) ^ t0 ^ t1;

    t0 = (x2^x3) & (y2^y3);
    x3 = (x3 & y3) ^ t0 ^ t1;
    x2 = (x2 & y2) ^ t0 ^ t2;

    t0 = (x0^x1) & (y0^y1);
    x1 = (x1 & y1) ^ t0 ^ t1;
    x0 = (x0 & y0) ^ t0 ^ t2;
}

__device__ __forceinline__
void G256_inv_quad(uint32_t &x7, uint32_t &x6, uint32_t &x5, uint32_t &x4, uint32_t &x3, uint32_t &x2, uint32_t &x1, uint32_t &x0)
{
    uint32_t t0,t1,t2,t3,t4,t5,t6,a,b;

    t3 = x7;
    t2 = x6;
    t1 = x5;
    t0 = x4;

    G16mul_quad(t3, t2, t1, t0, x3, x2, x1, x0);

    a = (x4 ^ x0);
    t0 ^= a;
    t2 ^= (x7 ^ x3) ^ (x5 ^ x1); 
    t1 ^= (x5 ^ x1) ^ a;
    t3 ^= (x6 ^ x2) ^ a;

    b = t0 ^ t1;
    t4 = (t2 ^ t3) & b;
    a = t4 ^ t3 ^ t1;
    t5 = (t3 & t1) ^ a;
    t6 = (t2 & t0) ^ a ^ (t2 ^ t0);

    t4 = (t5 ^ t6) & b;
    t1 = (t6 & t1) ^ t4;
    t0 = (t5 & t0) ^ t4;

    t4 = (t5 ^ t6) & (t2^t3);
    t3 = (t6 & t3) ^ t4;
    t2 = (t5 & t2) ^ t4;

    G16mul_quad(x3, x2, x1, x0, t1, t0, t3, t2);

    G16mul_quad(x7, x6, x5, x4, t1, t0, t3, t2);
}

__device__ __forceinline__
void transAtoX_quad(uint32_t &x0, uint32_t &x1, uint32_t &x2, uint32_t &x3, uint32_t &x4, uint32_t &x5, uint32_t &x6, uint32_t &x7)
{
    uint32_t t0, t1;
    t0 = x0 ^ x1 ^ x2;
    t1 = x5 ^ x6;
    x2 = t0 ^ t1 ^ x7;
    x6 = t0 ^ x3 ^ x6;
    x3 = x0 ^ x1 ^ x3 ^ x4 ^ x7;    
    x4 = x0 ^ x4 ^ t1;
    x2 = t0 ^ t1 ^ x7;
    x1 = x0 ^ x1 ^ t1;
    x7 = x0 ^ t1 ^ x7;
    x5 = x0 ^ t1;
}

__device__ __forceinline__
void transXtoA_quad(uint32_t &x0, uint32_t &x1, uint32_t &x2, uint32_t &x3, uint32_t &x4, uint32_t &x5, uint32_t &x6, uint32_t &x7)
{
    uint32_t t0,t2,t3,t5;

    x1 ^= x4;
    t0 = x1 ^ x6;
    x1 ^= x5;

    t2 = x0 ^ x2;
    x2 = x3 ^ x5;
    t2 ^= x2 ^ x6;
    x2 ^= x7;
    t3 = x4 ^ x2 ^ x6;

    t5 = x0 ^ x6;
    x4 = x3 ^ x7;
    x0 = x3 ^ x5;

    x6 = t0;    
    x3 = t2;
    x7 = t3;    
    x5 = t5;    
}

__device__ __forceinline__
void sbox_quad(uint32_t *r)
{
    transAtoX_quad(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);

    G256_inv_quad(r[2], r[4], r[1], r[7], r[3], r[0], r[5], r[6]);

    transXtoA_quad(r[7], r[1], r[4], r[2], r[6], r[5], r[0], r[3]);
    
    r[0] = ~r[0];
    r[1] = ~r[1];
    r[5] = ~r[5];
    r[6] = ~r[6];
}

__device__ __forceinline__
void G256_ShiftBytesP_quad(uint32_t &x7, uint32_t &x6, uint32_t &x5, uint32_t &x4, uint32_t &x3, uint32_t &x2, uint32_t &x1, uint32_t &x0)
{
    uint32_t t0,t1;

    int tpos = threadIdx.x & 0x03;
    int shift1 = tpos << 1;
    int shift2 = shift1+1 + ((tpos == 3)<<2);

    t0 = __byte_perm(x0, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x0, 0, 0x3232)>>shift2;
    x0 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x1, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x1, 0, 0x3232)>>shift2;
    x1 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x2, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x2, 0, 0x3232)>>shift2;
    x2 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x3, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x3, 0, 0x3232)>>shift2;
    x3 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x4, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x4, 0, 0x3232)>>shift2;
    x4 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x5, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x5, 0, 0x3232)>>shift2;
    x5 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x6, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x6, 0, 0x3232)>>shift2;
    x6 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x7, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x7, 0, 0x3232)>>shift2;
    x7 = __byte_perm(t0, t1, 0x5410);
}

__device__ __forceinline__
void G256_ShiftBytesQ_quad(uint32_t &x7, uint32_t &x6, uint32_t &x5, uint32_t &x4, uint32_t &x3, uint32_t &x2, uint32_t &x1, uint32_t &x0)
{
    uint32_t t0,t1;

    int tpos = threadIdx.x & 0x03;
    int shift1 = (1-(tpos>>1)) + ((tpos & 0x01)<<2);
    int shift2 = shift1+2 + ((tpos == 1)<<2);

    t0 = __byte_perm(x0, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x0, 0, 0x3232)>>shift2;
    x0 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x1, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x1, 0, 0x3232)>>shift2;
    x1 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x2, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x2, 0, 0x3232)>>shift2;
    x2 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x3, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x3, 0, 0x3232)>>shift2;
    x3 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x4, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x4, 0, 0x3232)>>shift2;
    x4 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x5, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x5, 0, 0x3232)>>shift2;
    x5 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x6, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x6, 0, 0x3232)>>shift2;
    x6 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x7, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x7, 0, 0x3232)>>shift2;
    x7 = __byte_perm(t0, t1, 0x5410);
}

#if __CUDA_ARCH__ < 300
/**
 * __shfl() returns the value of var held by the thread whose ID is given by srcLane.
 * If srcLane is outside the range 0..width-1, the thread’s own value of var is returned.
 */
#undef __shfl
#define __shfl(var, srcLane, width) (uint32_t)(var)
#endif

__device__ __forceinline__
void G256_MixFunction_quad(uint32_t *r)
{
#define SHIFT64_16(hi, lo)    __byte_perm(lo, hi, 0x5432)
#define A(v, u)             __shfl((int)r[v], ((threadIdx.x+u)&0x03), 4)
#define S(idx, l)            SHIFT64_16( A(idx, (l+1)), A(idx, l) )

#define DOUBLE_ODD(i, bc)        ( S(i, (bc)) ^ A(i, (bc) + 1) )
#define DOUBLE_EVEN(i, bc)        ( S(i, (bc)) ^ A(i, (bc)    ) )

#define SINGLE_ODD(i, bc)        ( S(i, (bc)) )
#define SINGLE_EVEN(i, bc)        ( A(i, (bc)) )
    uint32_t b[8];

#pragma unroll 8
    for(int i=0;i<8;i++)
        b[i] = DOUBLE_ODD(i, 1) ^ DOUBLE_EVEN(i, 3);

    G256_Mul2(b);
#pragma unroll 8
    for(int i=0;i<8;i++)
        b[i] = b[i] ^ DOUBLE_ODD(i, 3) ^ DOUBLE_ODD(i, 4) ^ SINGLE_ODD(i, 6);

    G256_Mul2(b);
#pragma unroll 8
    for(int i=0;i<8;i++)
        r[i] = b[i] ^ DOUBLE_EVEN(i, 2) ^ DOUBLE_EVEN(i, 3) ^ SINGLE_EVEN(i, 5);

#undef S
#undef A
#undef SHIFT64_16
#undef t
#undef X
}

__device__ __forceinline__
void groestl512_perm_P_quad(uint32_t *r)
{
    for(int round=0;round<14;round++)
    {
        G256_AddRoundConstantP_quad(r[7], r[6], r[5], r[4], r[3], r[2], r[1], r[0], round);
        sbox_quad(r);
        G256_ShiftBytesP_quad(r[7], r[6], r[5], r[4], r[3], r[2], r[1], r[0]);
        G256_MixFunction_quad(r);
    }
}

__device__ __forceinline__
void groestl512_perm_Q_quad(uint32_t *r)
{    
    for(int round=0;round<14;round++)
    {
        G256_AddRoundConstantQ_quad(r[7], r[6], r[5], r[4], r[3], r[2], r[1], r[0], round);
        sbox_quad(r);
        G256_ShiftBytesQ_quad(r[7], r[6], r[5], r[4], r[3], r[2], r[1], r[0]);
        G256_MixFunction_quad(r);
    }
}

__device__ __forceinline__
void groestl512_progressMessage_quad(uint32_t *state, uint32_t *message)
{
#pragma unroll 8
    for(int u=0;u<8;u++) state[u] = message[u];

    if ((threadIdx.x & 0x03) == 3) state[ 1] ^= 0x00008000;
    groestl512_perm_P_quad(state);
    if ((threadIdx.x & 0x03) == 3) state[ 1] ^= 0x00008000;
    groestl512_perm_Q_quad(message);
#pragma unroll 8
    for(int u=0;u<8;u++) state[u] ^= message[u];
#pragma unroll 8
    for(int u=0;u<8;u++) message[u] = state[u];
    groestl512_perm_P_quad(message);
#pragma unroll 8
    for(int u=0;u<8;u++) state[u] ^= message[u];
}

/* File included in quark/groestl (quark/jha,nist5/X11+) and groest/myriad coins for SM 3+ */

#define merge8(z,x,y)\
	z=__byte_perm(x, y, 0x5140); \

#define SWAP8(x,y)\
	x=__byte_perm(x, y, 0x5410); \
	y=__byte_perm(x, y, 0x7632);

#define SWAP4(x,y)\
	t = (y<<4); \
	t = (x ^ t); \
	t = 0xf0f0f0f0UL & t; \
	x = (x ^ t); \
	t=  t>>4;\
	y=  y ^ t;

#define SWAP2(x,y)\
	t = (y<<2); \
	t = (x ^ t); \
	t = 0xccccccccUL & t; \
	x = (x ^ t); \
	t=  t>>2;\
	y=  y ^ t;

#define SWAP1(x,y)\
	t = (y+y); \
	t = (x ^ t); \
	t = 0xaaaaaaaaUL & t; \
	x = (x ^ t); \
	t=  t>>1;\
	y=  y ^ t;


__device__ __forceinline__
void to_bitslice_quad(uint32_t *const __restrict__ input, uint32_t *const __restrict__ output)
{
	uint32_t other[8];
	uint32_t d[8];
	uint32_t t;
	const unsigned int n = threadIdx.x & 3;

	#pragma unroll
	for (int i = 0; i < 8; i++) {
		input[i] = __shfl((int)input[i], n ^ (3*(n >=1 && n <=2)), 4);
		other[i] = __shfl((int)input[i], (threadIdx.x + 1) & 3, 4);
		input[i] = __shfl((int)input[i], threadIdx.x & 2, 4);
		other[i] = __shfl((int)other[i], threadIdx.x & 2, 4);
		if (threadIdx.x & 1) {
			input[i] = __byte_perm(input[i], 0, 0x1032);
			other[i] = __byte_perm(other[i], 0, 0x1032);
		}
	}

	merge8(d[0], input[0], input[4]);
	merge8(d[1], other[0], other[4]);
	merge8(d[2], input[1], input[5]);
	merge8(d[3], other[1], other[5]);
	merge8(d[4], input[2], input[6]);
	merge8(d[5], other[2], other[6]);
	merge8(d[6], input[3], input[7]);
	merge8(d[7], other[3], other[7]);

	SWAP1(d[0], d[1]);
	SWAP1(d[2], d[3]);
	SWAP1(d[4], d[5]);
	SWAP1(d[6], d[7]);

	SWAP2(d[0], d[2]);
	SWAP2(d[1], d[3]);
	SWAP2(d[4], d[6]);
	SWAP2(d[5], d[7]);

	SWAP4(d[0], d[4]);
	SWAP4(d[1], d[5]);
	SWAP4(d[2], d[6]);
	SWAP4(d[3], d[7]);

	output[0] = d[0];
	output[1] = d[1];
	output[2] = d[2];
	output[3] = d[3];
	output[4] = d[4];
	output[5] = d[5];
	output[6] = d[6];
	output[7] = d[7];
}

__device__ __forceinline__
void from_bitslice_quad(const uint32_t *const __restrict__ input, uint32_t *const __restrict__ output)
{
	uint32_t d[8];
	uint32_t t;

	d[0] = __byte_perm(input[0], input[4], 0x7531);
	d[1] = __byte_perm(input[1], input[5], 0x7531);
	d[2] = __byte_perm(input[2], input[6], 0x7531);
	d[3] = __byte_perm(input[3], input[7], 0x7531);

	SWAP1(d[0], d[1]);
	SWAP1(d[2], d[3]);

	SWAP2(d[0], d[2]);
	SWAP2(d[1], d[3]);

	t = __byte_perm(d[0], d[2], 0x5410);
	d[2] = __byte_perm(d[0], d[2], 0x7632);
	d[0] = t;

	t = __byte_perm(d[1], d[3], 0x5410);
	d[3] = __byte_perm(d[1], d[3], 0x7632);
	d[1] = t;

	SWAP4(d[0], d[2]);
	SWAP4(d[1], d[3]);

	output[0] = d[0];
	output[2] = d[1];
	output[4] = d[0] >> 16;
	output[6] = d[1] >> 16;
	output[8] = d[2];
	output[10] = d[3];
	output[12] = d[2] >> 16;
	output[14] = d[3] >> 16;

	#pragma unroll 8
	for (int i = 0; i < 16; i+=2) {
		if (threadIdx.x & 1) output[i] = __byte_perm(output[i], 0, 0x1032);
		output[i] = __byte_perm(output[i], __shfl((int)output[i], (threadIdx.x+1)&3, 4), 0x7610);
		output[i+1] = __shfl((int)output[i], (threadIdx.x+2)&3, 4);
		if (threadIdx.x & 3) output[i] = output[i+1] = 0;
	}
}

/* transform-internal macros, not part of the interface */
#undef merge8
#undef SWAP8
#undef SWAP4
#undef SWAP2
#undef SWAP1
#undef DOUBLE_ODD
#undef DOUBLE_EVEN
#undef SINGLE_ODD
#undef SINGLE_EVEN

/* Groestl-512 of a 64-byte input, quad-thread: message[8] holds this lane's
 * quarter of the padded 16-word block, hash[16] is valid on lane 0 only
 * (zeroed on lanes 1..3, as from_bitslice_quad leaves it). */
__device__ __forceinline__
void groestl512_hash_quad(uint32_t *const message, uint32_t *const hash)
{
	uint32_t msgBitsliced[8];
	to_bitslice_quad(message, msgBitsliced);

	uint32_t state[8];
	groestl512_progressMessage_quad(state, msgBitsliced);

	from_bitslice_quad(state, hash);
}

#endif /* CUDA_GROESTL512_DEVICE_CUH */
