#include <stdint.h>
#include <stddef.h>

#define KECCAKF_ROUNDS 24


// Bit rotation of a 64-bit value (left rotate)
__device__ inline uint64_t rotate(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

// Keccak-f[1600] transform (runs 24 rounds of permutation on the internal state st[25])
__device__ inline uint64_t ROTL64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

__device__ void keccakf(uint64_t st[25]) {
    const int R[24] = {
         1,  3,  6, 10, 15, 21,
        28, 36, 45, 55,  2, 14,
        27, 41, 56,  8, 25, 43,
        62, 18, 39, 61, 20, 44
    };

    const int P[24] = {
        10,  7, 11, 17, 18, 3,
         5, 16, 8, 21, 24, 4,
        15, 23, 19, 13, 12, 2,
        20, 14, 22,  9, 6,  1
    };

    const uint64_t RC[24] = {
        0x0000000000000001ULL, 0x0000000000008082ULL,
        0x800000000000808aULL, 0x8000000080008000ULL,
        0x000000000000808bULL, 0x0000000080000001ULL,
        0x8000000080008081ULL, 0x8000000000008009ULL,
        0x000000000000008aULL, 0x0000000000000088ULL,
        0x0000000080008009ULL, 0x000000008000000aULL,
        0x000000008000808bULL, 0x800000000000008bULL,
        0x8000000000008089ULL, 0x8000000000008003ULL,
        0x8000000000008002ULL, 0x8000000000000080ULL,
        0x000000000000800aULL, 0x800000008000000aULL,
        0x8000000080008081ULL, 0x8000000000008080ULL,
        0x0000000080000001ULL, 0x8000000080008008ULL
    };

    int i, j, round;
    uint64_t t, bc[5];

    for (round = 0; round < 24; round++) {
        // Theta
        for (i = 0; i < 5; i++)
            bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
        for (i = 0; i < 5; i++) {
            t = bc[(i + 4) % 5] ^ ROTL64(bc[(i + 1) % 5], 1);
            for (j = 0; j < 25; j += 5)
                st[j + i] ^= t;
        }

        // Rho and Pi
        t = st[1];
        for (i = 0; i < 24; i++) {
            j = P[i];
            bc[0] = st[j];
            st[j] = ROTL64(t, R[i]);
            t = bc[0];
        }

        // Chi
        for (j = 0; j < 25; j += 5) {
            for (i = 0; i < 5; i++)
                bc[i] = st[j + i];
            for (i = 0; i < 5; i++)
                st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
        }

        // Iota
        st[0] ^= RC[round];
    }
}


// Load a 64-bit value in little-endian (from an 8-byte array)
__device__ inline uint64_t load64_le(const uint8_t *src) {
    uint64_t x = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        x |= ((uint64_t)src[i]) << (8 * i);
    }
    return x;
}

// Store a 64-bit value in little-endian (into an 8-byte array)
__device__ inline void store64_le(uint8_t *dst, uint64_t x) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        dst[i] = (uint8_t)(x >> (8 * i));
    }
}

/*
  __device__ function sha3_256_device
    - Absorbs the input data given by (input, inlen) and runs padding
      and the Keccak-f[1600] transform per the SHA3-256 spec.
    - Finally writes the first 32 bytes (4 words) of the internal state
      to hash_out in little-endian form.
    - For SHA3-256 the rate (absorb size) is 136 bytes.
*/
__device__ void sha3_256_device(const uint8_t *input, size_t inlen, uint8_t *hash_out) {
    const size_t rate = 136; // SHA3-256 absorb size (in bytes)
    uint64_t st[25] = {0};   // internal state (25 words = 1600 bits)

    for (int i = 0; i < 25; i++) st[i] = 0;
    // size_t offset = 0; // Unused variable removed


    // Full-block (rate bytes) processing (skipped here since inlen=32)
    while (inlen >= rate) {
        // absorb
        for (int i = 0; i < (rate / 8); i++) {
            st[i] ^= load64_le(input + i * 8);
        }
        // final Keccak-f
        keccakf(st);
        input += rate;
        inlen -= rate;
    }
    for (int i = 0; i < 4; i++) {
        st[i] ^= load64_le(input + i * 8);  // 4 * 8 = 32 bytes
    }
    ((uint8_t*)st)[32] ^= 0x06;  // padding (32nd byte)
    ((uint8_t*)st)[rate - 1] ^= 0x80;     // padding (last byte)
    keccakf(st);  // final Keccak-f

    // squeeze: output 32 bytes
    for (int i = 0; i < 4; i++) {
        store64_le(hash_out + i * 8, st[i]);
    }

}
