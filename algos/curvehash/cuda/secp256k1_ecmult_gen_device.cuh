#ifndef SECP256K1_ECMULT_GEN_DEVICE_CUH
#define SECP256K1_ECMULT_GEN_DEVICE_CUH

/*
 * Fixed-base scalar multiplication k*G on device.
 *
 * Method: 8-bit windowed fixed base, 32 windows. Precomputed table (host-built,
 * uploaded once) holds, per window j and byte value i in 1..255, the affine
 * point (i * 256^j) * G as raw 32-byte big-endian X||Y (64 bytes/entry; entry
 * i=0 is the point at infinity and is skipped). k = sum_j window_j * 256^j, so
 * k*G = sum_j (window_j * 256^j)*G = sum_j table[j][window_j]. Table is
 * 32*256*64 = 512 KB. (8-bit windows halve the point-adds vs 4-bit.)
 *
 * No blinding (a side-channel defense irrelevant to mining): the accumulator
 * starts at infinity and gej_add_ge_var handles infinity / doubling / negation,
 * so the running sum is always correct. The result must equal libsecp256k1's
 * k*G bit-for-bit (validated by the ecmult KAT vs secp256k1_ec_pubkey_create).
 *
 * Scalar byte order: k32 is 32-byte big-endian (k32[0] = MSB), matching the
 * curvehash phash / secp256k1 seckey convention. Window j (byte j from the LSB)
 * is simply k32[31 - j].
 */

#include "secp256k1_group_device.cuh"

#define SECP256K1_ECMULT_GEN_WINDOWS 32
#define SECP256K1_ECMULT_GEN_ENTRY   64  /* bytes per table entry: X[32]||Y[32] */

__device__ static void secp256k1_ecmult_gen(secp256k1_gej *r, const unsigned char *k32,
                                            const unsigned char *gtable) {
    secp256k1_ge add;
    add.infinity = 0;
    secp256k1_gej_set_infinity(r);
    for (int j = 0; j < SECP256K1_ECMULT_GEN_WINDOWS; j++) {
        int bits = k32[31 - j];      /* window j = byte j from the LSB */
        if (bits == 0) {
            continue; /* table[j][0] == infinity: additive no-op */
        }
        const unsigned char *p = gtable + (size_t)(j * 256 + bits) * SECP256K1_ECMULT_GEN_ENTRY;
        secp256k1_fe_set_b32(&add.x, p);
        secp256k1_fe_set_b32(&add.y, p + 32);
        add.infinity = 0;
        secp256k1_gej_add_ge_var(r, r, &add, NULL);
    }
}

/* k*G -> affine 32-byte big-endian X,Y. Returns 0 iff the result is infinity
 * (only when k == 0 mod n, which curvehash never hashes to). */
__device__ static int secp256k1_ecmult_gen_xy(unsigned char *x32, unsigned char *y32,
                                              const unsigned char *k32, const unsigned char *gtable) {
    secp256k1_gej rj;
    secp256k1_ge ra;
    secp256k1_ecmult_gen(&rj, k32, gtable);
    secp256k1_ge_set_gej_var(&ra, &rj);
    if (ra.infinity) {
        return 0;
    }
    secp256k1_fe_normalize_var(&ra.x);
    secp256k1_fe_normalize_var(&ra.y);
    secp256k1_fe_get_b32(x32, &ra.x);
    secp256k1_fe_get_b32(y32, &ra.y);
    return 1;
}

#endif /* SECP256K1_ECMULT_GEN_DEVICE_CUH */
