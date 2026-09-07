// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Init-time self-test for the blake256 family (docs/coding-guideline.md
 * section 7, layer 1). Runs once per device and FAILS CLOSED via
 * cuda/selftest_gate.cuh.
 *
 * Five legs, in the order they fail:
 *   1. KAT14 - host reference against a 14-round digest produced outside this
 *      codebase. This is BLAKE-256, the SHA-3 finalist, not BLAKE2s, so there
 *      is no hashlib oracle: the constants come from a from-spec generator
 *      that validates itself against two published BLAKE-256 vectors first.
 *   2. KAT8  - the same header through the 8-round reduction blakecoin and
 *      vanilla mine. No published vector exists for it (it is not standard
 *      BLAKE-256); the oracle is the same from-spec code at fewer rounds,
 *      independent of sph/blake.c but not a standard.
 *   3. NEG   - one flipped header bit must change the digest, which proves
 *      legs 1-2 are not comparing something against itself.
 *   4. DIFF  - GPU-vs-CPU over a nonce RANGE at 14 rounds, using the two
 *      accumulators the -D path uses. The only leg that can see a nonce the
 *      mining kernel never reported, and it exercises the shipping
 *      arithmetic: blake256_gpu_checksum_14() calls the same
 *      blake256_compress_14() the mining kernel does.
 *   5. NEGD  - the differential re-run with one nonce's contribution
 *      perturbed on the host must fail. A gate whose trigger never occurs in
 *      normal operation is untested by construction.
 *
 * SCOPE: leg 4 covers the 14-round kernel only. The 8-round and decred GPU
 * paths are covered by their KATs, not by the differential, and vanilla.cu /
 * decred.cu have their own kernels which this TU does not reach.
 */

#include <string.h>

#include <miner.h>
#include <cuda_helper.h>

#include "cuda/selftest_gate.cuh"

/* Host reference and the device-side pieces, all defined in blake256.cu. */
extern "C" void blake256hash(void *output, const void *input, int8_t rounds);
extern void blake256_setBlock_selftest(const uint32_t *pdata_words, int8_t rounds);
extern void blake256_differential(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t *acc);

/* BLAKE-256 of the 80-byte header built below, from a from-spec generator
 * validated against published BLAKE-256 vectors.
 *
 * These are in the word order blake256hash() RETURNS, i.e. byte-swapped state
 * words: sph_blake256_close() writes the digest big-endian, so a
 * little-endian host reads back swab32(H[k]). The DEVICE keeps the raw H[k]
 * and swaps only for the target compare. Both orders live in shipping code,
 * so the KAT is written in the order the host reference returns. */
static const uint32_t kat_digest_14[8] = {
	0xEE127EE2u, 0x4332943Cu, 0x444438B4u, 0x365C66D8u,
	0xC4542EA5u, 0xFFC7821Cu, 0x79B4ED34u, 0x95589BE4u,
};
static const uint32_t kat_digest_8[8] = {
	0x74748716u, 0x588E028Du, 0x45D64B66u, 0xB1F9056Bu,
	0x3F7F52FAu, 0xE315ADDEu, 0x8152423Fu, 0x97AA63BCu,
};
#define KAT_NONCE 0x1234abcdu

/* The words scanhash_blake256 would hold in pdata[]. The endianness subtlety is
 * that sph_blake256 reads the message as BIG-ENDIAN words, so the values BLAKE
 * actually compresses are these raw words - which is also why the device can be
 * handed &pdata[16] un-swapped and still agree with the host. */
static void blake256_selftest_pdata(uint32_t *pd, uint32_t nonce)
{
	for (int i = 0; i < 19; i++)
		pd[i] = 0x01020304u + i;
	pd[19] = nonce;
}

/* endiandata[] as the host re-verify builds it: be32enc over all 20 words. */
static void blake256_selftest_header(uint32_t *ed, const uint32_t *pd)
{
	for (int i = 0; i < 20; i++)
		be32enc(&ed[i], pd[i]);
}

/* The host half of the differential, kept here rather than shared with
 * scanhash's copy so a fault injected for leg 5 cannot reach the mining path. */
static void blake256_selftest_cpu_acc(const uint32_t *pd, uint32_t startNonce,
	uint32_t count, uint64_t *cpu, uint32_t perturb_at)
{
	uint32_t _ALIGN(64) ed[20], vhash[8], td[20];

	cpu[0] = cpu[1] = 0;
	memcpy(td, pd, sizeof(td));
	for (uint32_t i = 0; i < count; i++) {
		const uint32_t nonce = startNonce + i;
		td[19] = nonce;
		blake256_selftest_header(ed, td);
		blake256hash(vhash, ed, 14);
		/* blake256hash() returns the digest byte-swapped relative to the state
		 * words (sph writes it big-endian; a little-endian host reads back
		 * swab32(H[k])). The device accumulates raw h[6]/h[7], so un-swap here
		 * or every nonce mismatches. */
		uint64_t w = ((uint64_t) swab32(vhash[7]) << 32) | swab32(vhash[6]);
		if (i == perturb_at)
			w ^= 1ull; // leg 5 only: must break the comparison
		cpu[0] ^= w;
		cpu[1] ^= w * (2ull * (uint64_t) nonce + 1ull);
	}
}

__host__
bool blake256_device_selftest(int thr_id)
{
	/* Per-device, not per-process: the GPU leg only proves the device that ran
	 * it, so a second card must be tested on its own. */
	static bool tested[MAX_GPUS] = { 0 }, passed[MAX_GPUS] = { 0 };
	if (tested[thr_id]) return passed[thr_id];
	tested[thr_id] = true;

	uint32_t _ALIGN(64) pd[20], ed[20], dig[8];

	blake256_selftest_pdata(pd, KAT_NONCE);
	blake256_selftest_header(ed, pd);

	// --- legs 1 and 2: host reference vs the external oracle, both round counts ---
	blake256hash(dig, ed, 14);
	const bool kat14_ok = (memcmp(dig, kat_digest_14, sizeof(kat_digest_14)) == 0);

	blake256hash(dig, ed, 8);
	const bool kat8_ok = (memcmp(dig, kat_digest_8, sizeof(kat_digest_8)) == 0);

	// --- leg 3: negative test, one flipped header bit must move the digest ---
	uint32_t _ALIGN(64) ed_bad[20], dig_bad[8];
	memcpy(ed_bad, ed, sizeof(ed_bad));
	ed_bad[0] ^= 0x00000001u;
	blake256hash(dig_bad, ed_bad, 14);
	const bool neg_ok = (memcmp(dig_bad, kat_digest_14, sizeof(kat_digest_14)) != 0);

	/* Legs 4 and 5 need the job words the kernel reads, and they overwrite them,
	 * so scanhash must upload the real ones afterwards - it does, once per job. */
	const uint32_t start = 0x00010000u, count = 2048u;
	uint64_t gpu[2] = { 0, 0 }, cpu[2] = { 0, 0 }, cpu_bad[2] = { 0, 0 };

	blake256_setBlock_selftest(pd, 14);
	blake256_differential(thr_id, count, start, gpu);
	if (gpu[0] == 0 && gpu[1] == 0)
		selftest_cuda_fault(); // could not run != produced a wrong word

	// --- leg 4: GPU == CPU over the range ---
	blake256_selftest_cpu_acc(pd, start, count, cpu, UINT32_MAX);
	const bool diff_ok = (gpu[0] == cpu[0] && gpu[1] == cpu[1]);

	// --- leg 5: the same comparison with one host word perturbed must fail ---
	blake256_selftest_cpu_acc(pd, start, count, cpu_bad, count / 2);
	const bool negd_ok = !(gpu[0] == cpu_bad[0] && gpu[1] == cpu_bad[1]);

	passed[thr_id] = kat14_ok && kat8_ok && neg_ok && diff_ok && negd_ok;
	if (!passed[thr_id]) {
		gpulog(LOG_ERR, thr_id, "blake256 self-test FAILED"
			" (kat14 %d kat8 %d neg %d diff %d negd %d)",
			(int) kat14_ok, (int) kat8_ok, (int) neg_ok, (int) diff_ok, (int) negd_ok);
		/* A failing KAT must show what it got, or the next person re-derives the
		 * oracle from scratch to find out. */
		if (!kat14_ok || !kat8_ok) {
			uint32_t _ALIGN(64) g14[8], g8[8];
			blake256hash(g14, ed, 14);
			blake256hash(g8, ed, 8);
			gpulog(LOG_ERR, thr_id, "  host 14r %08x %08x %08x %08x %08x %08x %08x %08x",
				g14[0], g14[1], g14[2], g14[3], g14[4], g14[5], g14[6], g14[7]);
			gpulog(LOG_ERR, thr_id, "  want 14r %08x %08x %08x %08x %08x %08x %08x %08x",
				kat_digest_14[0], kat_digest_14[1], kat_digest_14[2], kat_digest_14[3],
				kat_digest_14[4], kat_digest_14[5], kat_digest_14[6], kat_digest_14[7]);
			gpulog(LOG_ERR, thr_id, "  host  8r %08x %08x %08x %08x %08x %08x %08x %08x",
				g8[0], g8[1], g8[2], g8[3], g8[4], g8[5], g8[6], g8[7]);
			gpulog(LOG_ERR, thr_id, "  want  8r %08x %08x %08x %08x %08x %08x %08x %08x",
				kat_digest_8[0], kat_digest_8[1], kat_digest_8[2], kat_digest_8[3],
				kat_digest_8[4], kat_digest_8[5], kat_digest_8[6], kat_digest_8[7]);
		}
		if (!diff_ok) {
			gpulog(LOG_ERR, thr_id, "  gpu %016llx / %016llx",
				(unsigned long long) gpu[0], (unsigned long long) gpu[1]);
			gpulog(LOG_ERR, thr_id, "  cpu %016llx / %016llx",
				(unsigned long long) cpu[0], (unsigned long long) cpu[1]);
			if (gpu[0] == cpu[0])
				gpulog(LOG_ERR, thr_id, "  pairs agree but their nonces do not:"
					" an index/permutation bug");
		}
	}
	return selftest_gate(thr_id, "blake256", passed[thr_id]);
}
