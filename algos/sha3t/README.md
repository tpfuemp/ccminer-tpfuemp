# sha3t — triple NIST SHA3-256

Guideline: `docs/coding-guideline.md`

`hash = SHA3-256^3( header80 )` — NIST FIPS 202 padding (0x06). CPU
reference / share re-verify: `sph/sha3.c` (tiny-sha3, `sha3_init/final`).

## Provenance

- `cuda_sha3t.cu` — Pkules donor kernel, already sp-style optimized when it
  arrived: first-round absorb precompute (`c_sha3t_mid[17]`/`c_sha3t_msg[6]`
  cover the constant 72 header bytes; only the nonce lane is folded
  per-thread). Launch shape retuned 2026-07-12 to 128 threads/block, 5
  blocks-per-SM, 1 nonce/thread (donor's 512/1/2 loses ~1-2%; parameterized
  via TPB/BPM/NPT defines). Saturated — no further kernel work planned.
- Since 2026-07 the round body / full permutation come from the shared
  `cuda/keccak_device.cuh` (bit-identical extraction); the sub-sm_61
  launch-shape branches were deleted per the arch floor.
- Init-time self-test: `algos/keccak/cuda_keccak_selftest.cu`, called from
  `sha3t_cpu_init`.

## Benchmarks

| Date | Card | Driver | CUDA | Intensity | Hashrate | Notes |
|---|---|---|---|---|---|---|
| 2026-07-12 | RTX 3060 | 595.95 | 11.8 | default (25) | 247.9 MH/s | baseline before shared-header migration (98% of keccak256-alone per-permutation rate) |
| 2026-07-12 | RTX 3060 | 595.95 | 11.8 | default (25) | 267.5 MH/s | after shared-header migration (+7.9%) |
| 2026-07-12 | RTX 3060 | 595.95 | 11.8 | default (25) | 256.4-261.5 MH/s | + launch shape 128/5/1 (warm card; beat donor 512/1/2 by ~1-2% in three back-to-back orderings) |
| 2026-07-12 | RTX 3060 | 595.95 | 11.8 | 25 | ~250-263 MH/s | **live, zpool sha3-256t: accepted 4/4, 0 rejects** (post-migration kernel, 128/5/1) |

Launch-shape sweep 2026-07-12 (warm card, within-sweep comparable):
**128/5/1 wins** 261.5; 256/2/2 258.5; 128/5/2 257.5; 512/1/1 257.1;
512/1/2 (donor) 256.9 — confirmed by a direct A/B pair 256.4 vs 254.1.
Same winner as sha3d. **Rejected after measuring:** `keccak_final_lane3`
truncation of pass 3 (253.6 vs 256.9 — the full permutation schedules
better here; sha3d keeps its truncation, where it measured neutral).
