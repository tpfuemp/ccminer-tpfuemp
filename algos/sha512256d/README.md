# sha512256d — double SHA-512/256

## Provenance

New CUDA port (2026-07).
CPU reference (donor): cpuminer-opt `algo/sha/sha512256d-4way.c` (scalar
`scanhash_sha512256d`, `H512_256` IV). Host skeleton copied from
`algos/sha256d/sha256d.cu`; device primitives are the new shared library
`cuda/sha512_device.cuh`.

## Consensus definition

- `hash = SHA512/256( SHA512/256(header80) truncated to 32 bytes )`,
  truncated to 32 bytes (state words 0..3, big-endian serialized).
- SHA-512/256 = FIPS 180-4 SHA-512 core (80 rounds, K512) seeded with the
  SHA-512/256 IV (`22312194FC2BF72C ...`) — **not** SHA-512 truncated with
  the plain IV.
- Nonce: 32-bit at `pdata[19]`, big-endian in the header, exactly like
  sha256d. Generic stratum path, default (LE) submit byte order.
- 80 bytes fit a single 128-byte SHA-512 block: each nonce = exactly
  2 transforms.

## Implementation notes / quirks

- The kernel screens with `cuda_swab64(q3) <= target_q3` (the share value's
  high qword in fulltest word order); the host recomputes the full double
  hash via `sph_sha512` + the IV-override trick and runs `fulltest` before
  submit — a kernel bug can only cause local rejects, never bad shares.
- Truncation to 256 bits is an output rule only: both hashes run all 80
  rounds + full feed-forward. No truncated final-round variant yet (that is
  an optimization-pass item, legal only on hash2).
- Init-time self-test (`cuda_sha512256d_selftest.cu`): FIPS "abc" KAT,
  host-vs-sph cross-check, GPU-vs-host header double hash, plus a negative
  test (flipped bit must change the digest). Logs only on failure.
- Launch: TPB 256, 1 nonce/thread, default intensity 24 (`1U<<24`).
  Per-job constants (`c_header`, `c_pre`) uploaded once per scanhash call.
- `--benchmark -D` logs every CPU-validated candidate (blue) — the only
  positive proof channel in benchmark mode (no submit; the API ACC counter
  never ticks there for any algo). Use it for every kernel A/B.

## Optimization log (2026-07-13, RTX 3060 sm_86, driver-locked clocks
   unavailable → measurement floor ±5% between process starts)

- **KEPT — per-job prehash (`sha512_prehash_split_host` +
  `sha512_transform_80_from_pre9`):** hash1 rounds 0..8 and the constant
  halves of round 9 hoisted to the host per job; kernel resumes at
  round 9's `+ w9`. Bit-exact (self-tested host+device vs full transform).
  Measured +1.7% at thermal steady state (508.9 vs 500.3 MH/s, consistent
  sign in all interleaved pairs); theoretically ~5% fewer rounds.
- **REJECTED — hash2 truncated final (q3-only, rounds 77..79 elided):**
  measured NEGATIVE (-1..-4%) in adjacent A/B pairs despite fewer rounds —
  the irregular 12-round tail hurts codegen more than 3 rounds save (same
  finding as the sha3 family). Code deleted; do not re-add without a
  measured win.
- **NO CHANGE — TPB sweep 128/256/512:** all within the ±5% noise band,
  uncorrelated with temperature; kept 256. Nonces-per-thread not swept:
  expected effect (<1%: launch overhead ~30/s, constant-cache reloads) is
  unresolvable under the noise floor — revisit only with locked clocks
  (`nvidia-smi -lgc`, needs admin).
- **REJECTED — donor kernel transcription (2026-07-13):** radifier's fully
  hand-unrolled Radiant kernel (`ccminer-radiator/cuda_rad.cu`, d40c089)
  was script-transcribed verbatim, proven bit-correct on GPU at every layer
  (host+GPU harnesses vs `hashlib.sha512_256`; in-miner `r` matched; known
  candidate `255e95a6` found and CPU-validated), then A/B-measured:
  **490.4 vs 489.4 MH/s — +0.2%, a wash.** nvcc 11.8/sm_86 already extracts
  the donor's wins from the clean library kernel. Kept the 8× smaller one.
  Fold constant if ever revisited: `lo32(q3) = r + 0x247f2d73`; donor magic
  `0xdb80d28d` ⟺ `vhash[7]==0`.
- **Benchmark-window trap (cost hours — remember):** `--benchmark` rescans
  the same 2^30-nonce window forever (work regen resets the nonce). A
  diff-1 screen has ~0.25 expected hits/window → 78% chance of permanent
  zero hits, indistinguishable from a broken kernel. Zero benchmark hits ≠
  broken; validate by widening the screen to a population with a known
  member (target 0x03 has candidate `255e95a6` for the 0x55555555 header).

## Benchmarks

| date | card | driver/CUDA | intensity | hashrate | notes |
|------|------|-------------|-----------|----------|-------|
| 2026-07-12 | RTX 3060 (sm_86) | CUDA 11.8 | 24 | ~530 MH/s | naive baseline (TPB 256, 1 nonce/thread), benchmark; 43/43 GPU candidates CPU-validated at target 0x03, 0 mismatches. 58–64 regs, 0 spills. |
| 2026-07-12 | RTX 3060 (sm_86) | CUDA 11.8 | 24 | ~510 MH/s | **live-validated** (naive kernel): zpool `sha512256d.na.mine.zpool.ca:3342`, 4/4 accepted, 0 rejects, share diffs 5.90/4.71/2.51/2.17 — confirms header build, target compare and default (LE) submit byte order. Card warm: 76 °C, 147 W. |
| 2026-07-13 | RTX 3060 (sm_86) | CUDA 11.8 | 24 | ~535 cool / ~505 hot MH/s | prehash kernel (lever A): +1.7% vs naive at matched thermal state; 44/44 benchmark candidates CPU-validated, 0 mismatches. |
| 2026-07-13 | RTX 3060 (sm_86) | CUDA 11.8 | 24 | ~537 MH/s | **live-validated** (prehash kernel): zpool `sha512256d.na.mine.zpool.ca:3342`, 2/2 accepted, 0 rejects, share diffs 12.13/7.46. Cool card (short run). |
