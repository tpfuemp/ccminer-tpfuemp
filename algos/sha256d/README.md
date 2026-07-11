# sha256d family (sha256d / sha256t / sha256csm)

Plan: `internal-docs/sha256d-cuda-optimization-plan.md` · Guideline: `internal-docs/coding-guideline.md`

## Provenance

- `cuda_sha256d.cu` — KlausT-lineage optimized double-SHA256 kernel (backported in
  `bff177c`, not tpruvot's). Contains 4-round host prehash, preextend, constant
  padding folded into K immediates, 32-nonce-per-thread ILP and last-round
  elision (`h == 0xa41f32e7` ⇔ h7 + H7 == 0, i.e. exactly 32 leading zero bits;
  GPU-side share target is fixed at diff 1, the CPU re-hash does the real
  `fulltest`). This kernel is the **donor** for plan §4b building blocks and is
  intentionally not rewritten onto the generic header primitives.
- `cuda_sha256t.cu`, `cuda_sha256csm.cu` — tpruvot 2017 lineage; since the
  2026-07 migration they consume `cuda/sha256_device.cuh`
  (`sha256_transform_full` / `sha256_final_to_target`) instead of private
  round-body copies.
- `cuda_sha256_selftest.cu` — init-time unit test of the shared header against
  OpenSSL (guideline §7 layer 1), called from `sha256d_init`/`sha256t_init`/
  `sha256csm_init`.
- SHA256Dv (Veil) is a distinct algo and lives in `algos/sha256dv/`.

## Known quirks (candidates for the step-6/7 optimization pass)

- `sha256t`/`sha256csm` kernels stage K into `__shared__ s_K[64*4]` (only 64
  words used) without a `__syncthreads()` after the fill — a long-shipped
  tpruvot pattern; replace or fix when the kernels are reworked.
- `sha256t`/`sha256csm` do no prehash of rounds 0–2 of the nonce block and no
  preextend; per the plan these come from the KlausT donor in step 7.

## Benchmarks

| Date | Card | Driver | CUDA | Algo | Intensity | Hashrate | Notes |
|---|---|---|---|---|---|---|---|
| 2026-07-12 | RTX 3060 | 595.95 | 11.8 | sha256d | default (25) | 1380.7 MH/s | baseline before shared-header migration |
| 2026-07-12 | RTX 3060 | 595.95 | 11.8 | sha256t | default (23) | 829.7 MH/s | baseline before shared-header migration |
| 2026-07-12 | RTX 3060 | 595.95 | 11.8 | sha256d | default (25) | 1397.9 MH/s | after shared-header migration (1a2a04c) |
| 2026-07-12 | RTX 3060 | 595.95 | 11.8 | sha256t | default (23) | 846.3 MH/s | after shared-header migration (1a2a04c) |
| 2026-07-12 | RTX 3060 | 595.95 | 11.8 | sha256csm | default | 1295.2 MH/s | after shared-header migration (1a2a04c) |
