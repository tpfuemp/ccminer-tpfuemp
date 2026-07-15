# heavy family

Guideline: `docs/coding-guideline.md`

## Provenance

- `heavy.cu` — Heavycoin (`-a heavy`) and Mjollnir (`-a mjollnir`, via the same
  `scanhash_heavy`): the HEFTY1-anchored multi-hash (hefty → {sha256, keccak512,
  blake512, groestl512} → combine).
- `bastion.cu` — Bastion (`-a bastion`): a HEFTY1 prelude feeding a per-nonce
  branching x-family chain (luffa/skein/fugue/whirlpool/shabal/hamsi/echo).
- Support kernels: `cuda_hefty1.cu` (HEFTY1), `cuda_sha256.cu`,
  `cuda_heavy_{blake512,keccak512,groestl512}.cu`, `cuda_combine.cu`,
  `cuda_bastion.cu`.

## Heavy's own blake512/keccak512/groestl512 (namespaced)

heavy's blake512/keccak512/groestl512 are **not** the x-family chain primitives:
they mix the per-nonce HEFTY1 hash into the message, use a variable 80/84-byte
block size, and carry heavycoin's own padding. They previously exported the bare
`blake512_cpu_*` / `keccak512_cpu_*` / `groestl512_cpu_*` names, which collide
**by exact signature** with the names the shared quark primitives need to
de-brand to (`blake512_cpu_init`, …). They cannot reuse `cuda/*_device.cuh` (that
library exposes only a fixed 64-byte hash, not the hefty-mixed message compress),
so they were renamed to a **`heavy_` namespace** (`heavy_blake512_*`, etc.) —
`heavy.cu` is their sole consumer, and the files are `cuda_heavy_*` — freeing the
bare names (and filenames) for the family-wide de-brand. `hefty_*` was already
namespaced; `sha256_*`/`combine_*` stay bare (heavy-only, no collision).

## Known issue: heavy/mjollnir `--benchmark`

`-a heavy`/`-a mjollnir --benchmark` stalls at startup **before `scanhash_heavy`
runs** (no "Intensity set" line prints, while `bastion` — same folder, identical
scanhash init — benchmarks fine at ~14.5 MH/s). The stall is in the ccminer.cpp
heavy-specific benchmark work-generation path, is pre-existing (independent of
the namespace rename / this migration), and is unverified against a pool
(heavycoin has no live pool). Left as-is. bastion benchmarks and mines normally.

## Correctness note

The rename is a consistent symbolic change (compiles + links; byte-identical
machine code). heavy still uses its Thrust-based per-stage nonce prune as before.
