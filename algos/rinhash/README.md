# rinhash (`-a rinhash`)

RinHash proof-of-work, relocated from the repo root (`rinhash/`).

RinHash chains three primitives: **BLAKE3 → Argon2d → SHA3-256**.

## Layout

Relocation only — every symbol (`scanhash_rinhash`) is unchanged, so the
dispatch wiring (`algos.h` `ALGO_RINHASH` / `"rinhash"`, `miner.h`,
`ccminer.cpp`) is untouched.

Only two translation units are compiled:

- `rinhash_scanhash.cpp` — the host scan driver (`scanhash_rinhash`).
- `rinhash.cu` — the GPU driver, which `#include`s the device implementations
  directly: `rinhash_device.cuh`, `argon2d_device.cuh`, `sha3-256.cu`, and
  `blake3_device.cuh` (→ `blaze3_cpu.cuh`).

The remaining files (`argon2d.cu`, `blake2b.cu`, `blake3.cu`, `sha3_256_device.cuh`,
etc.) travel with the folder but are not separate build entries — they are
`#include`d or kept as reference. All includes are own-folder-relative or from
the project include dirs (`miner.h`, `cuda_helper.h`, CUDA/thrust), with no
parent-relative (`../`) includes and no external references to the `rinhash/`
path, so the move is source-transparent.

> Note: this folder's `argon2d_device.cuh` is rinhash's own Argon2d device code,
> distinct from the `algos/argon2d/` algo folder.

## Validation

Rebuild + benchmark re-validation owed (relocation = rename; correctness follows
from the unchanged diff).
