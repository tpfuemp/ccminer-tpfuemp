# yescrypt (`-a yescrypt`, `yescryptr8`, `yescryptr16`, `yescryptr16v2`, `yescryptr24`, `yescryptr32`)

Yescrypt memory-hard proof-of-work, relocated from the repo root (`yescrypt/`).

Six parameterizations share one GPU implementation (each has its own
`ALGO_YESCRYPT*` enum, `scanhash_yescrypt*` and `yescrypt*_hash` entry point):

- `yescrypt`, `yescryptr8`, `yescryptr16`, `yescryptr16v2`, `yescryptr24`,
  `yescryptr32` — differing in the yescrypt N/r/ROM parameters and client key.

## Layout

Relocation only — every symbol is unchanged, so the dispatch wiring (`algos.h`,
`miner.h`, `ccminer.cpp`) is untouched.

- `yescrypt.cu` — the dispatcher and CPU-reference driver for all six variants.
- `cuda_yescrypt.cu` — the GPU kernel.

The yescrypt **CPU reference** lives in the shared `sph/` tree
(`sph/yescrypt-common.c`, `sph/yescrypt-opt.c`, `sph/yescrypt-platform.h`,
`sph/yescrypt.h`) and is used only by this algo. It was left in `sph/` — the
`.cu` reaches it via the include path (`#include "sph/yescrypt.h"`), so the move
is source-transparent (no `../` includes, no external references to the old
`yescrypt/` path).

## Build

Both `.cu` are repointed in `ccminer.vcxproj` + `.filters` (keeping the
`Source Files\CUDA\yescrypt` filter). The algo was previously Windows-only; for
parity with the Windows build it was added to `Makefile.am` (the two `.cu` plus
its `sph/yescrypt-*` sources/headers).

## Validation

Rebuild + benchmark re-validation owed (relocation = rename; correctness follows
from the unchanged diff). The host `yescrypt*_hash` re-verifies every GPU
candidate before submit.
