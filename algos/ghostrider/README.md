# ghostrider (`-a ghostrider`)

GhostRider (Raptoreum) proof-of-work, relocated from the repo root
(`ghostrider/`).

The chain runs a nonce-determined order of **15 core 512-bit hash algos**
interleaved with **3 CryptoNight-v1 rounds**, arranged in three groups of
(5 core + 1 CN). Both the core order (15 distinct algos) and the CN triple
(the first 3 of 6 CN-v1 variants) are selected per block from the input.

## Layout

Relocation only — every symbol (`scanhash_ghostrider`, `free_ghostrider`) is
unchanged, so the dispatch wiring (`algos.h` `ALGO_GHOSTRIDER` / `"ghostrider"`,
`miner.h`, `ccminer.cpp`) is untouched.

- `ghostrider.cu` — the full GhostRider driver: core/CN order derivation, the
  512-bit core-stage pipeline, GPU orchestration and CPU re-verify.

Dependencies are resolved by include-path and by link-time externs, so the move
is source-transparent:
- the shared x-family stage library via `#include "algos/common/cuda_x_stages.h"`
  and the `sph/*` headers;
- the CryptoNight-v1 GPU/CPU paths via `extern "C"` to the `*_gr` functions in
  `crypto/cryptonight-core.cu`, `crypto/cryptonight-extra.cu` and
  `crypto/cryptonight-cpu.cpp` (unmoved; linked as before).

There are no parent-relative (`../`) includes and no external references to the
old `ghostrider/` path.

## Build

Registered in both `ccminer.vcxproj` and (newly, for completeness)
`Makefile.am`; the CryptoNight `crypto/*` dependencies were already in both
build systems.

## Validation

Rebuild + benchmark/live re-validation owed (relocation = rename; correctness
follows from the unchanged diff). GhostRider is a live-validated algo in this
fork.
