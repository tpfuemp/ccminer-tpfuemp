# myriadgroestl (`-a myr-gr`)

Myriad-Groestl proof-of-work, relocated from the repo root as part of clearing
algo sources out of the root directory. Groestl-512 over the 80-byte header
followed by SHA-256.

## Layout

Relocation only — every symbol (`scanhash_myriad`, `free_myriad`, and the device
`myriadgroestl_cpu_*`) is unchanged, so the dispatch wiring (`algos.h`
`ALGO_MYR_GR`, `miner.h`, `ccminer.cpp` — live at `case ALGO_MYR_GR`) is
untouched.

- `myriadgroestl.cpp` — host driver / CPU reference (`sph_groestl` + `SHA256`).
- `cuda_myriadgroestl.cu` — the GPU kernels (`myriadgroestl_gpu_hash_quad` +
  `_gpu_hash_sha`) and launchers (`myriadgroestl_cpu_init/setBlock/hash/free`).

It is self-contained and **not** coupled to `cuda_groestlcoin.cu` — its symbols
are its own (`myriadgroestl_*`). The GPU Groestl core comes from the shared
`cuda/groestl512_device.cuh` (quad-warp bitsliced Groestl), included via the
project include path. Includes are project-include-dir (`miner.h`,
`cuda_helper.h`, `sph/sph_groestl.h`, `openssl/sha.h`) with no parent-relative
(`../`) includes, so the move is source-transparent.

## Validation

Rebuild + benchmark re-validation owed (relocation = rename; correctness follows
from the unchanged diff). The host reference re-verifies GPU candidates.
