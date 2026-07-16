# skein (`-a skein`, `-a skein2`)

The Skein-family standalone algos, relocated from the repo root as part of
clearing algo sources out of the root directory. Two live algos share this
folder (both dispatched in `ccminer.cpp`):

- `skein` (`ALGO_SKEIN`) — Skeincoin: Skein-512 over the 80-byte header, then
  SHA-256. Driver `skein.cu` (`scanhash_skeincoin`) + kernel `cuda_skeincoin.cu`
  (`skeincoin_setBlock_80` + the SHA-256 tail); the pair must stay together.
- `skein2` (`ALGO_SKEIN2`) — double-Skein (Woodcoin): two Skein-512 passes.
  Host driver `skein2.cpp` (`scanhash_skein2`), no own kernel.

## Layout

Relocation only — symbols unchanged, so the dispatch wiring (`algos.h`,
`miner.h`, `ccminer.cpp`) is untouched.

- `skein.cu` / `cuda_skeincoin.cu` — the Skeincoin driver + GPU kernel.
- `skein2.cpp` — the double-Skein driver.

Both drivers use the shared Skein-512 stage launchers already in
`algos/stages/` (`skein512_cpu_setBlock_80` / `skein512_cpu_hash_80`, and
`quark_skein512_cpu_hash_64` for skein2) plus the shared `cuda_check_hash`.
All includes are project-include-dir (`miner.h`, `sph/sph_skein.h`,
`cuda_helper.h`, `openssl/sha.h`); no parent-relative (`../`) includes and no
`#include` of these TUs (their symbols are linked), so the move is
source-transparent.

## Build

Per-file CUDA settings preserved: `skein.cu` `MaxRegCount=64`,
`cuda_skeincoin.cu` `MaxRegCount=48`, and the autotools `skein.o`
`--maxrregcount=64` rule was repointed to `algos/skein/`. All three files were
already in both build systems.

## Validation

Rebuild + benchmark re-validation owed (relocation = rename; correctness follows
from the unchanged diff). Both algos CPU-re-verify candidates before submit.
