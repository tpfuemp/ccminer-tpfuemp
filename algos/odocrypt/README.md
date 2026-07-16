# odocrypt (`-a odocrypt`)

OdoCrypt (DigiByte's Odo algorithm), relocated from the repo root (`odocrypt/`).
Odo is a periodically-regenerated cipher (the key/permutation rotates on a fixed
schedule) intended to resist fixed-function ASICs.

## Layout

Relocation only — every symbol (`scanhash_odocrypt`, the Odo cipher host
reference) is unchanged, so the dispatch wiring (`algos.h` `ALGO_ODO` /
`"odocrypt"`, `miner.h`, `ccminer.cpp`) is untouched.

- `cuda_odocrypt.cu` — the GPU kernel and scan driver.
- `odocrypt_host.cpp` — the Odo cipher host reference (CPU re-verify).
- `odocrypt.h` — shared Odo definitions.

## Move fix

`cuda_odocrypt.cu` used **parent-relative** includes (`"../miner.h"`,
`"../cuda_helper.h"`) that resolved to the repo root from the old location;
after the move they would have pointed at the non-existent `algos/miner.h`, so
they were changed to the include-path forms (`"miner.h"`, `"cuda_helper.h"`).
`odocrypt.h` is included own-folder-relative and there were no external
references to the old `odocrypt/` path, so the move is otherwise
source-transparent.

## Validation

Rebuild + benchmark re-validation owed (relocation = rename plus the include
fix; correctness follows from the diff). Submit byte-order for Odo is the known
sensitive point (see the odo port notes) — worth a live-share check.
