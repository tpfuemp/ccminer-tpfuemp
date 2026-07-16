# balloon (`-a balloon`)

Balloon memory-hard hashing (Boneh–Corrigan-Gibbs–Schechter), built on SHA-256,
relocated from the repo root (`balloon/`).

## Layout

Relocation only — every symbol (`scanhash_balloon`, the balloon core in
`balloon.cpp`) is unchanged, so the dispatch wiring (`algos.h`, `miner.h`,
`ccminer.cpp`) is untouched.

- `balloon.cpp` / `balloon.h` — the Balloon hash core.
- `balloon_scan.cpp` — the CUDA scan driver (`scanhash_balloon`).
- `cuda_balloon.cu` — the GPU kernel.
- `sha256-ref.c` / `sha256.h` — the reference SHA-256 used by Balloon.
- `sha256-sse.c` — an SSE SHA-256 variant that ships with the sources but is
  not compiled by either build system (left as-is).

## Move fix

`balloon_scan.cpp` used **parent-relative** includes (`"../miner.h"`,
`"../cuda_helper.h"`) that resolved to the repo root from the old location.
Those were changed to the include-path forms (`"miner.h"`, `"cuda_helper.h"`),
matching the sibling files (`balloon.cpp`, `cuda_balloon.cu`) so the move is
source-transparent. All other includes are own-folder-relative (`balloon.h`,
`sha256.h`) or from the project include dirs, and no external file referenced
the `balloon/` path.

## Validation

Rebuild + benchmark re-validation owed (relocation = rename plus the include
fix; correctness follows from the diff).
