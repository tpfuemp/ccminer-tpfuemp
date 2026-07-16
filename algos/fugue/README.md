# fugue (`-a fugue`)

Fuguecoin proof-of-work (Fugue-256), relocated from the repo root
(`fuguecoin.cpp`) as part of clearing algo sources out of the root directory.

## Layout

Relocation only — every symbol (`scanhash_fugue256`, `fugue256_hash`,
`free_fugue256`) is unchanged, so the dispatch wiring (`algos.h`
`ALGO_FUGUE256`, `miner.h`, `ccminer.cpp` — live at `case ALGO_FUGUE256`) is
untouched.

- `fuguecoin.cpp` — the host driver and CPU reference (`sph_fugue256`).

The GPU side is the shared Fugue-256 primitive in `algos/stages/`
(`cuda_fugue256.cu` — `fugue256_cpu_init` / `fugue256_cpu_setBlock` /
`fugue256_cpu_hash` / `fugue256_cpu_free`), so only the driver moved here.

- `fuguecoin.cpp` — the host driver / CPU reference.
- `cuda_fugue256.h` — the driver↔kernel interface (4 launcher declarations);
  its sole consumer is `fuguecoin.cpp`, so it was moved out of the repo root to
  live beside it (it is not build-registered — a plain included header).

Includes are project-include-dir (`sph/sph_fugue.h`, `miner.h`, `cuda_runtime.h`)
plus own-folder `"cuda_fugue256.h"`; no parent-relative (`../`) includes, so the
move is source-transparent.

## Validation

Rebuild + benchmark re-validation owed (relocation = rename; correctness follows
from the unchanged diff). The host `sph_fugue256` re-verifies GPU candidates.
