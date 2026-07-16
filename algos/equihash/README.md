# equihash (`-a equihash`, `-a equihash144` / `equihash144_5`)

Equihash proof-of-work, relocated from the repo root and renamed `equi/` →
`algos/equihash/` to match the algo name.

Two parameter sets share `ALGO_EQUIHASH` (the `(n,k)` + personalization is
selected at runtime, see `algos.h`):

- `equihash` — Equihash 200/9 (Zcash-style), solved by `cuda_equi.cu`.
- `equihash144` / `equihash144_5` — Equihash 144/5, solved by the Tromp solver
  `cuda_equi_tromp.cu` (built with `-DWN=144 -DWK=5 -DRESTBITS=4`).

## Layout

Relocation + folder rename only — every symbol (`scanhash_equihash`,
`free_equihash`, the verifier/stratum entry points) is unchanged, so the
dispatch wiring (`algos.h`, `miner.h`, `ccminer.cpp`) is untouched.

- `equi.cpp` — the miner driver (`scanhash_equihash` / `free_equihash`).
- `equihash.cpp` / `equihash.h` — the Equihash solver-verifier core.
- `equi-stratum.cpp` — the Equihash stratum protocol handling.
- `cuda_equi.cu` — the 200/9 GPU solver.
- `cuda_equi_tromp.cu` + `equi_miner_tromp.cuh` / `blake2b_tromp.cuh` /
  `cuda_equi_tromp.h` / `equi_tromp.h` — the Tromp 144/5 GPU solver.
- `eqcuda.hpp` — shared CUDA equihash definitions.
- `blake2/` — the vendored BLAKE2b used by the solvers (`blake2bx.cpp` is built
  with SSE on Win32).

All includes are own-folder-relative (`eqcuda.hpp`, `equihash.h`,
`blake2/blake2.h`, `blake2b_tromp.cuh`) or from the project include dirs
(`miner.h`, `cuda_helper.h`); there were no parent-relative (`../`) includes and
no external references to the old `equi/` path, so the rename is
source-transparent — no include edits were needed.

## Build

Per-file CUDA settings are preserved: `cuda_equi.cu` keeps its code-generation /
`-Xptxas -dlcm=ca -dscm=cs` options, and `cuda_equi_tromp.cu` keeps
`-DWN=144 -DWK=5 -DRESTBITS=4` + `compute_61`. On the autotools build only
`cuda_equi.cu` is compiled (the Tromp `.cu` is Windows/`ccminer.vcxproj`-only).

## Validation

Rebuild + benchmark/live re-validation owed (relocation + rename; correctness
follows from the unchanged diff). Equihash is a live, re-enabled algo in this
fork.
