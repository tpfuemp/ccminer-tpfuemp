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

## Shape

`s_cost = 128` → a 128 KB buffer per hash, one thread per hash, as a per-thread
stack array: 131,408 B stack frame, 71 registers, 0 spills (sm_86). The `delta`
index stream comes from the first 32 header bytes only, so it is nonce-independent
and shared by every thread; with local memory being interleaved per thread, a warp
reads the same block index in lockstep and the accesses coalesce.

## Rates

| card | run | rate |
|---|---|---|
| RTX 3060, `-i 14` | live (zpool) | ~39–40.6 kH/s at 62 °C, 162 W, 1890 MHz |
| RTX 3060, `-i 14` | `--benchmark` | ~38 kH/s cold → ~35 kH/s at 87 °C, 1727 MHz |

Higher intensities only add heat. `--benchmark` self-throttles and reads ~10% low;
A/Bs need interleaved pairs in one session.

## Move fix

`balloon_scan.cpp` used **parent-relative** includes (`"../miner.h"`,
`"../cuda_helper.h"`) that resolved to the repo root from the old location.
Those were changed to the include-path forms (`"miner.h"`, `"cuda_helper.h"`),
matching the sibling files (`balloon.cpp`, `cuda_balloon.cu`) so the move is
source-transparent. All other includes are own-folder-relative (`balloon.h`,
`sha256.h`) or from the project include dirs, and no external file referenced
the `balloon/` path.

## Validation

`--benchmark` loosens `ptarget[7]` to `0x00ffff` (the usual `0x0000ff` yields ~0
hits at a kH/s rate) and `balloon_cpu_hash` returns `UINT32_MAX` for "no candidate",
so a nonce whose host re-hash disagrees is reported rather than silently skipped.
`--debug` shows the validated hits: 30 hits, 0 `does not validate` in 90 s.
Live-verified on zpool (RTX 3060).

Dead code removal (`conv_onethread`, the `LOWMEM` branch and its `sbufs` argument,
unused `__constant__`s and their uploads, debug scaffolding) left the kernel's SASS
unchanged instruction-for-instruction, sm_61/75/86.
