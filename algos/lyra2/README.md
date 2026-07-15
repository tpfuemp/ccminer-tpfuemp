# lyra2 family (`-a lyra2`, `-a lyra2v2`, `-a lyra2z`)

Lyra2-based coins (tpruvot lineage, GPLv3), relocated from `lyra2/`.

- `lyra2RE.cu` — `scanhash_lyra2` (Lyra2RE): blake256 → keccak256 → lyra2 →
  skein256 → groestl256.
- `lyra2REv2.cu` — `scanhash_lyra2v2` (Lyra2REv2): blake256/keccak(fused
  `blakeKeccak256`) → cubehash256 → lyra2v2 → skein256 → cubehash256 → bmw256.
- `lyra2Z.cu` — `scanhash_lyra2Z` (Lyra2Z): blake256 → lyra2Z.
- `lyra2REv3.cu` — Lyra2REv3, **present but not wired** (no `algos.h` enum /
  dispatch and absent from both build systems, as before the migration).
  Relocated to preserve it; still unbuilt.

CPU references: `Lyra2.c`/`.h`, `Lyra2Z.c`/`.h`, `Sponge.c`/`.h` (the `LYRA2`
sponge), included by the dispatchers and by external consumers (allium, x21s,
evohash) via `algos/lyra2/Lyra2.h`.

## Layout

Relocation only (layout B), no de-brand:
- The Lyra2 **matrix primitives** moved to `algos/stages/`
  (`cuda_lyra2.cu` [`--maxrregcount=128`], `cuda_lyra2v2.cu`, `cuda_lyra2Z.cu`,
  and the arch-variant sidecars `cuda_lyra2*_sm{2,3,5}.cuh` +
  `cuda_lyra2_vectors.h`; the primitive `.cu`s and their `.cuh`s form one
  relative-include cluster). They keep their `lyra2*_cpu_*` symbol names because
  they are **co-owned** with the migrated allium/x21s and standalone evohash
  (which reach them via `extern`) — a de-brand is unnecessary here.
- The dispatchers + CPU references stay together in `algos/lyra2/`.
- The three dispatchers' `#include "lyra2/Lyra2.h"` and the external consumers
  (allium/x21s/evohash) were repointed to `"algos/lyra2/Lyra2.h"`; `lyra2Z.cu`'s
  relative `"Lyra2Z.h"` resolves in-dir.

This completes the entanglement opened by the Algo256 migration: the 256-bit
primitives (`blake256`/`bmw256`/`cubehash256`/`groestl256`/`skein256`) that were
kept branded there are still branded (co-owned with this lyra2 family) — a
family-wide 256-bit de-brand could now be attempted as a follow-up since both
Algo256 and lyra2 are migrated.

## Validation

Pure `git mv` + include/build-system repoint. Rebuild + benchmark re-validation
owed (lyra2 / lyra2v2 / lyra2z; plus allium / x21s / evohash which include the
relocated `Lyra2.h`).
