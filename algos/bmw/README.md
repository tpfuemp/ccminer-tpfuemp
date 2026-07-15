# bmw family (`-a bmw`/`bmw256`, `-a bmw512`)

Standalone Blue Midnight Wish coins (tpruvot lineage, GPLv3), relocated from
`Algo256/`.

- `bmw.cu` — `scanhash_bmw`: BMW-256 (a.k.a. `bmw256`). Uses the BMW-256 stage
  kernels in `algos/stages/`: `cuda_bmw.cu` (80-byte head: `bmw256_cpu_hash_80`,
  `bmw256_setBlock_80`, `bmw256_midstate_*`, built `--maxrregcount=76`) and
  `cuda_bmw256.cu` (`bmw256_cpu_hash_32`/`_init`/`_setTarget`, shared with the
  lyra2 family).
- `bmw512.cu` — `scanhash_bmw512`: BMW-512. Rides the shared quark BMW-512
  launcher (`quark_bmw512_cpu_*` forwarders → `algos/stages/cuda_bmw512.cu`); no
  dedicated primitive of its own.

## Layout

Relocation only (layout B). The `bmw256` primitives keep their names because
`cuda_bmw256.cu` is co-owned with the not-yet-migrated lyra2 family
(lyra2REv2/REv3); a de-brand would touch those, so it is deferred. Consumers
reach the primitives via `extern` declarations (unchanged by the move).

## Validation

Pure `git mv` + build-system repoint. Rebuild + benchmark re-validation owed.
