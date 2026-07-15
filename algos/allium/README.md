# allium (`-a allium`)

Lyra2RE-family chain for **Garlicoin (GRLC)** — tpruvot 2018, GPLv3.

```
Blake-256 (80-byte header, 14 rounds)  →  Keccak-256 (32)  →  LYRA2  →
CubeHash-256 (32)  →  LYRA2  →  Skein-256 (32)  →  Groestl-256 (32, terminal)
```

## Layout

- `allium.cu` — dispatcher (`scanhash_allium`, `allium_hash` CPU reference).

The stage launchers are **not** allium-owned — they are the shared Lyra2RE
primitives and stay where the lyra2 family keeps them:

- `Algo256/cuda_blake256.cu` — Blake-256 setBlock plus the fused
  `blakeKeccak256_cpu_hash_80` (Blake + Keccak over the 80-byte header in one
  launch), the Skein-256 and CubeHash-256 stages, and the Groestl-256 terminal
  live in `Algo256/`.
- `lyra2/` — the `lyra2_cpu_hash_32` sponge stage (matrix sized by SM: the
  full 8×8×3×4 matrix on sm_50, the packed 4×4 layout above it) and
  `lyra2_cpu_init`.

So this migration is a **relocation of the dispatcher only** — the file moved
from the repo root to `algos/allium/`; no de-brand or fusion, because the
primitives are co-owned with `-a lyra2re`/`-a lyra2rev2` and must keep their
current names and locations.

## Optimization

None here. The Blake+Keccak head is already fused (`blakeKeccak256_cpu_hash_80`)
and the Groestl-256 terminal carries its own on-device best-nonce compare
(`groestl256_cpu_hash_32` + `groestl256_getSecNonce`). Any further work belongs
in the shared lyra2-family primitives, not in this dispatcher.

## Validation

Full clean `/t:Rebuild` (0 errors). Benchmark (`--benchmark`, RTX 3060, target
loosened to `0x00FF` so the CPU re-verify fires non-vacuously): **0
does-not-validate / 0 CUDA errors**. Live pool run owed.
