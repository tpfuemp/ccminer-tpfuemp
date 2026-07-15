# hsr (`-a hsr`)

**HShare / HSR** — the x13 chain with an extra SM3 stage inserted after echo
(tpruvot lineage, GPLv3).

```
Blake-512 (80-byte header)  →  BMW-512  →  Groestl-512  →  Skein-512  →
JH-512  →  Keccak-512  →  Luffa-512  →  CubeHash-512  →  Shavite-512  →
SIMD-512  →  Echo-512  →  SM3  →  Hamsi-512  →  Fugue-512 (terminal)
```

## Layout

- `hsr.cu` — dispatcher (`scanhash_hsr`, `hsr_hash` CPU reference). Includes
  `algos/common/cuda_x_stages.h` and calls every hashing stage by its bare
  `<prim>512` name through that header's bridge.
- `cuda_hsr_sm3.cu` — the hsr-specific mid-chain SM3 stage
  (`sm3_cuda_hash_64`); not a shared x-family stage, so it keeps its `sm3_` name
  and lives with the dispatcher.
- `sm3.c` / `sm3.h` — the CPU SM3 reference used by `hsr_hash` (GmSSL, BSD).

Migrated from `x13/` (this was the last algo living there, so `x13/` is now
empty). The first 11 stages are identical to x11/x13, so hsr reuses the shared
launchers in `algos/stages/` and the shared **register-resident fused kernel**
(`algos/common/cuda_x_fused.cu`): the consecutive fusible run
skein→jh→keccak→luffa→cubehash runs in a single launch. The order is fixed, so
the fused sequence is uploaded once at init.

## Optimization

SM3 and Echo are **mid-chain** here (SM3, Hamsi, Fugue follow Echo), so Echo
uses the plain 64-byte launcher rather than the fused-compare terminal that x11
uses. Fugue is the terminal and the best nonce is found by the shared
`cuda_check_hash` / `cuda_check_hash_suppl` pass. No further fusion: SM3 breaks
the tail into runs of 1, and the only ≥2 fusible run (skein→cubehash) is already
fused.

## Validation

Full clean `/t:Rebuild` (0 errors). Benchmark (`--benchmark`, RTX 3060, target
loosened to `0x00ff` so the CPU re-verify fires non-vacuously): **0
does-not-validate / 0 CUDA errors**. Live pool run owed.
