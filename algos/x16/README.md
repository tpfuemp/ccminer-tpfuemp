# x16 family (x16r / x16rv2 / x16s)

Guideline: `docs/coding-guideline.md`

## Provenance

- `x16r.cu` / `x16s.cu` — tpruvot 2018 lineage; the 16 stage kernels are
  dispatched per-block from the hash-order nibbles of the previous block hash
  (x16r) or a sorted permutation of them (x16s).
- `x16rv2.cu` — penfold 2019: x16r with Tiger-192 inserted before the keccak,
  luffa and sha512 stages (`tiger192_cpu_hash_64` with `zero_pad_64=1`;
  `tiger192_cpu_hash_80` when the tiger'd stage comes first).
- `cuda_x16_*.cu` — 80-byte first-stage variants (echo, fugue, shabal,
  shavite, simd) plus the alexis 64-byte echo.
- `cuda_x16_echo512_64.cu` — since the 2026-07 migration a thin wrapper over
  the alexis section of `cuda/echo512_device.cuh`
  (`echo512_hash_64_alexis`); dispatched on sm >= 500, while
  `use_compat_kernels` falls back to `x11_echo512_*`. Keeps its legacy
  `compute_50/52`-only CodeGeneration override in the vcxproj (runs via PTX
  JIT on newer cards; A/B'd as a wash on sm_86).

## Device library

The 64-byte stage implementations live in `cuda/*_device.cuh` (§3), each with
an init-time GPU self-test against its `sph_*` reference
(`cuda/xfamily_selftest.cu`, §7 layer 1): blake512, bmw512, groestl512 (quad),
skein512, jh512, keccak512, luffa512, cubehash512, shavite512, simd (kept
multi-kernel, boundary stage), echo512 (tpruvot + alexis), hamsi512, fugue512,
shabal512, whirlpool512, sha512, tiger192. The per-stage launcher TUs still
live in their legacy folders (quark/, x11/, ...) until their own families
migrate.

## Baseline (pre-fusion)

| algo   | card     | driver | CUDA | intensity | rate |
|--------|----------|--------|------|-----------|------|
| x16r   | RTX 3060 | 595.95 | 11.8 | default   | ~12.8 MH/s (benchmark, full-chain order `0123456789ABCDEF`) |
| x16rv2 | RTX 3060 | 595.95 | 11.8 | default   | ~11.6 MH/s (benchmark, full-chain order) — matches **~11.6 MH/s live** (zpool 2026-07-13, 34/34 accepted over 5 min, multiple block orders) |
| x16s   | RTX 3060 | 595.95 | 11.8 | default   | ~12.5 MH/s (benchmark; sorted order runs all 16 stages) |

**Benchmark note:** `--benchmark` used to run the degenerate hash order
`AA55555555555555` (echo, echo, skein×14). Since 2026-07-13 the benchmark
header for x16r/x16rv2 encodes the order `0123456789ABCDEF` — every stage
exactly once, deterministic — so benchmark rates are honest full-chain
numbers, comparable across runs (validated: the x16rv2 benchmark matches the
live pool rate). The CPU re-hash validation also covers all 16 stages now.
