# x15 family (`-a x14`, `-a x15`, `-a whirlpool`)

The algos that lived in the old `x15/` folder, tpruvot lineage, GPLv3.

```
x14:  blake bmw groestl skein jh keccak luffa cubehash shavite simd echo hamsi fugue shabal
x15:  … x14 … + whirlpool
whirlpool (WHC):  Whirlpool ×4 (80-byte header then 3×64)
```

## Layout

- `x14.cu`, `x15.cu` — fixed-chain dispatchers. Both include
  `algos/common/cuda_x_stages.h` and call every 64-byte stage by its bare
  `<prim>512` name. The first 11 stages are identical to x11/x13, so they reuse
  the shared **register-resident fused kernel** (`algos/common/cuda_x_fused.cu`)
  for the skein→jh→keccak→luffa→cubehash run (uploaded once at init). Echo is
  mid-chain (hamsi/fugue/shabal[/whirlpool] follow), so it uses the plain
  64-byte echo launcher. x14's terminal is Shabal, x15's is Whirlpool; the best
  nonce is found by the shared `cuda_check_hash` pass.
- `whirlpool.cu` — standalone 4×Whirlpool (`-a whirlpool`, Whirlcoin). Uses the
  whirlpool stage's 80-byte fused-compare path (`whirlpool512_setBlock_80` +
  `whirlpool512_cpu_hash_80`, best nonce found on-device).
- `whirlpoolx.cu` + `cuda_whirlpoolx.cu` — `-a whirlpoolx`, **currently
  disabled** (commented out in `ccminer.cpp`, skipped in `bench.cpp`, not in the
  build). Relocated here for tidiness but not compiled; `whirlpoolx_*` names
  kept (single, dormant consumer).

## Shared stages moved to `algos/stages/` + de-branded

Two stage launchers are shared across the x14–x17/x21 families, so they moved to
`algos/stages/` and were de-branded to bare names (real symbols), with the old
prefixed names kept as thin forwarders for the not-yet-migrated consumers
(x17/skydoge/hmq17, x21s, ghostrider, evohash, bastion):

- `cuda_x14_shabal512.cu`: `x14_shabal512_cpu_*` → **`shabal512_cpu_*`**.
- `cuda_x15_whirlpool.cu`: `x15_whirlpool_cpu_*` → **`whirlpool512_cpu_*`**
  (also owns the standalone 80-byte `whirlpool512_setBlock_80` /
  `whirlpool512_cpu_hash_80`). Its T-tables header moved too
  (`cuda_whirlpool_tables.cuh`; `cuda/whirlpool512_device.cuh` include updated).
- `cuda_whirlpool512_80.cu` (was `cuda_x15_whirlpool_sm3.cu`): a second,
  self-contained Whirlpool implementation (own `mixTob*Tox` constant tables)
  that provides the **80-byte** entry points the x16-family chains consume —
  `x16_whirlpool512_init` / `x16_whirlpool512_setBlock_80` /
  `x16_whirlpool512_hash_80` (x16r/rv2/s, x21s, ghostrider). Its
  `oldwhirlpool_gpu_hash_80` is the `void*`-output overload, distinct from the
  whirlcoin target-compare kernel in `cuda_whirlpool512.cu`, so both TUs link
  together. The `x16_whirlpool512_*` names are kept until the x16 family
  migrates. (Legacy `whirlpool512_*_sm3` helpers in this TU are unreferenced.)

The bare-name bridge `#define`s in `cuda_x_stages.h` were replaced with real
declarations; the migrated x16 family / polytimos / veltor (which already called
the bare names through the bridge) now bind directly to the real symbols.

## Validation

Full clean `/t:Rebuild` (0 errors). Benchmark (`--benchmark`, RTX 3060, target
loosened so the CPU re-verify fires non-vacuously): **0 does-not-validate / 0
CUDA errors** for x14, x15 and whirlpool. Live pool run owed.
