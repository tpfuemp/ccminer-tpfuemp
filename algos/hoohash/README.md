# hoohash — HoohashV110 (PEPEPOW)

Relocated from the top-level `hoohash/` into `algos/hoohash/` (layout B). This was
a **relocation-only** migration: hoohash is a self-contained standalone algo with
its own device library, not part of any shared-stage family, and nothing outside
this folder consumes its symbols (only the usual `scanhash_hoohash` / `free_hoohash`
dispatcher wiring in `ccminer.cpp` / `bench.cpp` / `miner.h` / `algos.h`).

## Files

- `hoohash.cu` — scan driver (`scanhash_hoohash`, `free_hoohash`).
- `cuda_hoohashv110.cu` — the HoohashV110 device translation unit
  (`hoo_generateMatrix` / `hoo_matmul` + the on-device hash).
- `hoohash_device.cuh` — device helpers.
- `blake3_hoo_device.cuh` — bundled BLAKE3 used by the matrix step.

## ⚠️ Consensus-critical strict floating point

`cuda_hoohashv110.cu` **must** be compiled with strict FP so the on-device
transcendentals match the CPU/glibc oracle bit-for-bit. Both build systems carry
this verbatim; do not "clean it up":

- MSBuild: `FastMath=false`, `FmadOptimization=false`,
  `--fmad=false --prec-div=true --prec-sqrt=true --ftz=false`
  (plus `-allow-unsupported-compiler` / `_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH`).
- Autotools: dedicated rule `algos/hoohash/cuda_hoohashv110.o` with
  `--maxrregcount=128 --fmad=false --prec-div=true --prec-sqrt=true --ftz=false`.

Per `docs/coding-guideline.md`, this algo's KAT is re-run **per target arch**
(libdevice transcendentals are compiled per `sm_*` and can differ in the last ULP).

## Migration notes

Two **repo-root-relative** includes were repointed to the new location (they
resolve via the root include dir, not source-relative):

- `cuda_hoohashv110.cu`: `"hoohash/hoohash_device.cuh"` → `"algos/hoohash/hoohash_device.cuh"`
- `hoohash_device.cuh`: `"hoohash/blake3_hoo_device.cuh"` → `"algos/hoohash/blake3_hoo_device.cuh"`

`hoohash.cu` uses only search-path includes (`miner.h`, `cuda_helper.h`) — no edit.
