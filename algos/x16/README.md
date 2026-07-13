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
  shavite, simd) plus the alexis 64-byte echo. Each (except simd) now sources
  its generic device code from the matching `cuda/*_device.cuh`. The launchers
  are de-branded to bare `<prim>512_setBlock_80` / `<prim>512_cuda_hash_80`
  (matching the `groestl512_*`/`jh512_*` precedent); `x16_<prim>512_*` remain as
  forwarders until the other consumers (ghostrider, x21s) migrate to the bare
  names. Only `c_PaddedMessage80` is genuinely local. `cuda_x16_simd512_80.cu`
  is de-branded the same way but has no device library (simd is deliberately
  kept multi-kernel, a fusion boundary — see below), so its FFT machinery stays
  self-contained. Fugue keeps `x16_fugue512_cpu_init`/`cpu_free` `x16_`-named
  (the bare `fugue512_cpu_init`/`free` are the 64-byte x13 fugue bridge; the
  80-byte texture lifecycle is distinct, pending the x13/x16 fugue merge). The
  hamsi/whirlpool/sha512 80-byte launchers still live in their legacy family
  folders (`x13/`, `x15/`, `x17/`) and de-brand when those families migrate.
- `cuda_x16_echo512.cu` — the 80-byte first-stage echo launcher. Its generic
  device functions (`AES_2ROUND`, `echo_round`, `cuda_echo_round_80`,
  `echo_gpu_init`) were moved into the tpruvot section of
  `cuda/echo512_device.cuh`; only the `x16_echo512_*` launcher and the
  `c_PaddedMessage80` constant remain.
- `cuda_x16_fugue512.cu` — the 80-byte first-stage fugue launcher. Its generic
  transform (`mixtab0..3`, `TIX4`/`CMIX36`/`SMIX`, `SUB_ROR*`,
  `FUGUE512_3`/`FUGUE512_F`, `FUGUE_ROL/ROR`) lives in
  `cuda/fugue512_device.cuh`; only the `x16_fugue512_*` launcher, the
  `c_PaddedMessage80` constant, and the texture-backed mixtab load (kept
  distinct from the header's `__constant__` `fugue512_load_shared` path)
  remain.
- `cuda_x16_shabal512.cu` — the 80-byte first-stage shabal launcher. The
  generic transform + constants + the 80-byte hash body (`shabal512_hash_80`)
  live in `cuda/shabal512_device.cuh` (that header `#undef`s its macros, so the
  whole 80-byte body was relocated rather than the macros exposed). The
  launcher is de-branded to the bare `shabal512_setBlock_80` /
  `shabal512_cuda_hash_80` (matching the `groestl512_*`/`jh512_*` 80-byte
  precedent); `x16_shabal512_*` remain as forwarders until ghostrider/x21s
  migrate to the bare names. Only `c_PaddedMessage80` is genuinely local.
- `cuda_x16_shavite512.cu` — the 80-byte first-stage shavite launcher. The AES
  tables, `aes_round`, `AES_ROUND_NOKEY`, `KEY_EXPAND_ELT`, `c512` and
  `shavite_gpu_init` live in `cuda/shavite512_device.cuh`; only
  `c_PaddedMessage80` is local. The launcher is de-branded to the bare
  `shavite512_setBlock_80` / `shavite512_cpu_hash_80`; `x16_shavite512_*` remain
  as forwarders until ghostrider/x21s migrate (the dead `x16_shavite512_cpu_init`
  no-op was dropped). The kernel builds the padded 128-byte block and calls the
  shared `c512` with `count=640` (the header's `count==512` branch is 64-byte
  only, skipped here — verified identical for the 80-byte path). The dead
  `c512_80` was dropped.
- `cuda_x16_echo512_64.cu` — thin launcher over the alexis section of
  `cuda/echo512_device.cuh` (`echo512_hash_64_alexis`). Its canonical host name
  is the bare `echo512_cpu_hash_64` (the optimised 64-byte echo); the tpruvot
  compat variant is `echo512_cpu_hash_64_compat` (bridge to `x11_echo512_*`),
  used only in the `use_compat_kernels` branch (arch < 500, below the sm_61
  build floor — effectively dead). `x16_echo512_cpu_hash_64` remains as a
  forwarder for the not-yet-migrated consumers (x17, skydoge, x21s,
  ghostrider); it is removed once they call the bare name. The vcxproj
  CodeGeneration was normalised from the legacy `compute_50/52`-only override
  to the project default `compute_61/75/86` (native sm_86 cubin instead of
  PTX-JIT).

## Device library

The 64-byte stage implementations live in `cuda/*_device.cuh` (§3), each with
an init-time GPU self-test against its `sph_*` reference
(`cuda/xfamily_selftest.cu`, §7 layer 1): blake512, bmw512, groestl512 (quad),
skein512, jh512, keccak512, luffa512, cubehash512, shavite512, simd (kept
multi-kernel, boundary stage), echo512 (tpruvot + alexis), hamsi512, fugue512,
shabal512, whirlpool512, sha512, tiger192. The per-stage launcher TUs still
live in their legacy folders (quark/, x11/, ...) until their own families
migrate.

## Stage-launcher naming

The migrated sources call each 64-byte stage by its bare `<prim>512_cpu_*`
name (`blake512_cpu_hash_64`, `keccak512_cpu_hash_64`, ...). The launcher TUs
that define them still live in their originating family folders and keep the
old family prefix (`quark_`/`x11_`/`x13_`/`x14_`/`x15_`/`x17_`), so a name
bridge in `cuda_x16.h` forwards the bare names to the current real symbols.
Each family drops its bridge line and renames its launcher to the bare form
when it migrates; the prefixed name then has no users left and is removed. The
`_cpu_` infix keeps these host launchers distinct from the register-resident
device primitives (`blake512_hash_64`) and the bare 80-byte launchers
(`keccak512_cuda_hash_80`).

## Fused stage runs (O1)

`cuda_x16_fused.cu` executes maximal runs of >= 2 consecutive
register-resident stages in one launch (uniform in-kernel switch over a
constant order array), keeping the 64-byte state in registers instead of
bouncing it through `d_hash` between stages. Fusible set: blake, bmw, jh,
keccak, skein, luffa, cubehash, hamsi, shabal, sha512 + tiger192 (kernel
variant with its 6KB shared table). Boundary stages (groestl quad, shavite,
simd, echo, fugue, whirlpool) keep their standalone launchers. The scanhash
computes the run structure once per hash order; x16rv2 builds an effective
sequence with tiger pseudo-stages (id 16).

The fused kernel is pinned to `__launch_bounds__(256, 2)` (128 regs, some
spill): without the min-blocks clause ptxas allocates 235 regs and occupancy
halves, eating the entire win. Fused-path unit test
(`x16_fused_device_selftest`, runs once at first order change): per-stage
single launches, an 11-stage chained launch vs the sph chain, and specific
adjacency runs (`sha512·skein·keccak`, `jh·keccak·bmw·bmw`, `bmw·bmw·sha512`).
Those last cover a real blind spot: the deterministic benchmark order leaves
bmw / hamsi / shabal / sha512 **standalone**, so a *fused* sha512 or a bmw·bmw
pair is otherwise never exercised and only appears for live hash orders.

Device-library note learned here: primitive device functions must be
**address-space agnostic** — no `__ldg` on caller pointers (fused kernels
pass register-local state; `__ldg` then faults with error 717).

## Measured rates

| algo   | card     | driver | CUDA | intensity | rate |
|--------|----------|--------|------|-----------|------|
| x16r   | RTX 3060 | 595.95 | 11.8 | default   | ~12.0 MH/s (benchmark, fused; +0.8% vs unfused ~11.9); **live-validated** (19/19 accepted, 6 blocks solved; ~15-16 MH/s warm; live orders exercised fused runs up to length 7; no invalid shares) |
| x16rv2 | RTX 3060 | 595.95 | 11.8 | default   | ~12.0 MH/s (benchmark, fused; +3% vs unfused ~11.6); **~14-15.5 MH/s live** (zpool 2026-07-13, 32/32 accepted, vs ~11.6 live unfused) |
| x16s   | RTX 3060 | 595.95 | 11.8 | default   | ~12.4 MH/s (benchmark; fused delta within thermal drift) |

**Correctness note:** every GPU candidate is re-hashed on the host before
submit, so a kernel/thermal glitch can only ever cause a local reject
(`result ... does not validate on CPU!`), never a bad share. A live run at
86 °C produced a few such rejects for one order; the same build at 62 °C ran
19/19 clean — i.e. GPU compute instability under heat, caught by the guard,
not a logic error.

**Benchmark note:** `--benchmark` used to run the degenerate hash order
`AA55555555555555` (echo, echo, skein×14). Since 2026-07-13 the benchmark
header for x16r/x16rv2 encodes the order `0123456789ABCDEF` — every stage
exactly once, deterministic — so benchmark rates are honest full-chain
numbers, comparable across runs. The CPU re-hash validation also covers all
16 stages. **For fusion the alternating bench order is the worst case** (it
minimizes run lengths); random live orders cluster fusible stages into longer
runs, which is why the live gain (~+20-30%) far exceeds the +3% bench delta.
