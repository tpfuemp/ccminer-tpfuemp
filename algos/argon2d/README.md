# argon2d (`-a argon2d500`, `-a argon2d1000`, `-a argon2d4096`, `-a argon2d16000`)

Memory-hard Argon2d proof-of-work. Four coins share this folder — a single
Argon2d over the 80-byte header (password = salt = header, outlen = 32),
differing only in the Argon2 cost parameters and version:

| algo         | coin               | m_cost (KiB) | lanes | t_cost | version |
|--------------|--------------------|--------------|-------|--------|---------|
| argon2d500   | Dynamic (DYN)      | 500          | 8     | 2      | 0x10    |
| argon2d1000  | Zero Dynamics Cash | 1000         | 8     | 2      | 0x10    |
| argon2d4096  | Argentum / Myriad  | 4096         | 1     | 1      | 0x13    |
| argon2d16000 | Alterdot           | 16000        | 1     | 1      | 0x10    |

## Layout

- `argon2d.cu` — the per-coin variant table (`argon2d_variant`), CPU reference
  hashes (`argon2d*_hash`, used to re-verify every GPU candidate before
  submit), the shared `scanhash_argon2d` driver and the init-time self-test.
- `argon2d_fill.cu`, `blake2b_kernels.cu` — the GPU fill and BLAKE2b kernels.
- `argon2d_kernel.h`, `cudaexception.h` — device-side headers.
- `argon2ref/` — the vendored Argon2 reference implementation. `argon2.c`,
  `core.c`, `encoding.c`, `opt.c`, `thread.c` and `blake2/blake2b.c` are built;
  `run.c` / `test.c` / `bench.c` / `genkat.c` / `ref.c` ship with the upstream
  reference but are not compiled here.

> Note: `rinhash/argon2d_device.cuh` is a separate, rinhash-owned Argon2d device
> header and is unrelated to this folder.

## Variant coexistence (runtime geometry)

The kernels used to be compiled for exactly one `(m_cost, lanes, t_cost,
version)` tuple via `ALGO_*` enum constants, so only one coin per build could
be GPU-correct. The full parameter set (m_cost, lanes, passes, version,
total/segment blocks) is now passed to `argon2_initialize` / `argon2_finalize`
as runtime kernel arguments — the heavy `argon2_fill` kernel always took it
that way — so all variants are GPU-correct in one binary at no measurable cost
(the parameterized kernels are a tiny fraction of the pipeline).

Two device-code subtleties this required:

- **lanes < 8**: the fill kernel's final-block epilogue (XOR of each lane's
  last-column block, staged in the `c[8][6]` shared cache) hardcoded 8 lanes
  and 256 threads. It now XORs `lanes` rows with a stride loop over the
  block's `lanes*32` threads — bit-identical for lanes=8, and it is what made
  the lanes=1 coins (argon2d4096, argon2d16000) work on GPU at all.
  argon2d16000 had never produced a valid GPU share in this fork before this
  fix (its false candidates were silently discarded by the host re-verify).
- **Argon2 version 0x13** (argon2d4096): the version number is hashed into H0
  by `argon2_initialize`, so it must be per-variant. The v1.0-vs-v1.3
  difference inside the fill core (XOR-overwrite on passes after the first)
  never triggers at t_cost=1, so the fill kernel serves both versions; a
  future t_cost>1 v0x13 variant would need the version-conditional XOR there
  (see `argon2ref/core.c` `fill_block` `with_xor`).

## Provenance

Donor: `duality-solutions/Dynamic-GPU-Miner-Nvidia` (this fork's `argon2d/`
descends from it; `argon2d_fill.cu` was byte-identical before the
parameterization). Dynamic's chain params verified against the donor;
argon2d4096's (4096/1/1/v0x13) against cpuminer-opt's
`algo/argon2d/argon2d-gate.c` (`register_argon2d4096_algo`).

## Validation

- Init-time self-test: `argon2d500_hash`, `argon2d1000_0dync_hash` and
  `argon2d4096_hash` over a fixed 80-byte header must match digests from the
  independent official argon2 library (argon2-cffi, type=D, per-variant
  version), plus a one-bit-flip negative test proving the comparison isn't
  vacuous.
- The host `argon2d*_hash` re-verifies every GPU candidate before submit; a
  candidate that fails re-verify logs "does not validate on CPU!".
- Loosened-target benchmark battery (RTX 3060, 2026-07-18): every GPU
  candidate CPU-validated, zero mismatches — argon2d4096 499/499 (~13 kH/s),
  argon2d500 803/803, argon2d1000 763/763, argon2d16000 152/152 (~3.1 kH/s;
  first time this variant validated on GPU). Steady-state benchmark rates:
  argon2d500 ~196 kH/s, argon2d1000 ~88.5 kH/s.
- Live-validated on zpool (2026-07-18, RTX 3060): argon2d500 accepted 19/19,
  0 rejects, ~196 kH/s; argon2d4096 accepted 3/3, 0 rejects, ~13.7 kH/s
  (across a mid-run coin switch, so two distinct chains); argon2d16000
  accepted 8/8, 0 rejects, ~3.2 kH/s (the first live shares this variant ever
  produced on GPU in this fork).
