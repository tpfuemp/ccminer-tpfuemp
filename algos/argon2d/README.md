# argon2d (`-a argon2d1000`, `-a argon2d16000`)

Memory-hard Argon2d proof-of-work, relocated from the repo root (`argon2d/`).
Two coins share this folder — they differ only in the Argon2 memory cost:

- `argon2d1000` — Zero Dynamics Cash (`m_cost = 1000` KiB)
- `argon2d16000` — Alterdot (`m_cost = 16000` KiB)

Both run Argon2d with `lanes = 1`, `t_cost = 1`, `ARGON2_VERSION_10` over the
80-byte header (password = salt = header), producing a 32-byte output.

## Layout

Relocation only — every symbol, filename and include is unchanged, so the
dispatch wiring (`algos.h`, `miner.h`, `ccminer.cpp`) is untouched.

- `argon2d.cu` — dispatchers (`scanhash_argon2d1000` / `scanhash_argon2d16000`,
  `free_*`) and the CPU reference hash used to re-verify GPU candidates.
- `argon2d_fill.cu`, `blake2b_kernels.cu` — the GPU fill and BLAKE2b kernels.
- `argon2d_kernel.h`, `cudaexception.h` — device-side headers.
- `argon2ref/` — the vendored Argon2 reference implementation. `argon2.c`,
  `core.c`, `encoding.c`, `opt.c`, `thread.c` and `blake2/blake2b.c` are built;
  `run.c` / `test.c` / `bench.c` / `genkat.c` / `ref.c` ship with the upstream
  reference but are not compiled here.

Includes are self-relative (`argon2ref/argon2.h`, `argon2d_kernel.h`,
`../argon2.h` within `argon2ref/`) or resolved from the project include dirs
(`miner.h`, `cuda_helper.h`), so the move is source-transparent. It is
registered in both `ccminer.vcxproj` and `Makefile.am` (the autotools listing
was brought to parity with the Windows build).

> Note: `rinhash/argon2d_device.cuh` is a separate, rinhash-owned Argon2d device
> header and is unrelated to this folder.

## Validation

Rebuild + benchmark re-validation owed (relocation = rename; correctness follows
from the unchanged diff). The host `argon2d*_hash` re-verifies every GPU
candidate before submit.
