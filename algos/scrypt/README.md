# scrypt (`-a scrypt`, `-a scrypt-jane`)

Scrypt and scrypt-jane, relocated from the repo root (`scrypt/` folder plus the
`scrypt.cpp` / `scrypt-jane.cpp` dispatchers) into a single self-contained
`algos/scrypt/`.

> **Status: dead-dispatched in this fork.** `ALGO_SCRYPT` / `ALGO_SCRYPT_JANE`
> and their names still exist in `algos.h`, and `scanhash_scrypt` /
> `scanhash_scrypt_jane` are still defined here, but `ccminer.cpp` has no
> `case ALGO_SCRYPT: rc = scanhash_scrypt(...)` in its main dispatch switch
> (only the auxiliary nonce/min-max/n-factor switches reference the enums). So
> the algo cannot currently mine — re-enabling it would mean restoring the
> dispatch case (compare with how equihash was re-enabled). This relocation is
> housekeeping and changes no behaviour.

## Layout

Relocation only — every symbol is unchanged.

- `scrypt.cpp` — scrypt host driver (`scanhash_scrypt`).
- `scrypt-jane.cpp` — scrypt-jane host driver (`scanhash_scrypt_jane`).
- `salsa_kernel.cu` / `blake.cu` / `keccak.cu` / `sha256.cu` — the CUDA TUs.
  On Windows only these are compiled; the per-architecture kernels
  (`fermi_kernel.cu`, `kepler_kernel.cu`, `nv_kernel.cu`, `nv_kernel2.cu`,
  `test_kernel.cu`, `titan_kernel.cu`) are `#include`d into `salsa_kernel.cu`.
  On the autotools build they are compiled as separate TUs (with a special
  `compute_35` rule for `titan_kernel`).
- `code/` — the scrypt-jane portable/chacha reference headers.

## Move fixes

The two dispatchers referenced the folder with a `scrypt/` prefix
(`#include "scrypt/salsa_kernel.h"`, `#include "scrypt/code/..."`); now that they
live in the folder those were changed to own-folder paths (`"salsa_kernel.h"`,
`"code/..."`). All in-folder includes were already own-folder/`code/`-relative,
and there were no `../` or external references, so the move is otherwise
source-transparent. The `SCRYPT_KECCAK512` / `SCRYPT_CHACHA` /
`SCRYPT_CHOOSE_COMPILETIME` defines are global project settings and are
unaffected.

## Build

`ccminer.vcxproj` + `.filters` repoint the entries that exist there
(`salsa_kernel.h` + `blake`/`keccak`/`sha256`/`salsa_kernel` `.cu`); the host
`.cpp` were already absent from the Windows project and stay that way.
`Makefile.am` repoints the full source list and the `titan_kernel.o` rule.

## Validation

Rebuild owed. Because the algo is dead-dispatched, a `--benchmark`/live run is
not possible without first restoring the `ccminer.cpp` dispatch case.
