# balloon (`-a balloon`)

Balloon memory-hard hashing (Boneh-Corrigan-Gibbs-Schechter), built on SHA-256,
relocated from the repo root (`balloon/`).

## Layout

Relocation only  -  every symbol (`scanhash_balloon`, the balloon core in
`balloon.cpp`) is unchanged, so the dispatch wiring (`algos.h`, `miner.h`,
`ccminer.cpp`) is untouched.

- `balloon.cpp` / `balloon.h`  -  the Balloon hash core.
- `balloon_scan.cpp`  -  the CUDA scan driver (`scanhash_balloon`).
- `cuda_balloon.cu`  -  the GPU kernel.
- `sha256-ref.c` / `sha256.h`  -  the reference SHA-256 used by Balloon.
- `balloon_selftest.cpp`  -  the init-time consensus gate (see Validation).

## Shape

`s_cost = 128` gives a 128 KB buffer per hash, one thread per hash, held as a
per-thread stack array: 131,264 B stack frame, 111 registers on sm_86 (128 on
sm_61/75, where the project register cap binds), 0 spills. The `delta` index stream
derives from the first 32 header bytes only, so it is nonce-independent and shared by
every thread; local memory is interleaved per thread, so a warp reads the same block
index in lockstep and the accesses coalesce.

The index stream itself is AES-128-CTR (OpenSSL), which is why libcrypto is a link
dependency; everything else is SHA-256.

## Rates

| card | run | rate |
|---|---|---|
| RTX 3060, `-i 14` | live | ~35-41 kH/s, 160 W, 99-100% GPU utilisation |
| RTX 3060, `-i 14` | `--benchmark` | ~38 kH/s cold, ~35 kH/s once hot (1727 MHz) |

Higher intensities add no throughput - the card is already saturated at the
default - and they do add launch duration, because the kernel is not chunked.
One launch runs `batch / hashrate` seconds: measured at 0.43 s for `-i 14` and
0.87 s at `-i 15`, so about 1.7 s at `-i 16` and 3.5 s at `-i 17`. Those are
cool-card figures, and sustained mining throttles the part by roughly 18%, which
puts `-i 16` at ~2.0 s once hot.

That matters twice. Past about two seconds Windows resets a display-attached GPU,
and because a launch cannot be interrupted the same figure is how long the miner
keeps hashing a job the pool has already replaced. `scanhash_balloon` times its
first launch and warns if it lands in that range. Note a short test does not prove
an intensity is safe: the throttled figure is the one that counts.

## Move fix

`balloon_scan.cpp` used **parent-relative** includes (`"../miner.h"`,
`"../cuda_helper.h"`) that resolved to the repo root from the old location.
Those were changed to the include-path forms (`"miner.h"`, `"cuda_helper.h"`),
matching the sibling files (`balloon.cpp`, `cuda_balloon.cu`) so the move is
source-transparent. All other includes are own-folder-relative (`balloon.h`,
`sha256.h`) or from the project include dirs, and no external file referenced
the `balloon/` path.

## Validation

`--benchmark` loosens `ptarget[7]` to `0x00ffff` (the usual `0x0000ff` yields ~0
hits at a kH/s rate) and `balloon_cpu_hash` returns `UINT32_MAX` for "no candidate",
so a nonce whose host re-hash disagrees is reported rather than silently skipped.
`--debug` shows the validated hits (`found =>`); release builds print nothing when a
candidate passes, so a silent run is ambiguous without it.

`balloon_selftest.cpp` runs at init and **refuses to start on failure**. Four legs: a
pinned known-answer vector for the CPU reference; the real launcher must return the
lowest passing nonce over a span, as enumerated on the host; re-running with
`max_nonce` below that winner must exclude it; and perturbing the device input by one
bit must make the check fail. The screen threshold is taken from the enumerated data
rather than fixed, so the test cannot go vacuous, and because the launcher returns a
nonce rather than a digest the check can also detect a *missed* nonce.

Live-verified: 53/53 accepted, 0 rejected, 12 blocks, at a share difficulty of 0.0001
- far below 1, which is the regime that exercises the GPU-side target screen hardest.

Dead code removal (`conv_onethread`, the `LOWMEM` branch and its `sbufs` argument,
unused `__constant__`s and their uploads, debug scaffolding) left the kernel's SASS
unchanged instruction-for-instruction, sm_61/75/86.
