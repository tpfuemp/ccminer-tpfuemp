# qubit family

Guideline: `docs/coding-guideline.md`

## Provenance

- `qubit.cu` — tpruvot/ccminer qubit (Myriadcoin-Qubit / Geocoin lineage): the
  fixed 5-stage chain luffa(80) → cubehash → shavite → simd → echo.
- `deep.cu` — deepcoin: luffa(80) → cubehash → echo.
- `luffa.cu` — luffa (Doomcoin): the single luffa(80) stage.

All three call the bare `<prim>512_cpu_*` device-launcher names through the
`algos/common/cuda_x_stages.h` bridge (the 64-byte cubehash/shavite/simd/echo
launcher TUs still live in their legacy `x11/` folders; the bridge forwards
until they de-brand). shavite uses the sp-optimised 64-byte launcher (bare
`shavite512_cpu_hash_64`), simd the sp `+20%` kernel, and echo the optimised
alexis 64-byte launcher (`echo512_cpu_hash_64`) with the tpruvot `*_compat`
variant only on arch < 500 (below the sm_61 build floor — effectively dead).

## The luffa-80 first stage

The 80-byte Luffa-512 first stage (klausT midstate precalc) was extracted from
the old branded `qubit/qubit_luffa512.cu` to the shared stage
`algos/stages/cuda_luffa512_80.cu`, de-branded to the bare
`luffa512_setBlock_80` / `luffa512_cpu_hash_80` names. The job-invariant
round-constant upload is folded into `setBlock_80` (no separate `cpu_init`,
which would otherwise collide with the 64-byte `luffa512_cpu_init`). The
branded `qubit_luffa512_*` names remain as thin forwarders for the not-yet
migrated consumers (x16/x21s/ghostrider/timetravel). No fusion is possible in
these chains — cubehash is an isolated run-of-1 and everything after it is a
boundary stage.

## Fused terminal (qubit, deep)

qubit and deep end in a **fixed** echo stage, so the terminal is folded with the
on-device target compare: `echo512_cpu_hash_64_final` writes the two best nonces
into `d_resNonce` via an atomicExch chain, eliding the echo `d_hash` store and
both `cuda_check_hash` / `cuda_check_hash_suppl` passes. The compat path
(arch < 500) keeps the tpruvot echo + `cuda_check_hash`. luffa is a single
luffa-80 stage with no echo terminal, so it keeps `cuda_check_hash`.

## Correctness note

Every GPU candidate is re-hashed on the host (`qubithash` / `deephash` /
`luffa_hash`) before submit, so a kernel/thermal glitch can only ever cause a
local reject (`result … does not validate on CPU!`), never a bad share.

## Measured rates

| algo  | card     | driver | CUDA | intensity | rate |
|-------|----------|--------|------|-----------|------|
| qubit | RTX 3060 | 595.95 | 11.8 | 19        | ~34.7 MH/s; echo-terminal fusion **live-validated** (zpool, 4/4 accepted, 0 rejects). Benchmark rate flat within thermal drift on a warm card — fusion does strictly less work, kept for correctness/consistency (cool-card interleaved A/B owed). 0 CPU-validation failures |
| deep  | RTX 3060 | 595.95 | 11.8 | 19        | ~61.6 MH/s (benchmark, echo-terminal fusion; 0 CPU-validation failures). Live owed |
| luffa | RTX 3060 | 595.95 | 11.8 | 21        | ~435 MH/s (benchmark; 0 CPU-validation failures). Live owed |
