# x17 family (`-a x17`, `-a hmq1725`, `-a skydoge`)

Three algos that share the x17 stage set (tpruvot lineage, GPLv3).

```
x17 (fixed 17-stage chain):
Blake-512 (80-byte header)  →  BMW-512  →  Groestl-512  →  Skein-512  →
JH-512  →  Keccak-512  →  Luffa-512  →  CubeHash-512  →  Shavite-512  →
SIMD-512  →  Echo-512  →  Hamsi-512  →  Fugue-512  →  Shabal-512  →
Whirlpool-512  →  SHA-512  →  Haval-256 (terminal)
```

- `x17.cu` — the fixed 17-stage chain (`scanhash_x17`, `x17hash` CPU reference).
- `hmq17.cu` — HMQ1725, a **per-nonce branching** variant (each stage picks one
  of two hashes by a data bit, walked through branch/merge compaction buffers).
  Not a fixed chain, so it is **not fusible** — relocation only.
- `skydoge.cu` — a fixed-chain x17 variant (yiimp/cpuminer-opt lineage).

## Layout

Each dispatcher includes `algos/common/cuda_x_stages.h` and calls **every stage
by its bare `<prim>512` name** through that header's bridge (no `quark_`/`x11_`/
`x13_`/`x14_`/`x15_`/`x16_` prefixes remain). All stage launchers are the shared
ones in `algos/stages/`. Where the bare name maps to the optimised launcher this
also adopts it: 64-byte Shavite is the sp kernel (3-arg), the 64-byte Echo is the
alexis kernel, and the old combined `luffaCubehash` launch is split into the bare
`luffa512` + `cubehash512` stages (all bit-identical output, proven family-wide).

The two x17-specific stages moved to `algos/stages/` during this migration and
were **de-branded** to their bare `<prim>` names (real symbols):

- `cuda_sha512.cu` — 64-byte SHA-512 stage (`sha512_cpu_*`) plus the 80-byte
  `x16_sha512` head variant used by the x16 family. Built with
  `--maxrregcount=80`.
- `cuda_haval256.cu` — the Haval-256 terminal stage (`haval256_cpu_*`;
  `outlen` selects the 256- or 512-bit tail write).

Every caller (x17/hmq1725/skydoge here, and x21s) now calls these by their bare
names, so the transitional `x17_sha512_*` / `x17_haval256_*` forwarders were
removed — the de-brand is complete.

## Optimization

Relocation + de-brand only — no stage fusion in this pass. The fixed x17 chain
has the same skein→jh→keccak→luffa→cubehash fusible run as x13/x11 and could
adopt the shared register-resident fused kernel; that is left as a follow-up.
Haval is the terminal and the best nonce is found by the shared
`cuda_check_hash` / `cuda_check_hash_suppl` pass.

## Validation

The initial relocation (branded calls) was benchmark- and live-validated:
x17 ~13.2 MH/s (debug run: `found =>` fires ~24×/25s, every candidate passes the
`x17hash` CPU re-verify), hmq1725 ~6.36 MH/s, skydoge ~10.4 MH/s, all **0
does-not-validate / 0 CUDA errors** at `ptarget[7]=0x00ff`; **skydoge
live-confirmed on zpool (skydoge.na.mine.zpool.ca:7091): 4/4 accepted, 0
rejects**.

The subsequent call-site de-brand (sp shavite / alexis echo / split
luffa+cubehash) is byte-identical by construction but changes which kernels run,
so a fresh clean `/t:Rebuild` + benchmark re-validation is **owed** before it is
considered proven. x17/hmq1725 own live runs also owed.
