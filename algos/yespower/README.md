# yespower 1.0 -- coin variants

Every yespower 1.0 coin runs the same hash. They differ only in `(N, r, pers)`, so
they are **aliases of `-a yespower`**, not separate algos: `algo_to_int()` in
`algos.h` maps the name, and the table in `yespower.cu` supplies the parameters.

`pers` is hashed. A single wrong byte produces a hash that looks perfectly healthy
locally and is rejected by the pool 100 % of the time, so the strings are treated as
consensus constants: cross-checked against two independent sources, length-checked
against a literal written beside each one, and verified byte-exact in the compiled
object. Do not retype them from a rendered page, a chat message, or anything that
passes through a text filter -- one such quote silently dropped the words "The",
"of", "or" and "for", which is invisible without a length check.

## Supported aliases

| `-a` name | N | r | pers | bytes |
|---|---|---|---|---|
| `yespower` | 2048 | 32 | (none, or `--yespower-key`) | 0 |
| `yespowerr16`, `yenten` | 4096 | 16 | (none) | 0 |
| `yespowerSUGAR`, `sugarchain` | 2048 | 32 | `Satoshi Nakamoto 31/Oct/2008 Proof-of-work is essentially one-CPU-one-vote` | 74 |
| `yespowerURX` | 2048 | 32 | `UraniumX` | 8 |
| `yespowerLTNCG` | 2048 | 32 | `LTNCGYES` | 8 |
| `yespowerMGPC` | 2048 | 32 | `Magpies are birds of the Corvidae family.` | 41 |
| `yespowerTIDE` | 2048 | 8 | (none) | 0 |
| `yespowerARWN` | 2048 | 32 | `ARWN` | 4 |
| `yespowerADVC` | 2048 | 32 | `Let the quest begin` | 19 |
| `yespowerIC` | 2048 | 32 | `IsotopeC` | 8 |
| `yespowerIOTS` | 2048 | 32 | `Iots is committed to the development of IOT` | 43 |
| `yespowerLITB` | 2048 | 32 | `LITBpower: The number of LITB working or available for proof-of-work mini` | 73 |
| `cpupower` (see below) | 2048 | 32 | `CPUpower: The number of CPU working or available for proof-of-work mining` | 73 |

Names are case-insensitive. The startup line logs what was actually selected --
`yespower 1.0: N=... r=... key=... (n bytes)` -- which is the cheapest way to confirm a
variant took effect before letting it mine.

## Separate algos, not aliases

- **`yespowereqpay`** (EqPay) has ordinary yespower 1.0 parameters -- N=2048,
  r=32, an 88-byte `pers` -- but is **not** a parameter variant, because its
  proof-of-work covers a **181-byte** header rather than 80. EqPay is
  Qtum-derived, so the hashed serialization appends `hashStateRoot`,
  `hashUTXORoot`, a null `prevoutStake` whose index is `0xffffffff`, and an
  empty block-signature length. No `(N, r, pers)` triple changes the length of
  the hashed header, so `--yespower-param`/`--yespower-key` cannot reach it and
  it is dispatched as its own algo. See the EqPay section below.
- **`power2b`** is a separate algo (blake2b head/tail), not an alias.

## Precedence

An explicit `--yespower-param` / `--yespower-key` clears the alias preset, so the
flags win regardless of where they sit on the command line. A per-pool `"algo"` in
the config re-selects on every pool switch: the pool struct keeps the *name*,
because the algo int alone cannot carry which coin was meant.

## Validation status

### Live pool

Only an accepted share proves a `pers` string, because host and GPU read the same
table and would agree with each other on a wrong one.

| Date | `-a` | Pool | Job diff | Shares | H/s | H/W |
|---|---|---|---|---|---|---|
| 2026-09-15 | `yespowerLTNCG` | `yespowerLTNCG.na.mine.zpool.ca:6245` | 0.000 (stratum 0.05) | **5/5 accepted, 0 rejects** | 613 | 6.4 at 96 W |
| 2026-09-15 | `power2b` | `power2b.na.mine.zpool.ca:6242` | 0.000 (stratum 0.05) | **12/12 accepted, 0 rejects** | 636 | 6.1 at 104 W |
| 2026-09-15 | `yespowerMGPC` | `yespowerMGPC.na.mine.zpool.ca:6247` | 0.000 (stratum 0.05) | **13/13 accepted, 0 rejects** | 635 | 6.7 at 94 W |
| 2026-09-15 | `yespowereqpay` | `yespowerEQPAY.na.mine.zpool.ca:6249` | 0.000 (stratum 0.1) | **3/3 accepted, 0 rejects** | 611 | 7.6 at 79 W |
| 2026-09-15 | `yespowerURX` | `yespowerURX.na.mine.zpool.ca:6236` | 0.000 (stratum 0.05) | **8/8 accepted, 0 rejects** | 612 | 6.4 at 95 W |
| 2026-09-15 | `yespowerADVC` | `yespowerADVC.na.mine.zpool.ca:6248` | 0.000 (stratum 0.05) | **6/6 accepted, 0 rejects** | 616 | 6.2 at 99 W |
| 2026-08-19 | `yespowerTIDE` | `yespowerTIDE.na.mine.zpool.ca:6239` | **0.095** (stratum 0.5) | **3/3 accepted, 0 rejects** | 1 062 | 19.4 at 54 W |
| 2026-08-19 | `yespowerSUGAR` | `yespowerSUGAR.na.mine.zpool.ca:6241` | 0.000 (stratum 0.01) | **6/6 accepted, 0 rejects** | 268 | -- |
| 2026-08-18 | `yespower` (keyless) | zpool | 0.000 | 2/2 accepted, 0 rejects | 267 | 4.8 at 55 W |

RTX 3060 / driver 595.95. Two of these carry weight beyond "it runs":

- **SUGAR is the first keyed variant a pool has confirmed**, so the 74-byte literal
  and the whole alias -> parameter -> kernel-constant path are consensus-proven rather
  than merely self-consistent. Its 268 H/s matches keyless `yespower`'s 267 H/s, as
  expected -- `pers` enters only the head/tail HMAC key, not the SMix work.
- **`power2b` is the regression check for the head selector**, not a new variant: it
  was already pool-proven, and the EqPay work replaced the `bool b2b` that chose the
  head/tail primitive with an `int`, touching all three launchers. 12/12 accepted on
  the BLAKE2b path confirms the widening did not disturb it -- and it is the one algo
  here that would fail silently if it had, since a wrong head still produces
  well-formed digests (it once mined SHA-256 against a BLAKE2b network, 0/9).
- **EqPay proves the whole 181-byte header path**, which no local gate can reach: its
  genesis KAT hashes a constant, so the Stratum root parse, the per-word byte swap and
  the `prevoutStake.n` placement were unproven until a pool accepted a share. Each of
  those failing produces a well-formed digest and 100 % rejects, so three accepts is
  evidence for all of them. Block 2550654, chain height confirming the port serves
  EQPAY itself rather than another coin under a family name.
- **TIDE confirms the r=8 kernel and TIDE's own `(N, r)`**, the two things that had no
  independent source. It also ran at job diff **0.095** -- ~19x harder shares than the
  other two runs -- so its three shares exercise the target comparison over a much
  wider range than a diff-0.005 run does.

Still unconfirmed **by a pool**, individually: `ARWN`, `IC`, `IOTS`, `LITB`.
Each is double-sourced (cpuminer-opt's README
and zpool's published parameters agreeing byte-for-byte), so the risk is low, but a wrong
`pers` in any one of them would show up only as 100 % rejects on that coin.

**`cpupower` can no longer be validated, and that is permanent.** CPUchain has
migrated from its Bitcoin-fork "classic" chain to an EVM chain, and mining rewards
on the classic chain are **permanently disabled** -- so no pool serves the algo this
entry implements, and its `pers` will stay double-sourced-but-unproven. The entry is
kept because the code is unaffected and costs nothing; this is the same position
`-a skunk` is in, where the coin died rather than the kernel.
The successor is announced as **"YespowerEVM"**, which is a *different* variant
rather than a new pool for this one -- do not point `-a cpupower` at it when it
launches, and do not assume this 73-byte `pers` carries over until its own spec is
read.

`ADVC` was the exception to "double-sourced": its `pers` has a single source, the
upstream cpuminer-opt-kudaraidee commit `62791ef` that added
`register_yespoweradvc_algo`, read from the raw patch rather than from rendered
text. A pool has now accepted **6/6** ADVC shares, so the 19-byte literal is
consensus-proven and the single source no longer matters -- which is the point of
the table above: an accepted share is the only thing that discharges a `pers`.

### Kernel shapes

| Shape | Gate |
|---|---|
| r=16, r=32 | KAT against `sph/yespower_ref.c`, plus live pool (`-a yespower` keyless and `yespowerSUGAR` keyed) |
| r=8 (`yespowerTIDE`) | **462 GPU candidates re-hashed by the reference at r=8, 0 mismatches** (2026-08-19, `-a yespowerTIDE --benchmark -D`, 20 s; the loose benchmark target keeps the candidate path busy, so this is a real GPU-vs-CPU differential, not a vacuous one) **and pool-confirmed 3/3 at job diff 0.095** |

Measured on an RTX 3060 (2026-08-19, `--benchmark`): r=32 variants ~240 H/s
(224 MB of V), `yespowerTIDE` at r=8 ~920-985 H/s (56 MB of V) -- the ~4x tracks the
quarter-size scratchpad.

**A green GPU-vs-CPU check does not validate the parameters.** Both sides read the
same `(N, r, pers)` from the table, so they agree with each other and would agree just
as happily on a wrong `pers`. Only an accepted pool share settles that -- which is why
the live-pool table above, not this one, is what closed out `yespowerTIDE`.

Every GPU candidate is re-hashed on the host by the reference before submission, so
a kernel bug costs a local reject, never a bad share. A wrong `pers` is the failure
this cannot catch: host and GPU would agree with each other and disagree with the
coin.

## EqPay (`-a yespowereqpay`)

The hashed header is 181 bytes:

| offset | size | field | offset | size | field |
|---:|---:|---|---:|---:|---|
| 0 | 4 | nVersion | 80 | 32 | `hashStateRoot` (pool) |
| 4 | 32 | hashPrevBlock | 112 | 32 | `hashUTXORoot` (pool) |
| 36 | 32 | hashMerkleRoot | 144 | 32 | `prevoutStake.hash`, zero |
| 68 | 4 | nTime | 176 | 4 | `prevoutStake.n` = `0xffffffff` |
| 72 | 4 | nBits | 180 | 1 | signature length, 0 |
| 76 | 4 | nNonce | | | |

Only the head changes: yespower 1.0 replaces the PBKDF2 salt with `pers`, so the
header never reaches the salt and smix/pwxform are untouched. The device side is
`yp_sha256_181()` selected by the `YP_HEAD_SHA256_181` template value; the 181
bytes are three SHA-256 blocks against two for an 80-byte header.

Three things are easy to get wrong here, and each was:

- **`prevoutStake.n` must be `0xffffffff`.** Zero yields a well-formed digest
  that every local check accepts and the pool rejects.
- **The extended tail is stored as the big-endian DECODE of the wire**, like the
  other header words, so one `be32enc()` loop reproduces the wire for all 46
  words. Storing raw wire bytes byte-swaps the tail on the GPU side only, since
  the kernel reads those words as SHA-256 input.
- **The state roots arrive with each 32-bit word byte-swapped.** Undo that per
  word and keep the word order; reversing all 32 bytes is wrong.

### The notify

Confirmed against zpool's port on 2026-09-15. `mining.notify` carries **10
parameters**, with both roots as a single 128-hex (64-byte) value in **slot 3**,
immediately after prevhash -- the slot lbry uses for its claim:

```
[ job_id, prevhash, <64 bytes: hashStateRoot || hashUTXORoot>,
  coinb1, coinb2, merkle[], version, nbits, ntime, clean ]
```

A job without them is **refused** rather than mined with a fallback: they are
consensus data, there is no safe default, and mining without them fails only at
the pool. (Upstream ships a hard-coded fallback root array; it is deliberately
not copied here.)

### Gate

`yespower_eqpay_genesis_kat()` runs at init and fails closed. It checks the
digest of EqPay's own genesis header against the chain's constant -- the only
check in this file with an external oracle, since everything else compares this
GPU to this repo's CPU reference and both can be wrong together. Two further
legs perturb byte 80 and byte 180, which a head hashing 80 or 180 bytes could
not reach, so they are length probes rather than decoration.

The gate cannot see the live header, though: it hashes a fixed constant, so the
root parse, the byte swap and the `prevoutStake.n` placement are covered only by
an accepted share. `-D` adds a per-job GPU-vs-host differential over the same
181-byte path; watch for **distinct** accumulators between jobs, since equal ones
would mean the instrument is returning a constant rather than hashing.
