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
| `yespowerIC` | 2048 | 32 | `IsotopeC` | 8 |
| `yespowerIOTS` | 2048 | 32 | `Iots is committed to the development of IOT` | 43 |
| `yespowerLITB` | 2048 | 32 | `LITBpower: The number of LITB working or available for proof-of-work mini` | 73 |
| `cpupower` | 2048 | 32 | `CPUpower: The number of CPU working or available for proof-of-work mining` | 73 |

Names are case-insensitive. The startup line logs what was actually selected --
`yespower 1.0: N=... r=... key=... (n bytes)` -- which is the cheapest way to confirm a
variant took effect before letting it mine.

## Not supported

- **`yespowerADVC`, `yespowerEQPAY`** (both listed by zpool): no byte-exact source
  for their `pers` was found, so they are deliberately absent rather than guessed.
  `-a yespower --yespower-param 2048,32 --yespower-key "<pers>"` reaches them once
  the string is known from the coin's own source.
- **`power2b`** is a separate algo (blake2b head/tail), not an alias, and its GPU
  path is not implemented.

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
| 2026-08-19 | `yespowerSUGAR` | `yespowerSUGAR.na.mine.zpool.ca:6241` | 0.000 (stratum 0.01) | **6/6 accepted, 0 rejects** | 268 | -- |
| 2026-08-19 | `yespowerTIDE` | `yespowerTIDE.na.mine.zpool.ca:6239` | **0.095** (stratum 0.5) | **3/3 accepted, 0 rejects** | 1 062 | 19.4 at 54 W |
| 2026-08-18 | `yespower` (keyless) | zpool | 0.000 | 2/2 accepted, 0 rejects | 267 | 4.8 at 55 W |

RTX 3060 / driver 595.95. Two of these carry weight beyond "it runs":

- **SUGAR is the first keyed variant a pool has confirmed**, so the 74-byte literal
  and the whole alias -> parameter -> kernel-constant path are consensus-proven rather
  than merely self-consistent. Its 268 H/s matches keyless `yespower`'s 267 H/s, as
  expected -- `pers` enters only the head/tail HMAC key, not the SMix work.
- **TIDE confirms the r=8 kernel and TIDE's own `(N, r)`**, the two things that had no
  independent source. It also ran at job diff **0.095** -- ~19x harder shares than the
  other two runs -- so its three shares exercise the target comparison over a much
  wider range than a diff-0.005 run does.

Still unconfirmed **by a pool**, individually: `URX`, `LTNCG`, `MGPC`, `ARWN`,
`IC`, `IOTS`, `LITB`, `cpupower`. Each is double-sourced (cpuminer-opt's README and
zpool's published parameters agreeing byte-for-byte), so the risk is low, but a wrong
`pers` in any one of them would show up only as 100 % rejects on that coin.

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
