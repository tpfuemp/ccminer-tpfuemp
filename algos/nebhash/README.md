# Nebhash (NEB, `neb-pool/1`) - `-a nebhash` / `-a neb`

Integer-only proof of work used by NEB. One attempt derives a key from an
80-byte header with BLAKE3, draws 16 samples from a fixed dataset with ChaCha8,
sums the training gradients of a convolutional model on those samples, clips
the sum, hashes it into a digest and then, with the key, into a 256-bit
little-endian ticket that is compared with the pool's share target. No floating
point, no DAG and no epoch regeneration.

`-a neb` is an alias for `-a nebhash`. Both plaintext `stratum+tcp://` and TLS
`stratum+ssl://` / `stratum+tcps://` are supported.

## Dataset

The dataset is consensus-pinned. Download `neb-datasets.bin` from the NEB
release page and pass it with `--neb-dataset=PATH` (default: `neb-datasets.bin`
in the working directory). The official file is 16,087,037 bytes:

```text
SHA256  57e22a4b8aacecb2ebae62f76307ddccb04198ed5c4ba15c1dee90ca7a7c2174
BLAKE3  790b57fbd830f3dcbe7de9d1146aa89a0a20799118e30dd48a343a5c98f19e72
```

The miner computes the BLAKE3 digest of the whole file at startup and compares
it with the `dataset_hash` sent in `mining.subscribe`. A mismatch aborts mining
with `Nebhash: pool dataset hash does not match the local file`. The file is a
user asset and is not part of this repository. It holds two model slots, both
of architecture KEP1 (784 inputs, 26,000 weights; 13,404 and 7,089 samples).

## Layout

| File | Role |
| :-- | :-- |
| `nebhash.cpp` / `nebhash.h` | ccminer bridge: dataset load, header build, scan selection, self-test, host re-verify and submit. |
| `nebhash2.c` / `nebhash2.h` | Portable C reference for nebhash v2 (KEP1/RET2): model init, sample gradient, share evaluation, ticket, model step and state hashes. |
| `nebhash2_table_cuda.cu` / `.h` | CUDA scans: the table scan and the sparse scan (below). |
| `shares.c` / `shares.h` | Portable reference for the v1 training-share PoW and the shared constants (`NEB_NB=16`, `NEB_CLIP=256`, ...). |
| `chacha8.c` | Portable ChaCha8 used by both references. BLAKE3 is the tree's shared `cuda/blake3_device.cuh`. |
| `LICENSE.NEB` | Apache-2.0 licence of the vendored reference code. |

## How the GPU scans

For each model the host evaluates every sample once with the C reference and
uploads one row per sample: the convolution gradients (int8, int16 or int32,
the narrowest width that holds the layer) and, per linear layer, the error and
input factors whose outer product is the gradient.

**Table scan.** One thread block per attempt gathers the rows of its 16
samples, sums, clips and hashes. Samples whose output channel or linear output
is inactive are skipped, since they add exact zeros. On Pascal the launch shape
(128 or 32 threads per attempt) is timed per model.

**Sparse scan.** A sample whose gradient is all zero adds only a constant to
the digest, and on trained models most samples are like that. An attempt with
no active draw therefore has one fixed digest, and an attempt with exactly one
active draw has the digest of that sample alone; both are precomputed with the
table. Every evaluated attempt has the same chance to be a share (the ticket is
a BLAKE3 of digest and key), so the sparse scan derives the key and the 16
draws of every nonce and fully evaluates only the attempts with at most one
active draw. The reported hash rate counts evaluated attempts only, which is
what the pool sees.

After each model change the miner times both scans on its first calls and keeps
the faster one; the sparse scan is used only after it has passed the self-test.
An untrained model (for example the `--benchmark` model) has few all-zero
samples and stays on the table scan.

The scan window adapts to the share target so that a call expects about 1/8
share, and a scan never crosses `max_nonce`, the end of the thread's nonce
slice (GPUs share one header).

## Stratum (`neb-pool/1`)

| Message | Shape |
| :-- | :-- |
| `mining.subscribe` | `["<user agent>", "neb-pool/1"]` -> object with `session`, `extranonce1`, `extranonce2_size`, `chain_id`, `dataset_hash`. |
| `mining.set_target` | `["<32-byte LE hex>"]` - the share threshold. |
| `mining.set_model` | `[slot, "<weights_hash hex>", "<packed weights hex>"]`. |
| `mining.notify` | One job object: `job_id`, `version`, `parent`, `time_start`, `bits`, `target`, `share_target`, `coinbase_prefix/suffix`, `branch`, `tx_count`, `shares_root`, `height`, `slot`, `weights_hash`, `expiry`. |
| `mining.submit` | `[login, job_id, extranonce2_hex, ntime(number), nonce(number)]`. |
| `mining.block` | Optional pool extension, sent only after the node accepted a candidate. Params are either `["<job_id>", "<nonce>"]` or an object with `job_id`, `extranonce2`, `ntime`, `nonce`, `blockhash`, `height`. ccminer then logs `yay!!! ... solved: N`. |

The notify target fields describe chain/block state; the value that gates shares
is `mining.set_target`. Target-to-difficulty uses yiimp's
`2^256 / (target + 1)`. The `sd=` password option is interpreted by the pool
(yiimp scale `sd * 1e10`); the miner never parses it.

## Correctness

- Every GPU hit is recomputed on the host with `neb2_share_eval()` before a
  `mining.submit`; a kernel bug can only cause a local miss, never a bad share.
- Every model a mining thread uploads is self-tested before use. Table scan:
  the C reference tickets of 32 nonces of a fixed header, then the GPU hit set
  at a bound equal to the second-lowest ticket (exactly the two lowest nonces)
  and at that ticket minus one (exactly the lowest), for every launch shape.
  Sparse scan: the tickets of the first 32 attempts it evaluates are compared
  with the C reference, then the same two hit-set checks. A mismatch stops the
  miner; a CUDA resource failure only warns.
- `NEBHASH_SELFTEST_FAULT=1`..`5` breaks one of these checks on purpose and
  must stop the miner (testing only).
- `NEB_NEGATIVE_CONTROL_TABLE_CLIP` builds a kernel that clips one below
  `NEB_CLIP`, for checking that a differential test detects a wrong gradient.

## Usage

```bash
ccminer -a nebhash \
  -o stratum+ssl://pool.example.org:3645 \
  -u <neb-address>.<worker> -p c=NEB \
  --neb-dataset=/path/to/neb-datasets.bin
```

`stratum+tcps://` is accepted as an alias for `stratum+ssl://`. TLS certificate
verification stays enabled; use `--cert` only to supply a custom CA bundle.

`--benchmark -a nebhash --neb-dataset=...` mines a synthetic job: a model of
slot 0 drawn from a fixed seed (an untrained model, so its rate is only
comparable with other benchmark runs, not with a live pool), a header per
thread and a target of one share per 2^22 attempts. Every hit is re-verified
on the CPU like a pool share, and the counts are logged every 30 s.

## Rates

| GPU | Slot 0 | Slot 1 | Notes |
| :-- | :-- | :-- | :-- |
| GTX 1080 Ti (sm_61) | ~27-31 MH/s | ~180-220 MH/s | live pool, sparse scan, mainnet models of 2026-11-27 |

Rates depend on the model state: the share of all-zero samples decides how many
nonces the sparse scan evaluates.

## Provenance

`nebhash2.{c,h}`, `shares.{c,h}`, `chacha8.{c,h}` and the reference-derived
table kernel are vendored from the NEB implementation under Apache-2.0
(`LICENSE.NEB`). The ccminer bridge, the sparse scan and the stratum glue follow
the project's GPLv2.
