# ccminer-tpfuemp

A CUDA miner for NVIDIA GPUs — a fork of
[ccminer-kudaraidee](https://github.com/Kudaraidee/ccminer-kudaraidee) (which is
itself based on [tpruvot/ccminer](https://github.com/tpruvot/ccminer)), retargeted
to **CUDA 11.8** and modern NVIDIA hardware, with a reorganised `algos/` source
tree and a growing set of added/optimised algorithms.

See [README.txt](README.txt) for the upstream feature/change history, and
[CREDITS.txt](CREDITS.txt) for lineage, contributor credits, and the donation
address.

Requirements
------------

> **This fork targets modern NVIDIA hardware and CUDA 11.8 only.**

- **CUDA Toolkit:** 11.8 (fixed). Older toolkits (10.x, 11.0–11.7) are not supported.
- **Build toolchain (Windows):** Visual Studio 2022, project file `ccminer.vcxproj`.
- **Supported GPU architectures:** Pascal (`sm_61`, GTX 10-series) and newer —
  Turing (`sm_75`) and Ampere (`sm_86`). The default build ships native SASS for
  `sm_61 / sm_75 / sm_86` plus a `compute_86` PTX fallback for later cards.
- **Dropped architectures:** Maxwell (`sm_50/52`), Kepler and Fermi are no longer
  supported; all architecture-specific code paths below `sm_61` have been removed.

**On a pre-Pascal GPU (Maxwell/Kepler/Fermi), or need an older CUDA Toolkit?**
This fork has intentionally dropped that support. Use the upstream project, which
retains the wider hardware and toolkit range:
**https://github.com/tpruvot/ccminer**

Supported algorithms
---------------------

Select with `-a <name>`. Common aliases are shown in parentheses.

| `-a` name | Coin / description |
|---|---|
| `0x10` | ChainOX |
| `allium` | Lyra2 + Blake2s (Garlicoin) |
| `anime` | Animecoin |
| `argon2d1000` | Zero Dynamics Cash (DYN) |
| `argon2d16000` | Alterdot (ADOT) |
| `balloon` | Balloon hash |
| `bastion` | Hefty bastion |
| `bitcore` (`timetravel10`) | Timetravel-10 |
| `blake` | Blake-256 (SFR) |
| `blake2b` | Blake2-B 512 (BCX) |
| `blake2s` | Blake2-S 256 (NEVA) |
| `blakecoin` | Fast Blake-256 (8 rounds) |
| `bmw` | BMW-256 |
| `bmw512` | BMW-512 |
| `c11` (`flax`) | X11 variant |
| `cryptolight` (`cryptonight-lite`) | AEON CryptoNight (MEM/2) |
| `cryptonight` | Monero-style CryptoNight |
| `decred` | Decred Blake-256 |
| `deep` | Deepcoin |
| `dmd-gr` (`diamond`) | Diamond-Groestl |
| `equihash` (`equi`, `equihash144`) | Zcash Equihash 200/9 (+ 144/5 Tromp) |
| `evohash` | EvoAI |
| `fresh` | Freshcoin (Shavite-80) |
| `fugue256` | Fuguecoin |
| `ghostrider` (`gr`) | GhostRider (Raptoreum) |
| `gostcoin` | Double GOST R 34.11 |
| `groestl` | Groestlcoin |
| `heavy` | Heavycoin *(build-gated: `WITH_HEAVY_ALGO`)* |
| `heavyhash` | HeavyHash (oBTC) |
| `hmq1725` (`hmq17`) | HMQ1725 (Doubloons / Espers) |
| `hoohash` (`hoohashv110`, `pepepow`) | HoohashV110 (PEPEPOW) |
| `hsr` (`hshare`) | HShare / HSR (X13 + SM3) |
| `jackpot` | JHA v8 |
| `jha` | JHA |
| `keccak` | Keccak-256 (deprecated) |
| `keccakc` | Keccak-256 (CreativeCoin) |
| `lbry` | LBRY Credits (SHA/RIPEMD) |
| `luffa` (`doom`) | Joincoin |
| `lyra2` (`lyra2re`) | Lyra2RE (CryptoCoin) |
| `lyra2v2` (`lyra2rev2`) | Lyra2REv2 (VertCoin) |
| `lyra2z` | Lyra2Z (ZeroCoin) |
| `lyra2z330` | Lyra2Z330 |
| `mjollnir` | Mjollnir (Hefty hash) |
| `myr-gr` | Myriad-Groestl |
| `neoscrypt` | NeoScrypt (FeatherCoin, Phoenix, UFO…) |
| `xaya` (`neoscrypt-xaya`) | NeoScrypt (XAYA variant) |
| `nist5` | NIST5 (TalkCoin) |
| `odo` (`odocrypt`) | Odocrypt (DigiByte) |
| `penta` | Pentablake (5× Blake-512) |
| `phi` (`phi1612`) | PHI1612 (BHCoin) |
| `polytimos` | Polytimos |
| `quark` | Quark |
| `qubit` | Qubit |
| `rinhash` | RinHash (Blake3 + Argon2d + SHA3-256) |
| `s3` | S3 (1Coin) |
| `scrypt` | Scrypt |
| `scrypt-jane` | Scrypt-Jane (ChaCha) |
| `sha256csm` | SHA256csm (Galleoncoin) |
| `sha256d` (`bitcoin`, `sha256`) | SHA256d (Bitcoin) |
| `sha256dv` | SHA256d (Veil) |
| `sha256t` | SHA256 ×3 |
| `sha3d` | BSHA3 (Yilacoin, Kylacoin) |
| `sha3t` (`sha3-256t`) | SHA3-256T (Fjarcode, Bitcoin III) |
| `sha512256d` | Double SHA-512/256 (Radiant) |
| `sia` | SIA (Blake2B) |
| `sib` | Sibcoin (X11 + Streebog) |
| `skein` | Skein-SHA2 (Skeincoin) |
| `skein2` | Double Skein (Woodcoin) |
| `skunk` | Skein-Cube-Fugue-Streebog |
| `skydoge` | SkyDoge |
| `timetravel` | Timetravel (Machinecoin, permuted ×8) |
| `tribus` | Tribus (Denarius) |
| `vanilla` | Blake256-8 (VNL) |
| `veltor` (`thorsriddle`) | Veltor (Thorsriddle + Streebog) |
| `whirlcoin` | Old Whirlcoin (Whirlpool) |
| `whirlpool` (`whirl`) | Whirlpool |
| `whirlpoolx` | WhirlpoolX |
| `wildkeccak` | Boolberry |
| `x11` | X11 (DarkCoin) |
| `x11evo` | Permuted X11 (Revolver) |
| `x13` | X13 (MaruCoin) |
| `x14` | X14 |
| `x15` | X15 |
| `x16r` | X16R (Ravencoin) |
| `x16rt` | X16RT (Veil) |
| `x16rv2` | X16Rv2 |
| `x16s` | X16S (Pigeoncoin) |
| `x17` | X17 |
| `x21s` | X21S |
| `yescrypt` | Yescrypt / Globalboost-Y (BSTY), custom `--yescrypt-param` |
| `yescryptr8` | BitZeny (ZNY) |
| `yescryptr16` | Yenten (YTN) |
| `yescryptr16v2` | PPTP |
| `yescryptr24` | JagariCoinR |
| `yescryptr32` | WAVI |
| `zr5` (`ziftr`) | ZR5 (ZiftrCoin) |

Run `ccminer --help` for the authoritative list and per-algo notes.

Building — Windows
-------------------

Open `ccminer.vcxproj` in Visual Studio 2022 with the CUDA 11.8 toolkit
installed, and build the `Release|x64` configuration. From the command line:

    msbuild ccminer.vcxproj /p:Configuration=Release /p:Platform=x64 /t:Build /m

The executable is produced at `x64\Release\ccminer.exe`.

To change the target GPU architectures, edit the CodeGeneration entries in the
project (default `sm_61 / sm_75 / sm_86`).

Building — Linux
----------------

Requires the CUDA 11.8 toolkit and a C/C++ build toolchain
(`build-essential`, `autoconf`/`automake`, and the `curl`, `openssl` and `zlib`
development packages). Then:

    ./build.sh

`build.sh` regenerates the autotools configure script and builds **out-of-tree**
into `build-linux/`, so the resulting binary is `build-linux/ccminer` and the
source tree stays free of build artefacts.

- To change the target GPU architectures, edit `nvcc_ARCH` in `Makefile.am`
  (default `sm_61 / sm_75 / sm_86`) and re-run `./build.sh`.
- **WSL2:** the CUDA driver library `libcuda.so` lives in `/usr/lib/wsl/lib`
  rather than under the toolkit; `configure.sh` already adds it to `LDFLAGS`.

Source-code dependencies
------------------------

The build links against OpenSSL, libcurl and pthreads. For Windows, recent
prebuilt OpenSSL/curl `.lib` files for x86 and x64 are already included in the
tree (`compat/`). To rebuild curl from source, clone its submodule:

    git clone https://github.com/peters/curl-for-windows.git compat/curl-for-windows

Credits & donation
------------------

Lineage, contributor credits, and the donation address are in
[CREDITS.txt](CREDITS.txt).
