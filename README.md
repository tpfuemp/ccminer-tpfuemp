# ccminer

Based on Christian Buchner's &amp; Christian H.'s CUDA project, no more active on github since 2014.

Check the [README.txt](README.txt) for the additions

GRLC donation address: GUZA18kQyjfKDPLdcrHKhvcvdoS1JcUr2V (fancyIX)

This branck is for improving allium performance forked from [lenis0012](https://github.com/lenis0012/ccminer)

Requirements (this fork)
-------------------------

> **This fork targets modern NVIDIA hardware and CUDA 11.8 only.**

- **CUDA Toolkit:** 11.8 (fixed). Older toolkits (10.x, 11.0–11.7) are no longer supported.
- **Build toolchain (Windows):** Visual Studio 2022, project file `ccminer.vcxproj`.
- **Supported GPU architectures:** Pascal (`sm_61`, GTX 10-series) and newer — Turing (`sm_75`) and Ampere (`sm_86`). The default build ships native SASS for `sm_61 / sm_75 / sm_86` plus a `compute_86` PTX fallback for later cards.
- **Dropped architectures:** Maxwell (`sm_50/52`), Kepler and Fermi are no longer supported. All architecture-specific code paths below `sm_61` have been removed.

**On a pre-Pascal GPU (Maxwell/Kepler/Fermi), or do you need an older CUDA Toolkit?**
This fork has intentionally dropped that support. Use the upstream project instead, which retains the wider hardware and toolkit range: **https://github.com/tpruvot/ccminer**

About source code dependencies
------------------------------

This project requires some libraries to be built :

- OpenSSL (prebuilt for win)
- Curl (prebuilt for win)
- pthreads (prebuilt for win)

The tree now contains recent prebuilt openssl and curl .lib for both x86 and x64 platforms (windows).

To rebuild them, you need to clone this repository and its submodules :
    git clone https://github.com/peters/curl-for-windows.git compat/curl-for-windows


Compile on Linux
----------------

Please see [INSTALL](https://github.com/tpruvot/ccminer/blob/linux/INSTALL) file or [project Wiki](https://github.com/tpruvot/ccminer/wiki/Compatibility)
