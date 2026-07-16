# To change the cuda arch, edit Makefile.am and run ./build.sh
#
# Resolves ./configure relative to this script's location, so it works whether
# run in the source tree or from a separate (out-of-tree) build dir.
# LDFLAGS adds the WSL2 driver lib dir (libcuda.so lives in /usr/lib/wsl/lib,
# not cuda/lib64); harmless on native Linux (just an extra, empty search path).

srcdir="$(cd "$(dirname "$0")" && pwd)"

extracflags="-march=native -D_REENTRANT -falign-functions=16 -falign-jumps=16 -falign-labels=16"

CUDA_CFLAGS="-O3 -lineno -Xcompiler -Wall  -D_FORCE_INLINES" \
	"$srcdir/configure" CXXFLAGS="-O3 $extracflags" LDFLAGS="-L/usr/lib/wsl/lib" \
	--with-cuda=/usr/local/cuda --with-nvml=libnvidia-ml.so
