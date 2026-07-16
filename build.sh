#!/bin/bash

# Out-of-tree build into build-linux/. This keeps the (Windows-shared) source
# tree free of build artefacts, and links the binary to build-linux/ccminer so
# it does not collide with the Windows MSBuild IntDir directory ./ccminer/.

# export PATH="$PATH:/usr/local/cuda/bin/"

srcdir="$(cd "$(dirname "$0")" && pwd)"
cd "$srcdir"

# Clear any prior IN-TREE configure state, else an out-of-tree configure refuses
# with "source directory already configured; run make distclean there first".
[ -f Makefile ] && make distclean >/dev/null 2>&1
rm -f config.status Makefile

./autogen.sh || echo done

# Fresh out-of-tree build dir.
rm -rf build-linux
mkdir -p build-linux
cd build-linux

"$srcdir/configure.sh"

make -j 8
