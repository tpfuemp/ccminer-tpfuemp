#!/usr/bin/env bash
# Build the HiveOS custom-miner archive expected by /hive/miners/custom/custom-get.
# The version comes from configure.ac and is written into the packaged manifest.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
srcdir="$(cd "$here/.." && pwd)"
version="$(sed -n 's/^AC_INIT(\[[^]]*\], *\[\([^]]*\)\].*/\1/p' "$srcdir/configure.ac" | head -n 1)"
[[ -n $version ]] || { echo "cannot read the version from configure.ac" >&2; exit 1; }
binary="$srcdir/build-linux/ccminer"

if [[ ! -x $binary ]]; then
	"$srcdir/build.sh"
fi

stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
minerdir="$stage/ccminer-tpfuemp"
mkdir -p "$minerdir"

install -m 0755 "$binary" "$minerdir/ccminer"
install -m 0755 "$here/h-config.sh" "$minerdir/h-config.sh"
install -m 0755 "$here/h-run.sh" "$minerdir/h-run.sh"
install -m 0755 "$here/h-stats.sh" "$minerdir/h-stats.sh"
install -m 0644 "$here/h-manifest.conf" "$minerdir/h-manifest.conf"
printf 'CUSTOM_VERSION=%s\n' "$version" >> "$minerdir/h-manifest.conf"
install -m 0644 "$here/README.md" "$minerdir/README.md"

archive="$srcdir/ccminer-tpfuemp-$version.tar.gz"
tar -czf "$archive" -C "$stage" ccminer-tpfuemp
echo "$archive"
