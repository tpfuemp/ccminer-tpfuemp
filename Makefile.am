# allow to use Host cuda functions in C/C++
DEF_INCLUDES = @CUDA_INCLUDES@

if WANT_JANSSON
JANSSON_INCLUDES= -I$(top_srcdir)/compat/jansson
else
JANSSON_INCLUDES=
endif

EXTRA_DIST		= autogen.sh README.md README.txt LICENSE.txt \
			  ccminer.sln ccminer.vcxproj ccminer.vcxproj.filters \
			  compat/gettimeofday.c compat/getopt/getopt_long.c

SUBDIRS = compat

bin_PROGRAMS = ccminer

ccminer_SOURCES	= elist.h miner.h compat.h \
			  compat/inttypes.h compat/stdbool.h compat/unistd.h \
			  compat/sys/time.h compat/getopt/getopt.h \
			  crc32.c algos/heavy/hefty1.c \
			  ccminer.cpp pools.cpp util.cpp bench.cpp bignum.cpp \
			  api.cpp hashlog.cpp nvml.cpp stats.cpp sysinfos.cpp cuda.cpp \
			  nvsettings.cpp \
			  algos/equihash/equi-stratum.cpp algos/equihash/equi.cpp algos/equihash/blake2/blake2bx.cpp \
			  algos/equihash/equihash.cpp algos/equihash/cuda_equi.cu algos/equihash/cuda_equi_tromp.cu \
			  algos/allium/allium.cu \
			  algos/balloon/balloon.cpp algos/balloon/balloon_scan.cpp \
			  algos/balloon/sha256-ref.c algos/balloon/cuda_balloon.cu \
			  algos/balloon/balloon.h algos/balloon/sha256.h \
			  algos/rinhash/rinhash_scanhash.cpp algos/rinhash/rinhash.cu \
			  algos/curvehash/curvehash_oracle.cpp algos/curvehash/curvehash.cu algos/curvehash/secp256k1_unity.c \
			  algos/verthash/verthash.cu algos/verthash/cuda_verthash.cu \
			  algos/verthash/verthash-cpu.c algos/verthash/verthash-data.cpp \
			  algos/kawpow/cuda_kawpow.cu \
			  algos/kawpow/kawpow.cpp algos/kawpow/kawpow_core.cpp \
			  algos/kawpow/kawpow_dag.cpp algos/kawpow/kawpow_jit.cpp \
			  algos/kawpow/ethash/ethash.cpp algos/kawpow/ethash/progpow.cpp algos/kawpow/ethash/managed.cpp \
			  algos/kawpow/ethash/primes.c algos/kawpow/ethash/keccak.c \
			  algos/kawpow/ethash/keccakf1600.c algos/kawpow/ethash/keccakf800.c \
			  algos/progpow_multi/progpow_pp.cpp algos/progpow_multi/ppmulti_jit.cpp \
			  algos/progpow_multi/ppmulti_dag.cpp algos/progpow_multi/ppmulti_epoch.cpp \
			  algos/progpow_multi/ppmulti_core.cpp \
			  algos/progpow_multi/ppmulti_bridge.cpp \
			  algos/hoohash/cuda_hoohashv110.cu algos/hoohash/hoohash.cu \
			  algos/hoohash/hoohash_device.cuh \
			  algos/evohash/evohash.cu \
			  algos/ghostrider/ghostrider.cu \
			  algos/argon2d/argon2d.cu algos/argon2d/argon2d_fill.cu \
			  algos/argon2d/blake2b_kernels.cu \
			  algos/argon2d/argon2ref/argon2.c algos/argon2d/argon2ref/core.c \
			  algos/argon2d/argon2ref/encoding.c algos/argon2d/argon2ref/opt.c \
			  algos/argon2d/argon2ref/thread.c algos/argon2d/argon2ref/blake2/blake2b.c \
			  algos/argon2d/argon2d_kernel.h algos/argon2d/cudaexception.h \
			  algos/argon2d/argon2ref/argon2.h algos/argon2d/argon2ref/core.h \
			  algos/argon2d/argon2ref/encoding.h algos/argon2d/argon2ref/thread.h \
			  algos/argon2d/argon2ref/blake2/blake2.h algos/argon2d/argon2ref/blake2/blake2-impl.h \
			  algos/argon2d/argon2ref/blake2/blamka-round-opt.h algos/argon2d/argon2ref/blake2/blamka-round-ref.h \
			  algos/yescrypt/yescrypt.cu algos/yescrypt/cuda_yescrypt.cu \
			  sph/yescrypt-common.c sph/yescrypt-opt.c sph/sha256_Y.c \
			  sph/yescrypt.h sph/yescrypt-platform.h \
			  algos/odocrypt/odocrypt_host.cpp algos/odocrypt/cuda_odocrypt.cu \
			  algos/heavyhash/heavyhash.cu \
			  algos/heavyhash/cuda_heavyhash.cu \
			  algos/heavyhash/keccak_tiny.c \
			  algos/heavyhash/heavyhash-gate.c \
			  algos/heavy/heavy.cu \
			  algos/heavy/cuda_heavy_blake512.cu \
			  algos/heavy/cuda_combine.cu \
			  algos/heavy/cuda_heavy_groestl512.cu \
			  algos/heavy/cuda_hefty1.cu \
			  algos/heavy/cuda_heavy_keccak512.cu \
			  algos/heavy/cuda_sha256.cu \
			  algos/heavy/bastion.cu algos/heavy/cuda_bastion.cu \
			  algos/fugue/fuguecoin.cpp algos/stages/cuda_fugue256.cu sph/fugue.c uint256.h \
			  algos/groestlcoin/groestlcoin.cpp algos/groestlcoin/cuda_groestlcoin.cu algos/groestlcoin/cuda_groestlcoin.h \
			  algos/myriadgroestl/myriadgroestl.cpp algos/myriadgroestl/cuda_myriadgroestl.cu \
			  algos/lyra2/Lyra2.c algos/lyra2/Sponge.c \
			  algos/lyra2/lyra2RE.cu algos/stages/cuda_lyra2.cu \
			  algos/lyra2/lyra2REv2.cu algos/stages/cuda_lyra2v2.cu \
			  algos/lyra2/Lyra2Z.c algos/lyra2/lyra2Z.cu algos/stages/cuda_lyra2Z.cu \
			  algos/stages/cuda_bmw256.cu algos/stages/cuda_cubehash256.cu \
			  algos/stages/cuda_blake256.cu algos/stages/cuda_groestl256.cu \
			  algos/stages/cuda_skein256.cu \
			  algos/blake256/blake256.cu algos/blake256/decred.cu algos/blake256/vanilla.cu \
			  algos/blake2s/blake2s.cu sph/blake2s.c \
			  algos/bmw/bmw.cu algos/bmw/bmw512.cu algos/stages/cuda_bmw.cu \
			  algos/gost/cuda_gosthash.cu algos/gost/gost.cu \
			  algos/cryptonight/xmr-rpc.cpp algos/cryptonight/wildkeccak-cpu.cpp algos/cryptonight/wildkeccak.cu \
			  algos/cryptonight/cryptolight.cu algos/cryptonight/cryptolight-core.cu algos/cryptonight/cryptolight-cpu.cpp \
			  algos/cryptonight/cryptonight.cu algos/cryptonight/cryptonight-core.cu algos/cryptonight/cryptonight-extra.cu \
			  algos/cryptonight/cryptonight-cpu.cpp algos/cryptonight/oaes_lib.cpp algos/cryptonight/aesb.cpp algos/cryptonight/cpu/c_keccak.c \
			  algos/jha/jha.cu algos/jha/jackpotcoin.cu algos/stages/cuda_jha_keccak512.cu \
			  algos/jha/cuda_jha_compactionTest.cu algos/common/cuda_checkhash.cu \
			  algos/stages/cuda_jh512.cu algos/stages/cuda_blake512.cu algos/stages/cuda_groestl512.cu algos/stages/cuda_skein512.cu \
			  algos/stages/cuda_bmw512.cu algos/stages/cuda_keccak512.cu \
			  algos/quark/animecoin.cu algos/quark/nist5.cu \
			  algos/quark/quarkcoin.cu algos/stages/cuda_quark_compactionTest.cu \
			  algos/neoscrypt/neoscrypt.cpp algos/neoscrypt/neoscrypt-cpu.c algos/neoscrypt/cuda_neoscrypt.cu \
			  algos/pentablake/pentablake.cu algos/skein/skein.cu algos/skein/cuda_skeincoin.cu algos/skein/skein2.cpp algos/zr5/zr5.cu \
			  algos/skunk/skunk.cu algos/skunk/cuda_skunk.cu algos/stages/cuda_streebog.cu \
			  algos/sha256d/sha256d.cu algos/sha256d/cuda_sha256d.cu algos/sha256d/sha256t.cu algos/sha256d/cuda_sha256t.cu \
			  algos/sha256d/cuda_sha256_selftest.cu cuda/sha256_device.cuh \
			  algos/sha256d/sha256csm.cu algos/sha256d/cuda_sha256csm.cu \
			  algos/sha512256d/sha512256d.cu algos/sha512256d/cuda_sha512256d.cu \
			  algos/sha512256d/cuda_sha512256d_selftest.cu cuda/sha512_device.cuh \
			  algos/sha256dv/sha256dv.cu \
			  algos/blake2b/blake2b.cu \
			  algos/keccak/keccak256.cu algos/keccak/cuda_keccak256.cu \
			  algos/keccak/cuda_keccak_selftest.cu cuda/keccak_device.cuh \
			  cuda/xfamily_selftest.cu cuda/blake512_device.cuh \
			  cuda/jh512_device.cuh cuda/bmw512_device.cuh cuda/skein512_device.cuh \
			  cuda/luffa512_device.cuh cuda/shabal512_device.cuh cuda/cubehash512_device.cuh \
			  cuda/hamsi512_device.cuh cuda/fugue512_device.cuh cuda/groestl512_device.cuh cuda/echo512_device.cuh cuda/shavite512_device.cuh cuda/whirlpool512_device.cuh cuda/tiger192_device.cuh \
			  algos/sha3d/sha3d.cu algos/sha3d/cuda_sha3d.cu \
			  algos/sha3t/sha3t.cu algos/sha3t/cuda_sha3t.cu \
			  algos/sia/sia.cu algos/sia/sia-rpc.cpp sph/blake2b.c \
			  sph/bmw.c sph/blake.c sph/groestl.c sph/jh.c sph/keccak.c sph/skein.c \
			  sph/cubehash.c sph/echo.c sph/luffa.c sph/sha2.c sph/shavite.c sph/simd.c \
			  sph/hamsi.c sph/hamsi_helper.c sph/streebog.c sph/tiger.c \
			  sph/shabal.c sph/whirlpool.c sph/sha2big.c sph/haval.c \
			  sph/ripemd.c sph/sph_sha2.c sph/sha3d.c sph/sha3.c \
			  algos/polytimos/polytimos.cu \
			  algos/lbry/lbry.cu algos/lbry/cuda_lbry_merged.cu \
			  algos/qubit/qubit.cu algos/qubit/deep.cu algos/qubit/luffa.cu \
			  algos/tribus/tribus.cu algos/stages/cuda_echo512_final.cu \
			  algos/x11/x11.cu algos/x11/fresh.cu algos/stages/cuda_luffa512.cu algos/stages/cuda_luffa512_80.cu algos/stages/cuda_cubehash512.cu algos/stages/cuda_cubehash512_80.cu \
			  algos/stages/cuda_shavite512_legacy.cu algos/stages/cuda_shavite512.cu algos/stages/cuda_simd512.cu algos/stages/cuda_echo512_compat.cu \
			  algos/stages/cuda_luffaCubehash512.cu algos/x11/x11evo.cu algos/x11/timetravel.cu algos/x11/bitcore.cu \
			  algos/x13/x13.cu algos/stages/cuda_hamsi512.cu algos/stages/cuda_fugue512.cu \
			  algos/hsr/hsr.cu algos/hsr/cuda_hsr_sm3.cu algos/hsr/sm3.c \
			  algos/x15/x14.cu algos/x15/x15.cu algos/stages/cuda_shabal512.cu algos/stages/cuda_whirlpool512.cu \
			  algos/stages/cuda_whirlpool512_80.cu \
			  algos/x15/whirlpool.cu \
			  algos/whirlpoolx2/whirlpoolx2.cu algos/whirlpoolx2/cuda_whirlpoolx2.cu \
			  algos/x16/x16r.cu algos/x16/x16rv2.cu algos/x16/x16s.cu algos/stages/cuda_echo512_80.cu algos/stages/cuda_fugue512_80.cu \
			  algos/stages/cuda_shabal512_80.cu algos/stages/cuda_shavite512_80.cu algos/stages/cuda_simd512_80.cu \
			  algos/stages/cuda_echo512_64.cu algos/common/cuda_x_fused.cu \
			  algos/x17/x17.cu algos/x17/skydoge.cu algos/x17/hmq1725.cu algos/soterg/soterg.cu algos/stages/cuda_haval256.cu algos/stages/cuda_sha512.cu \
			  algos/x11/phi.cu \
			  algos/x11/c11.cu algos/x11/s3.cu algos/x11/sib.cu algos/x11/veltor.cu algos/x11/0x10.cu \
			  algos/x21/x21s.cu algos/stages/cuda_sha256_2.cu algos/stages/cuda_tiger192.cu \
			  algos/x25x/x25x.cu algos/x25x/cuda_x25x_panama.cu algos/x25x/cuda_x25x_lane.cu \
			  algos/x25x/cuda_x25x_blake2s.cu algos/x25x/cuda_x25x_shuffle.cu algos/x25x/cuda_x25x_swifftx.cu \
			  algos/x25x/swifftx.c sph/sph_panama.c algos/x25x/lane.c

# scrypt
ccminer_SOURCES += algos/scrypt/scrypt.cpp algos/scrypt/scrypt-jane.cpp \
    algos/scrypt/blake.cu algos/scrypt/keccak.cu algos/scrypt/sha256.cu \
    algos/scrypt/salsa_kernel.cu algos/scrypt/test_kernel.cu \
    algos/scrypt/fermi_kernel.cu algos/scrypt/kepler_kernel.cu \
    algos/scrypt/nv_kernel.cu algos/scrypt/nv_kernel2.cu algos/scrypt/titan_kernel.cu

if HAVE_NVML
nvml_defs = -DUSE_WRAPNVML
nvml_libs = -ldl
endif

if HAVE_WINDOWS
ccminer_SOURCES += compat/winansi.c
endif

ccminer_LDFLAGS  = $(PTHREAD_FLAGS) @CUDA_LDFLAGS@
ccminer_LDADD    = @LIBCURL@ @JANSSON_LIBS@ @PTHREAD_LIBS@ @WS2_LIBS@ @CUDA_LIBS@ @OPENMP_CFLAGS@ @LIBS@ $(nvml_libs)
ccminer_CPPFLAGS = @LIBCURL_CPPFLAGS@ @OPENMP_CFLAGS@ $(CPPFLAGS) $(PTHREAD_FLAGS) -fno-strict-aliasing $(JANSSON_INCLUDES) $(DEF_INCLUDES) $(nvml_defs) -I$(top_srcdir)/algos/curvehash/secp256k1

if HAVE_OSX
ccminer_CPPFLAGS += -I/usr/local/llvm/lib/clang/4.0.0/include
ccminer_LDFLAGS += -L/usr/local/llvm/lib
ccminer_LDADD += -lomp
endif

#ccminer_CPPFLAGS += -DUSE_LIBSODIUM
#ccminer_LDFLAGS += -Lalgos/equihash/lib
#ccminer_LDADD += -lsodium
ccminer_LDADD += -lcuda
# NVRTC: per-period ProgPoW kernel JIT (kawpow_jit / ppmulti_jit). Windows links
# nvrtc.lib via the vcxproj; on Linux it lives in $CUDA/lib64 (@CUDA_LDFLAGS@).
ccminer_LDADD += -lnvrtc

nvcc_ARCH :=
nvcc_ARCH += -gencode=arch=compute_86,code=\"sm_86,compute_86\"
nvcc_ARCH += -gencode=arch=compute_75,code=\"sm_75,compute_75\"
nvcc_ARCH += -gencode=arch=compute_61,code=\"sm_61,compute_61\"
#nvcc_ARCH += -gencode=arch=compute_52,code=\"sm_52,compute_52\"
#nvcc_ARCH += -gencode=arch=compute_50,code=\"sm_50,compute_50\"
#nvcc_ARCH += -gencode=arch=compute_35,code=\"sm_35,compute_35\"
#nvcc_ARCH += -gencode=arch=compute_30,code=\"sm_30,compute_30\"

nvcc_FLAGS = $(nvcc_ARCH) @CUDA_INCLUDES@ -I. -I$(srcdir) @CUDA_CFLAGS@
nvcc_FLAGS += $(JANSSON_INCLUDES) --ptxas-options="-v"

# we're now targeting all major compute architectures within one binary.
.cu.o:
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=128 -o $@ -c $<

# Tromp equihash solver: parameterised for 144/5 (WN=144 selects the
# BUCKBITS=20/RESTBITS=4 code paths); no register cap, ptxas cache hints.
algos/equihash/cuda_equi_tromp.o: algos/equihash/cuda_equi_tromp.cu
	$(NVCC) $(nvcc_FLAGS) -DWN=144 -DWK=5 -DRESTBITS=4 -Xptxas -dlcm=ca -Xptxas -dscm=cs -o $@ -c $<

algos/stages/cuda_lyra2.o: algos/stages/cuda_lyra2.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=128 -o $@ -c $<

# HoohashV110 device TU: consensus-critical strict FP (must match the glibc oracle).
# No fast-math; disable FMA contraction; precise div/sqrt; no flush-to-zero.
algos/hoohash/cuda_hoohashv110.o: algos/hoohash/cuda_hoohashv110.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=128 --fmad=false --prec-div=true --prec-sqrt=true --ftz=false -o $@ -c $<

algos/blake256/blake256.o: algos/blake256/blake256.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=64 -o $@ -c $<

algos/stages/cuda_bmw.o: algos/stages/cuda_bmw.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=76 -o $@ -c $<

algos/cryptonight/cryptonight-core.o: algos/cryptonight/cryptonight-core.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=64 -o $@ -c $<

algos/cryptonight/cryptonight-extra.o: algos/cryptonight/cryptonight-extra.cu
	$(NVCC) $(nvcc_FLAGS) -o $@ -c $<

algos/heavy/cuda_hefty1.o: algos/heavy/cuda_hefty1.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=80 -o $@ -c $<

algos/stages/cuda_luffa512_80.o: algos/stages/cuda_luffa512_80.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=80 -o $@ -c $<

# Luffa and Echo are faster with 80 registers than 128
algos/stages/cuda_luffa512.o: algos/stages/cuda_luffa512.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=80 -o $@ -c $<

algos/stages/cuda_luffaCubehash512.o: algos/stages/cuda_luffaCubehash512.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=76 -o $@ -c $<

algos/stages/cuda_simd512.o: algos/stages/cuda_simd512.cu
	$(NVCC) $(nvcc_FLAGS) -Xcompiler -Wno-unused-variable -o $@ -c $<

algos/stages/cuda_hamsi512.o: algos/stages/cuda_hamsi512.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=72 -o $@ -c $<

algos/stages/cuda_sha512.o: algos/stages/cuda_sha512.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=80 -o $@ -c $<

algos/stages/cuda_blake512.o: algos/stages/cuda_blake512.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=80 -o $@ -c $<

algos/stages/cuda_jh512.o: algos/stages/cuda_jh512.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=80 -o $@ -c $<

algos/stages/cuda_keccak512.o: algos/stages/cuda_keccak512.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=88 -o $@ -c $<

algos/stages/cuda_quark_compactionTest.o: algos/stages/cuda_quark_compactionTest.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=80 -o $@ -c $<

algos/jha/cuda_jha_compactionTest.o: algos/jha/cuda_jha_compactionTest.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=80 -o $@ -c $<


algos/scrypt/titan_kernel.o: algos/scrypt/titan_kernel.cu
	$(NVCC) $(nvcc_FLAGS) -gencode=arch=compute_35,code=\"sm_35,compute_35\" -o $@ -c $<

algos/skein/skein.o: algos/skein/skein.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=64 -o $@ -c $<

