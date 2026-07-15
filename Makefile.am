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
			  equi/equi-stratum.cpp equi/equi.cpp equi/blake2/blake2bx.cpp \
			  equi/equihash.cpp equi/cuda_equi.cu \
			  algos/allium/allium.cu \
			  balloon/balloon.cpp balloon/balloon_scan.cpp \
			  balloon/sha256-ref.c balloon/cuda_balloon.cu \
			  balloon/balloon.h balloon/sha256.h \
			  rinhash/rinhash_scanhash.cpp rinhash/rinhash.cu \
			  algos/hoohash/cuda_hoohashv110.cu algos/hoohash/hoohash.cu \
			  algos/hoohash/hoohash_device.cuh \
			  evohash/evohash.cu \
			  odocrypt/odocrypt_host.cpp odocrypt/cuda_odocrypt.cu \
			  heavyhash/heavyhash.cu \
			  heavyhash/cuda_heavyhash.cu \
			  heavyhash/keccak_tiny.c \
			  heavyhash/heavyhash-gate.c \
			  algos/heavy/heavy.cu \
			  algos/heavy/cuda_heavy_blake512.cu \
			  algos/heavy/cuda_combine.cu \
			  algos/heavy/cuda_heavy_groestl512.cu \
			  algos/heavy/cuda_hefty1.cu \
			  algos/heavy/cuda_heavy_keccak512.cu \
			  algos/heavy/cuda_sha256.cu \
			  algos/heavy/bastion.cu algos/heavy/cuda_bastion.cu \
			  fuguecoin.cpp algos/stages/cuda_fugue256.cu sph/fugue.c uint256.h \
			  groestlcoin.cpp cuda_groestlcoin.cu cuda_groestlcoin.h \
			  myriadgroestl.cpp cuda_myriadgroestl.cu \
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
			  gost/cuda_gosthash.cu gost/gost.cu \
			  crypto/xmr-rpc.cpp crypto/wildkeccak-cpu.cpp crypto/wildkeccak.cu \
			  crypto/cryptolight.cu crypto/cryptolight-core.cu crypto/cryptolight-cpu.cpp \
			  crypto/cryptonight.cu crypto/cryptonight-core.cu crypto/cryptonight-extra.cu \
			  crypto/cryptonight-cpu.cpp crypto/oaes_lib.cpp crypto/aesb.cpp crypto/cpu/c_keccak.c \
			  algos/jha/jha.cu algos/jha/jackpotcoin.cu algos/stages/cuda_jha_keccak512.cu \
			  algos/jha/cuda_jha_compactionTest.cu cuda_checkhash.cu \
			  algos/stages/cuda_jh512.cu algos/stages/cuda_blake512.cu algos/stages/cuda_groestl512.cu algos/stages/cuda_skein512.cu \
			  algos/stages/cuda_bmw512.cu algos/stages/cuda_keccak512.cu \
			  algos/quark/animecoin.cu algos/quark/nist5.cu \
			  algos/quark/quarkcoin.cu algos/stages/cuda_quark_compactionTest.cu \
			  neoscrypt/neoscrypt.cpp neoscrypt/neoscrypt-cpu.c neoscrypt/cuda_neoscrypt.cu \
			  algos/pentablake/pentablake.cu skein.cu cuda_skeincoin.cu skein2.cpp algos/zr5/zr5.cu \
			  algos/skunk/skunk.cu algos/skunk/cuda_skunk.cu algos/stages/cuda_streebog.cu \
			  algos/sha256d/sha256d.cu algos/sha256d/cuda_sha256d.cu algos/sha256d/sha256t.cu algos/sha256d/cuda_sha256t.cu \
			  algos/sha256d/cuda_sha256_selftest.cu cuda/sha256_device.cuh \
			  algos/sha512256d/sha512256d.cu algos/sha512256d/cuda_sha512256d.cu \
			  algos/sha512256d/cuda_sha512256d_selftest.cu cuda/sha512_device.cuh \
			  algos/sha256dv/sha256dv.cu \
			  algos/keccak/keccak256.cu algos/keccak/cuda_keccak256.cu \
			  algos/keccak/cuda_keccak_selftest.cu cuda/keccak_device.cuh \
			  cuda/xfamily_selftest.cu cuda/blake512_device.cuh \
			  cuda/jh512_device.cuh cuda/bmw512_device.cuh cuda/skein512_device.cuh \
			  cuda/luffa512_device.cuh cuda/shabal512_device.cuh cuda/cubehash512_device.cuh \
			  cuda/hamsi512_device.cuh cuda/fugue512_device.cuh cuda/groestl512_device.cuh cuda/echo512_device.cuh cuda/shavite512_device.cuh cuda/whirlpool512_device.cuh cuda/tiger192_device.cuh \
			  algos/sha3d/sha3d.cu algos/sha3d/cuda_sha3d.cu \
			  algos/sha3t/sha3t.cu algos/sha3t/cuda_sha3t.cu \
			  sia/sia.cu sia/sia-rpc.cpp sph/blake2b.c \
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
			  algos/x16/x16r.cu algos/x16/x16rv2.cu algos/x16/x16s.cu algos/stages/cuda_echo512_80.cu algos/stages/cuda_fugue512_80.cu \
			  algos/stages/cuda_shabal512_80.cu algos/stages/cuda_shavite512_80.cu algos/stages/cuda_simd512_80.cu \
			  algos/stages/cuda_echo512_64.cu algos/common/cuda_x_fused.cu \
			  algos/x17/x17.cu algos/x17/skydoge.cu algos/x17/hmq17.cu algos/stages/cuda_haval256.cu algos/stages/cuda_sha512.cu \
			  algos/x11/phi.cu \
			  algos/x11/c11.cu algos/x11/s3.cu algos/x11/sib.cu algos/x11/veltor.cu algos/x11/0x10.cu \
			  algos/x21/x21s.cu algos/stages/cuda_sha256_2.cu algos/stages/cuda_tiger192.cu 
			  
# scrypt
ccminer_SOURCES += scrypt.cpp scrypt-jane.cpp \
    scrypt/blake.cu scrypt/keccak.cu scrypt/sha256.cu \
    scrypt/salsa_kernel.cu scrypt/test_kernel.cu \
    scrypt/fermi_kernel.cu scrypt/kepler_kernel.cu \
    scrypt/nv_kernel.cu scrypt/nv_kernel2.cu scrypt/titan_kernel.cu

if HAVE_NVML
nvml_defs = -DUSE_WRAPNVML
nvml_libs = -ldl
endif

if HAVE_WINDOWS
ccminer_SOURCES += compat/winansi.c
endif

ccminer_LDFLAGS  = $(PTHREAD_FLAGS) @CUDA_LDFLAGS@
ccminer_LDADD    = @LIBCURL@ @JANSSON_LIBS@ @PTHREAD_LIBS@ @WS2_LIBS@ @CUDA_LIBS@ @OPENMP_CFLAGS@ @LIBS@ $(nvml_libs)
ccminer_CPPFLAGS = @LIBCURL_CPPFLAGS@ @OPENMP_CFLAGS@ $(CPPFLAGS) $(PTHREAD_FLAGS) -fno-strict-aliasing $(JANSSON_INCLUDES) $(DEF_INCLUDES) $(nvml_defs)

if HAVE_OSX
ccminer_CPPFLAGS += -I/usr/local/llvm/lib/clang/4.0.0/include
ccminer_LDFLAGS += -L/usr/local/llvm/lib
ccminer_LDADD += -lomp
endif

#ccminer_CPPFLAGS += -DUSE_LIBSODIUM
#ccminer_LDFLAGS += -Lequi/lib
#ccminer_LDADD += -lsodium
ccminer_LDADD += -lcuda

nvcc_ARCH :=
nvcc_ARCH += -gencode=arch=compute_86,code=\"sm_86,compute_86\"
nvcc_ARCH += -gencode=arch=compute_75,code=\"sm_75,compute_75\"
nvcc_ARCH += -gencode=arch=compute_61,code=\"sm_61,compute_61\"
#nvcc_ARCH += -gencode=arch=compute_52,code=\"sm_52,compute_52\"
#nvcc_ARCH += -gencode=arch=compute_50,code=\"sm_50,compute_50\"
#nvcc_ARCH += -gencode=arch=compute_35,code=\"sm_35,compute_35\"
#nvcc_ARCH += -gencode=arch=compute_30,code=\"sm_30,compute_30\"

nvcc_FLAGS = $(nvcc_ARCH) @CUDA_INCLUDES@ -I. @CUDA_CFLAGS@
nvcc_FLAGS += $(JANSSON_INCLUDES) --ptxas-options="-v"

# we're now targeting all major compute architectures within one binary.
.cu.o:
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=128 -o $@ -c $<

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

crypto/cryptonight-core.o: crypto/cryptonight-core.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=64 -o $@ -c $<

crypto/cryptonight-extra.o: crypto/cryptonight-extra.cu
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


scrypt/titan_kernel.o: scrypt/titan_kernel.cu
	$(NVCC) $(nvcc_FLAGS) -gencode=arch=compute_35,code=\"sm_35,compute_35\" -o $@ -c $<

skein.o: skein.cu
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=64 -o $@ -c $<

