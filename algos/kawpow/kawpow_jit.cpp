// SPDX-License-Identifier: GPL-3.0-or-later
//
// KawPoW ProgPoW per-period JIT (see kawpow_jit.h).
//
// The generated kernel uses the single-thread-per-nonce execution model (all 16
// lanes held in a local mix[16][32] array), matching algos/kawpow/cuda_kawpow.cu
// kawpow_gpu_hash. Only the period-specific ProgPoW program (progPowLoop) is
// generated; the fixed device prefix (keccak_f800, kiss99, fnv1a, init/reduce
// and the kawpow_search entry) is embedded verbatim. keccak_f800 here is kept
// byte-identical to cuda_kawpow.cu's copy — the JIT string cannot #include repo
// headers (plan/guideline note).
//
// Program generation walks the same KISS99 + Fisher-Yates sequence as
// progpow::mix_rng_state (algos/kawpow/ethash/progpow.cpp), consuming the RNG in
// the identical order to round(), so the baked register indices and selectors
// match the reference bit-for-bit.

#include "kawpow_jit.h"

#include <cstdio>
#include <sstream>
#include <vector>
#include <nvrtc.h>

// ---- Host mirror of the ProgPoW RNG (must match progpow.cpp exactly) --------
namespace {

constexpr uint32_t PP_REGS = 32;
constexpr uint32_t PP_LANES = 16;
constexpr int PP_CNT_CACHE = 11;
constexpr int PP_CNT_MATH = 18;
constexpr int PP_DAG_LOADS = 4;

inline uint32_t fnv1a(uint32_t& h, uint32_t d) { return h = (h ^ d) * 0x01000193u; }

struct kiss99_t { uint32_t z, w, jsr, jcong; };

inline uint32_t kiss99(kiss99_t& st)
{
    st.z = 36969 * (st.z & 0xffff) + (st.z >> 16);
    st.w = 18000 * (st.w & 0xffff) + (st.w >> 16);
    st.jcong = 69069 * st.jcong + 1234567;
    st.jsr ^= (st.jsr << 17);
    st.jsr ^= (st.jsr >> 13);
    st.jsr ^= (st.jsr << 5);
    return (((st.z << 16) + st.w) ^ st.jcong) + st.jsr;
}

// mix_rng_state, matching progpow::mix_rng_state.
struct mix_rng_state
{
    kiss99_t rng;
    uint32_t dst_seq[PP_REGS];
    uint32_t src_seq[PP_REGS];
    uint32_t dst_cnt = 0;
    uint32_t src_cnt = 0;

    explicit mix_rng_state(uint64_t period)
    {
        const uint32_t seed0 = (uint32_t)period;
        const uint32_t seed1 = (uint32_t)(period >> 32);
        uint32_t h = 0x811c9dc5u;
        const uint32_t z = fnv1a(h, seed0);
        const uint32_t w = fnv1a(h, seed1);
        const uint32_t jsr = fnv1a(h, seed0);
        const uint32_t jcong = fnv1a(h, seed1);
        rng = kiss99_t{z, w, jsr, jcong};

        for (uint32_t i = 0; i < PP_REGS; ++i) { dst_seq[i] = i; src_seq[i] = i; }
        for (uint32_t i = PP_REGS; i > 1; --i)
        {
            uint32_t j = kiss99(rng) % i;
            uint32_t t = dst_seq[i - 1]; dst_seq[i - 1] = dst_seq[j]; dst_seq[j] = t;
            j = kiss99(rng) % i;
            t = src_seq[i - 1]; src_seq[i - 1] = src_seq[j]; src_seq[j] = t;
        }
    }

    uint32_t next_dst() { return dst_seq[(dst_cnt++) % PP_REGS]; }
    uint32_t next_src() { return src_seq[(src_cnt++) % PP_REGS]; }
    uint32_t next_rng() { return kiss99(rng); }
};

// Baked merge: a = f(a, b). Mirrors progpow::random_merge / kawpowminer merge().
std::string merge(const std::string& a, const std::string& b, uint32_t r)
{
    const uint32_t x = ((r >> 16) % 31) + 1;
    switch (r % 4)
    {
    case 0: return a + " = (" + a + " * 33u) + " + b + ";\n";
    case 1: return a + " = (" + a + " ^ " + b + ") * 33u;\n";
    case 2: return a + " = ROTL32(" + a + ", " + std::to_string(x) + ") ^ " + b + ";\n";
    default: return a + " = ROTR32(" + a + ", " + std::to_string(x) + ") ^ " + b + ";\n";
    }
}

// Baked math: d = g(a, b). Mirrors progpow::random_math / kawpowminer math().
std::string math(const std::string& d, const std::string& a, const std::string& b, uint32_t r)
{
    switch (r % 11)
    {
    case 0: return d + " = " + a + " + " + b + ";\n";
    case 1: return d + " = " + a + " * " + b + ";\n";
    case 2: return d + " = mul_hi(" + a + ", " + b + ");\n";
    case 3: return d + " = min(" + a + ", " + b + ");\n";
    case 4: return d + " = ROTL32(" + a + ", (" + b + ") % 32u);\n";
    case 5: return d + " = ROTR32(" + a + ", (" + b + ") % 32u);\n";
    case 6: return d + " = " + a + " & " + b + ";\n";
    case 7: return d + " = " + a + " | " + b + ";\n";
    case 8: return d + " = " + a + " ^ " + b + ";\n";
    case 9: return d + " = clz(" + a + ") + clz(" + b + ");\n";
    default: return d + " = popcount(" + a + ") + popcount(" + b + ");\n";
    }
}

inline std::string lane(uint32_t reg) { return "mix[" + std::to_string(reg) + "]"; }

// Generates the period-specialized progPowLoop (one round; called 64x at runtime
// with the same program, matching round()'s by-value mix_rng_state).
//
// Warp-cooperative form (kawpowminer ProgPow::getKern, KERNEL_CUDA): 16 threads
// share a nonce, each holds one lane's mix[32]. Cache/math ops are per-thread on
// its own registers; the full-DAG global load is a coalesced 16x16-byte read
// with the item index broadcast from lane (loop%16) via __shfl_sync. The RNG
// walk is identical to progpow::round() (validated), only the emitted code form
// differs from the single-thread version.
std::string gen_progpowloop(uint64_t period)
{
    mix_rng_state st(period);
    std::stringstream s;

    s << "__device__ __forceinline__ void progPowLoop(const uint32_t loop,\n"
      << "        uint32_t* mix,\n"
      << "        const dag_t* __restrict__ g_dag,\n"
      << "        const uint32_t* __restrict__ c_dag,\n"
      << "        const uint32_t hack_false)\n{\n";
    s << "    dag_t data_dag;\n";
    s << "    uint32_t offset, data;\n";
    s << "    const uint32_t lane_id = threadIdx.x & (PROGPOW_LANES - 1);\n";
    // Global load issued at the top, consumed at the bottom, so the cache/math ops
    // hide the DRAM latency. The hack_false __threadfence_block()s (never executed;
    // hack_false is a runtime-0 kernel arg) stop the compiler sinking the load next
    // to its use, preserving the latency-hiding window (kawpowminer technique).
    s << "    offset = __shfl_sync(0xFFFFFFFFu, mix[0], loop % PROGPOW_LANES, PROGPOW_LANES);\n";
    s << "    offset %= PROGPOW_DAG_ELEMENTS;\n";  // compile-time constant -> fast modulo
    s << "    offset = offset * PROGPOW_LANES + (lane_id ^ loop) % PROGPOW_LANES;\n";
    s << "    data_dag = g_dag[offset];\n";
    s << "    if (hack_false) __threadfence_block();\n";

    // Interleaved cache/math ops, exactly as progpow::round().
    const int max_ops = PP_CNT_CACHE > PP_CNT_MATH ? PP_CNT_CACHE : PP_CNT_MATH;
    for (int i = 0; i < max_ops; ++i)
    {
        if (i < PP_CNT_CACHE)
        {
            const uint32_t src = st.next_src();
            const uint32_t dst = st.next_dst();
            const uint32_t sel = st.next_rng();
            s << "    // cache " << i << "\n";
            s << "    data = c_dag[" << lane(src) << " % PROGPOW_CACHE_WORDS];\n";
            s << "    " << merge(lane(dst), "data", sel);
        }
        if (i < PP_CNT_MATH)
        {
            const uint32_t src_rnd = st.next_rng() % (PP_REGS * (PP_REGS - 1));
            const uint32_t src1 = src_rnd % PP_REGS;
            uint32_t src2 = src_rnd / PP_REGS;
            if (src2 >= src1) ++src2;
            const uint32_t sel1 = st.next_rng();
            const uint32_t dst = st.next_dst();
            const uint32_t sel2 = st.next_rng();
            s << "    // math " << i << "\n";
            s << "    " << math("data", lane(src1), lane(src2), sel1);
            s << "    " << merge(lane(dst), "data", sel2);
        }
    }

    // Consume the global load at the end (dsts[0] = reg 0).
    s << "    if (hack_false) __threadfence_block();\n";
    for (int i = 0; i < PP_DAG_LOADS; ++i)
    {
        const uint32_t dst = (i == 0) ? 0 : st.next_dst();
        const uint32_t sel = st.next_rng();
        const std::string src = "data_dag.s[" + std::to_string(i) + "]";
        s << "    " << merge(lane(dst), src, sel);
    }
    s << "}\n";
    return s.str();
}

// Fixed device prefix: typedefs/macros + keccak_f800 (byte-identical to
// cuda_kawpow.cu) + kiss99 + fnv1a. progPowLoop is inserted after this.
const char* kDevicePrefix = R"CUDA(
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
// NOTE: do NOT typedef size_t here -- NVRTC's builtin header already declares it
// (as unsigned long). A conflicting typedef is only a warning under dynamically
// loaded NVRTC but a hard "invalid redeclaration" error under the statically
// linked NVRTC (which force-includes __nv_nvrtc_builtin_header.h). size_t is not
// used in the emitted device code anyway.

#define ROTL32(x,n) __funnelshift_l((x), (x), (n))
#define ROTR32(x,n) __funnelshift_r((x), (x), (n))
#define min(a,b) ((a) < (b) ? (a) : (b))
#define mul_hi(a,b) __umulhi((a), (b))
#define clz(a) __clz((a))
#define popcount(a) __popc((a))

#define PROGPOW_LANES        16
#define PROGPOW_REGS         32
#define PROGPOW_DAG_LOADS    4
#define PROGPOW_CACHE_WORDS  4096
#define FNV_OFFSET_BASIS     0x811c9dc5u

typedef struct __align__(16) { uint32_t s[PROGPOW_DAG_LOADS]; } dag_t;

__device__ __forceinline__ uint32_t rol32(uint32_t x, unsigned s)
{
    return (x << s) | (x >> (32 - s));
}

__device__ __forceinline__ uint32_t cuda_swab32(uint32_t x)
{
    return __byte_perm(x, x, 0x0123);
}

__device__ __forceinline__ uint32_t fnv1a_d(uint32_t h, uint32_t d) { return (h ^ d) * 0x01000193u; }
#define FNV1A(h, d) ((h) = ((h) ^ (d)) * 0x01000193u)

typedef struct { uint32_t z, w, jsr, jcong; } kiss99_t;
__device__ uint32_t kiss99(kiss99_t* st)
{
    st->z = 36969 * (st->z & 0xffff) + (st->z >> 16);
    st->w = 18000 * (st->w & 0xffff) + (st->w >> 16);
    st->jcong = 69069 * st->jcong + 1234567;
    st->jsr ^= (st->jsr << 17);
    st->jsr ^= (st->jsr >> 13);
    st->jsr ^= (st->jsr << 5);
    return (((st->z << 16) + st->w) ^ st->jcong) + st->jsr;
}

// Expand the per-nonce keccak seed into this lane's 32-register mix state.
// Identical to progpow::init_mix's per-lane KISS99 seeding.
__device__ __forceinline__ void fill_mix(uint32_t seed0, uint32_t seed1, uint32_t lane_id, uint32_t* mix)
{
    const uint32_t z = fnv1a_d(FNV_OFFSET_BASIS, seed0);
    const uint32_t w = fnv1a_d(z, seed1);
    kiss99_t st;
    st.z = z; st.w = w;
    st.jsr = fnv1a_d(w, lane_id);
    st.jcong = fnv1a_d(st.jsr, lane_id);
    #pragma unroll
    for (int i = 0; i < PROGPOW_REGS; i++)
        mix[i] = kiss99(&st);
}

__device__ void keccak_f800(uint32_t state[25])
{
    const uint32_t RC[22] = {
        0x00000001, 0x00008082, 0x0000808A, 0x80008000, 0x0000808B, 0x80000001,
        0x80008081, 0x00008009, 0x0000008A, 0x00000088, 0x80008009, 0x8000000A,
        0x8000808B, 0x0000008B, 0x00008089, 0x00008003, 0x00008002, 0x00000080,
        0x0000800A, 0x8000000A, 0x80008081, 0x00008080 };
    uint32_t Aba, Abe, Abi, Abo, Abu, Aga, Age, Agi, Ago, Agu;
    uint32_t Aka, Ake, Aki, Ako, Aku, Ama, Ame, Ami, Amo, Amu;
    uint32_t Asa, Ase, Asi, Aso, Asu;
    uint32_t Eba, Ebe, Ebi, Ebo, Ebu, Ega, Ege, Egi, Ego, Egu;
    uint32_t Eka, Eke, Eki, Eko, Eku, Ema, Eme, Emi, Emo, Emu;
    uint32_t Esa, Ese, Esi, Eso, Esu;
    uint32_t Ba, Be, Bi, Bo, Bu, Da, De, Di, Do, Du;
    Aba = state[0];  Abe = state[1];  Abi = state[2];  Abo = state[3];  Abu = state[4];
    Aga = state[5];  Age = state[6];  Agi = state[7];  Ago = state[8];  Agu = state[9];
    Aka = state[10]; Ake = state[11]; Aki = state[12]; Ako = state[13]; Aku = state[14];
    Ama = state[15]; Ame = state[16]; Ami = state[17]; Amo = state[18]; Amu = state[19];
    Asa = state[20]; Ase = state[21]; Asi = state[22]; Aso = state[23]; Asu = state[24];
    for (int round = 0; round < 22; round += 2) {
        Ba = Aba ^ Aga ^ Aka ^ Ama ^ Asa; Be = Abe ^ Age ^ Ake ^ Ame ^ Ase;
        Bi = Abi ^ Agi ^ Aki ^ Ami ^ Asi; Bo = Abo ^ Ago ^ Ako ^ Amo ^ Aso;
        Bu = Abu ^ Agu ^ Aku ^ Amu ^ Asu;
        Da = Bu ^ rol32(Be, 1); De = Ba ^ rol32(Bi, 1); Di = Be ^ rol32(Bo, 1);
        Do = Bi ^ rol32(Bu, 1); Du = Bo ^ rol32(Ba, 1);
        Ba = Aba ^ Da; Be = rol32(Age ^ De, 12); Bi = rol32(Aki ^ Di, 11);
        Bo = rol32(Amo ^ Do, 21); Bu = rol32(Asu ^ Du, 14);
        Eba = Ba ^ (~Be & Bi) ^ RC[round]; Ebe = Be ^ (~Bi & Bo);
        Ebi = Bi ^ (~Bo & Bu); Ebo = Bo ^ (~Bu & Ba); Ebu = Bu ^ (~Ba & Be);
        Ba = rol32(Abo ^ Do, 28); Be = rol32(Agu ^ Du, 20); Bi = rol32(Aka ^ Da, 3);
        Bo = rol32(Ame ^ De, 13); Bu = rol32(Asi ^ Di, 29);
        Ega = Ba ^ (~Be & Bi); Ege = Be ^ (~Bi & Bo); Egi = Bi ^ (~Bo & Bu);
        Ego = Bo ^ (~Bu & Ba); Egu = Bu ^ (~Ba & Be);
        Ba = rol32(Abe ^ De, 1); Be = rol32(Agi ^ Di, 6); Bi = rol32(Ako ^ Do, 25);
        Bo = rol32(Amu ^ Du, 8); Bu = rol32(Asa ^ Da, 18);
        Eka = Ba ^ (~Be & Bi); Eke = Be ^ (~Bi & Bo); Eki = Bi ^ (~Bo & Bu);
        Eko = Bo ^ (~Bu & Ba); Eku = Bu ^ (~Ba & Be);
        Ba = rol32(Abu ^ Du, 27); Be = rol32(Aga ^ Da, 4); Bi = rol32(Ake ^ De, 10);
        Bo = rol32(Ami ^ Di, 15); Bu = rol32(Aso ^ Do, 24);
        Ema = Ba ^ (~Be & Bi); Eme = Be ^ (~Bi & Bo); Emi = Bi ^ (~Bo & Bu);
        Emo = Bo ^ (~Bu & Ba); Emu = Bu ^ (~Ba & Be);
        Ba = rol32(Abi ^ Di, 30); Be = rol32(Ago ^ Do, 23); Bi = rol32(Aku ^ Du, 7);
        Bo = rol32(Ama ^ Da, 9); Bu = rol32(Ase ^ De, 2);
        Esa = Ba ^ (~Be & Bi); Ese = Be ^ (~Bi & Bo); Esi = Bi ^ (~Bo & Bu);
        Eso = Bo ^ (~Bu & Ba); Esu = Bu ^ (~Ba & Be);
        Ba = Eba ^ Ega ^ Eka ^ Ema ^ Esa; Be = Ebe ^ Ege ^ Eke ^ Eme ^ Ese;
        Bi = Ebi ^ Egi ^ Eki ^ Emi ^ Esi; Bo = Ebo ^ Ego ^ Eko ^ Emo ^ Eso;
        Bu = Ebu ^ Egu ^ Eku ^ Emu ^ Esu;
        Da = Bu ^ rol32(Be, 1); De = Ba ^ rol32(Bi, 1); Di = Be ^ rol32(Bo, 1);
        Do = Bi ^ rol32(Bu, 1); Du = Bo ^ rol32(Ba, 1);
        Ba = Eba ^ Da; Be = rol32(Ege ^ De, 12); Bi = rol32(Eki ^ Di, 11);
        Bo = rol32(Emo ^ Do, 21); Bu = rol32(Esu ^ Du, 14);
        Aba = Ba ^ (~Be & Bi) ^ RC[round + 1]; Abe = Be ^ (~Bi & Bo);
        Abi = Bi ^ (~Bo & Bu); Abo = Bo ^ (~Bu & Ba); Abu = Bu ^ (~Ba & Be);
        Ba = rol32(Ebo ^ Do, 28); Be = rol32(Egu ^ Du, 20); Bi = rol32(Eka ^ Da, 3);
        Bo = rol32(Eme ^ De, 13); Bu = rol32(Esi ^ Di, 29);
        Aga = Ba ^ (~Be & Bi); Age = Be ^ (~Bi & Bo); Agi = Bi ^ (~Bo & Bu);
        Ago = Bo ^ (~Bu & Ba); Agu = Bu ^ (~Ba & Be);
        Ba = rol32(Ebe ^ De, 1); Be = rol32(Egi ^ Di, 6); Bi = rol32(Eko ^ Do, 25);
        Bo = rol32(Emu ^ Du, 8); Bu = rol32(Esa ^ Da, 18);
        Aka = Ba ^ (~Be & Bi); Ake = Be ^ (~Bi & Bo); Aki = Bi ^ (~Bo & Bu);
        Ako = Bo ^ (~Bu & Ba); Aku = Bu ^ (~Ba & Be);
        Ba = rol32(Ebu ^ Du, 27); Be = rol32(Ega ^ Da, 4); Bi = rol32(Eke ^ De, 10);
        Bo = rol32(Emi ^ Di, 15); Bu = rol32(Eso ^ Do, 24);
        Ama = Ba ^ (~Be & Bi); Ame = Be ^ (~Bi & Bo); Ami = Bi ^ (~Bo & Bu);
        Amo = Bo ^ (~Bu & Ba); Amu = Bu ^ (~Ba & Be);
        Ba = rol32(Ebi ^ Di, 30); Be = rol32(Ego ^ Do, 23); Bi = rol32(Eku ^ Du, 7);
        Bo = rol32(Ema ^ Da, 9); Bu = rol32(Ese ^ De, 2);
        Asa = Ba ^ (~Be & Bi); Ase = Be ^ (~Bi & Bo); Asi = Bi ^ (~Bo & Bu);
        Aso = Bo ^ (~Bu & Ba); Asu = Bu ^ (~Ba & Be);
    }
    state[0] = Aba;  state[1] = Abe;  state[2] = Abi;  state[3] = Abo;  state[4] = Abu;
    state[5] = Aga;  state[6] = Age;  state[7] = Agi;  state[8] = Ago;  state[9] = Agu;
    state[10] = Aka; state[11] = Ake; state[12] = Aki; state[13] = Ako; state[14] = Aku;
    state[15] = Ama; state[16] = Ame; state[17] = Ami; state[18] = Amo; state[19] = Amu;
    state[20] = Asa; state[21] = Ase; state[22] = Asi; state[23] = Aso; state[24] = Asu;
}
)CUDA";

// Fixed device suffix: the warp-cooperative kawpow_search entry. 16 threads share
// a nonce; a group of 16 lanes processes 16 nonces (its own threads' nonces), one
// at a time via the h-loop with the seed broadcast by __shfl_sync. Structure
// follows kawpowminer progpow_search. target is the top 64 bits of the 256-bit
// share target (big-endian); the exact 256-bit check is done on the host.
const char* kDeviceEntry = R"CUDA(
struct kawpow_result { uint32_t found; uint32_t nonce_lo; uint32_t mix[8]; uint32_t final[8]; };

extern "C" __global__ void kawpow_search(
    const uint32_t* __restrict__ header, uint64_t start_nonce,
    const dag_t* __restrict__ g_dag,
    uint64_t target, kawpow_result* __restrict__ result, uint32_t hack_false)
{
    __shared__ uint32_t c_dag[PROGPOW_CACHE_WORDS];

    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t nonce = start_nonce + gid;
    const uint32_t lane_id = threadIdx.x & (PROGPOW_LANES - 1);

    const uint32_t rvn[15] = {
        0x00000072, 0x00000041, 0x00000056, 0x00000045, 0x0000004E,
        0x00000043, 0x0000004F, 0x00000049, 0x0000004E, 0x0000004B,
        0x00000041, 0x00000057, 0x00000050, 0x0000004F, 0x00000057 };

    // Cooperatively stage the first 16 KB of the DAG into shared memory (== L1).
    for (uint32_t word = threadIdx.x * PROGPOW_DAG_LOADS; word < PROGPOW_CACHE_WORDS;
         word += blockDim.x * PROGPOW_DAG_LOADS) {
        dag_t load = g_dag[word / PROGPOW_DAG_LOADS];
        #pragma unroll
        for (int i = 0; i < PROGPOW_DAG_LOADS; ++i) c_dag[word + i] = load.s[i];
    }
    __syncthreads();

    // Initial keccak seed for this thread's own nonce.
    uint32_t state2[8];
    {
        uint32_t state[25];
        #pragma unroll
        for (int i = 0; i < 8; ++i) state[i] = header[i];
        state[8] = (uint32_t)nonce;
        state[9] = (uint32_t)(nonce >> 32);
        #pragma unroll
        for (int i = 10; i < 25; ++i) state[i] = rvn[i - 10];
        keccak_f800(state);
        #pragma unroll
        for (int i = 0; i < 8; ++i) state2[i] = state[i];
    }

    uint32_t digest[8];  // this thread's nonce mix digest
    #pragma unroll 1
    for (uint32_t h = 0; h < PROGPOW_LANES; ++h) {
        uint32_t mix[PROGPOW_REGS];
        const uint32_t s0 = __shfl_sync(0xFFFFFFFFu, state2[0], h, PROGPOW_LANES);
        const uint32_t s1 = __shfl_sync(0xFFFFFFFFu, state2[1], h, PROGPOW_LANES);
        fill_mix(s0, s1, lane_id, mix);

        #pragma unroll 1
        for (uint32_t loop = 0; loop < 64; ++loop)
            progPowLoop(loop, mix, g_dag, c_dag, hack_false);

        // Reduce this lane's mix to a 32-bit value, then cross-lane to 256 bits.
        uint32_t digest_lane = FNV_OFFSET_BASIS;
        #pragma unroll
        for (int i = 0; i < PROGPOW_REGS; ++i) FNV1A(digest_lane, mix[i]);

        uint32_t dt[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) dt[i] = FNV_OFFSET_BASIS;
        for (int i = 0; i < PROGPOW_LANES; i += 8)
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                FNV1A(dt[j], __shfl_sync(0xFFFFFFFFu, digest_lane, i + j, PROGPOW_LANES));

        if (h == lane_id)
            #pragma unroll
            for (int i = 0; i < 8; ++i) digest[i] = dt[i];
    }

    // Final keccak for this thread's nonce.
    uint32_t state[25];
    #pragma unroll
    for (int i = 0; i < 8; ++i) state[i] = state2[i];
    #pragma unroll
    for (int i = 8; i < 16; ++i) state[i] = digest[i - 8];
    #pragma unroll
    for (int i = 16; i < 25; ++i) state[i] = rvn[i - 16];
    keccak_f800(state);

    // Top-64-bit big-endian screen (host re-verifies the full 256 bits).
    const uint64_t res = ((uint64_t)cuda_swab32(state[0]) << 32) | cuda_swab32(state[1]);
    if (res > target) return;

    if (atomicExch(&result->found, 1u) == 0u) {
        result->nonce_lo = gid;
        #pragma unroll
        for (int i = 0; i < 8; ++i) { result->mix[i] = digest[i]; result->final[i] = state[i]; }
    }
}
)CUDA";

} // anonymous namespace

std::string kawpow_progpow_source(uint64_t period, uint32_t num_items)
{
    std::string src;
    src.reserve(1 << 15);
    // DAG element count baked as a compile-time constant (constant modulo).
    src += "#define PROGPOW_DAG_ELEMENTS ";
    src += std::to_string(num_items);
    src += "u\n";
    src += kDevicePrefix;
    src += "\n// ProgPoW program for period ";
    src += std::to_string(period);
    src += "\n";
    src += gen_progpowloop(period);
    src += kDeviceEntry;
    return src;
}

// ---- JIT compile + cache ----------------------------------------------------

kawpow_jit::~kawpow_jit()
{
    if (cached_module_)
        cuModuleUnload(cached_module_);
}

bool kawpow_jit::get(uint64_t period, uint32_t num_items, CUfunction* fn)
{
    // period uniquely determines the epoch (hence num_items), so key on period.
    if (cached_module_ && cached_period_ == period)
    {
        *fn = cached_fn_;
        return true;
    }

    const std::string src = kawpow_progpow_source(period, num_items);
    ++compiles_;

    nvrtcProgram prog;
    nvrtcResult nr = nvrtcCreateProgram(&prog, src.c_str(), "kawpow.cu", 0, nullptr, nullptr);
    if (nr != NVRTC_SUCCESS)
    {
        fprintf(stderr, "kawpow_jit: nvrtcCreateProgram: %s\n", nvrtcGetErrorString(nr));
        return false;
    }

    char arch_opt[64];
    snprintf(arch_opt, sizeof(arch_opt), "--gpu-architecture=compute_%d", sm_arch_);
    const char* opts[] = { arch_opt, "--std=c++14" };
    nr = nvrtcCompileProgram(prog, 2, opts);
    if (nr != NVRTC_SUCCESS)
    {
        size_t log_size = 0;
        nvrtcGetProgramLogSize(prog, &log_size);
        std::vector<char> log(log_size ? log_size : 1);
        nvrtcGetProgramLog(prog, log.data());
        fprintf(stderr, "kawpow_jit: nvrtcCompileProgram failed:\n%s\n", log.data());
        nvrtcDestroyProgram(&prog);
        return false;
    }

    size_t ptx_size = 0;
    nvrtcGetPTXSize(prog, &ptx_size);
    std::vector<char> ptx(ptx_size);
    nvrtcGetPTX(prog, ptx.data());
    nvrtcDestroyProgram(&prog);

    CUmodule mod = nullptr;
    CUresult cr = cuModuleLoadData(&mod, ptx.data());
    if (cr != CUDA_SUCCESS)
    {
        const char* es = nullptr; cuGetErrorString(cr, &es);
        fprintf(stderr, "kawpow_jit: cuModuleLoadData: %s\n", es ? es : "?");
        return false;
    }
    CUfunction f = nullptr;
    cr = cuModuleGetFunction(&f, mod, "kawpow_search");
    if (cr != CUDA_SUCCESS)
    {
        const char* es = nullptr; cuGetErrorString(cr, &es);
        fprintf(stderr, "kawpow_jit: cuModuleGetFunction: %s\n", es ? es : "?");
        cuModuleUnload(mod);
        return false;
    }

    if (cached_module_)
        cuModuleUnload(cached_module_);
    cached_module_ = mod;
    cached_fn_ = f;
    cached_period_ = period;
    *fn = f;
    return true;
}
