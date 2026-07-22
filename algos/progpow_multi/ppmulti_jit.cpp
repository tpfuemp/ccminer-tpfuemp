// SPDX-License-Identifier: GPL-3.0-or-later
//
// Parameterized ProgPoW per-period JIT (see ppmulti_jit.h). Derived from
// algos/kawpow/kawpow_jit.cpp; the register/cache/math counts and the keccak
// seal are emitted from pp_params so one generator serves MeowPow / EvrProgPow /
// FiroPoW. The host-side RNG walk mirrors progpow_pp::mix_rng_state
// (algos/progpow_multi/progpow_pp.cpp) consuming the KISS99 stream in the exact
// order of round(), so the baked register indices/selectors match the reference
// bit-for-bit for whichever variant is being generated.

#include "ppmulti_jit.h"

#include <cstdio>
#include <sstream>
#include <vector>
#include <nvrtc.h>

namespace {

constexpr uint32_t PP_MAX_REGS = 32;
constexpr uint32_t PP_LANES = 16;
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

// Host mirror of progpow_pp::mix_rng_state (runtime register count).
struct mix_rng_state
{
    kiss99_t rng;
    uint32_t num_regs;
    uint32_t dst_seq[PP_MAX_REGS];
    uint32_t src_seq[PP_MAX_REGS];
    uint32_t dst_cnt = 0;
    uint32_t src_cnt = 0;

    mix_rng_state(uint64_t period, uint32_t regs) : num_regs(regs)
    {
        const uint32_t seed0 = (uint32_t)period;
        const uint32_t seed1 = (uint32_t)(period >> 32);
        uint32_t h = 0x811c9dc5u;
        const uint32_t z = fnv1a(h, seed0);
        const uint32_t w = fnv1a(h, seed1);
        const uint32_t jsr = fnv1a(h, seed0);
        const uint32_t jcong = fnv1a(h, seed1);
        rng = kiss99_t{z, w, jsr, jcong};

        for (uint32_t i = 0; i < num_regs; ++i) { dst_seq[i] = i; src_seq[i] = i; }
        for (uint32_t i = num_regs; i > 1; --i)
        {
            uint32_t j = kiss99(rng) % i;
            uint32_t t = dst_seq[i - 1]; dst_seq[i - 1] = dst_seq[j]; dst_seq[j] = t;
            j = kiss99(rng) % i;
            t = src_seq[i - 1]; src_seq[i - 1] = src_seq[j]; src_seq[j] = t;
        }
    }

    uint32_t next_dst() { return dst_seq[(dst_cnt++) % num_regs]; }
    uint32_t next_src() { return src_seq[(src_cnt++) % num_regs]; }
    uint32_t next_rng() { return kiss99(rng); }
};

// Baked merge: a = f(a, b). Mirrors progpow_pp::random_merge.
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

// Baked math: d = g(a, b). Mirrors progpow_pp::random_math.
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

// Period-specialized progPowLoop (one round; run 64x at runtime). RNG walk is
// identical to progpow_pp::round() for the given register/cache/math counts.
std::string gen_progpowloop(const pp_params& p, uint64_t period)
{
    mix_rng_state st(period, (uint32_t)p.num_regs);
    std::stringstream s;

    s << "__device__ __forceinline__ void progPowLoop(const uint32_t loop,\n"
      << "        uint32_t* mix,\n"
      << "        const dag_t* __restrict__ g_dag,\n"
      << "        const uint32_t* __restrict__ c_dag,\n"
      << "        const uint32_t hack_false)\n{\n";
    s << "    dag_t data_dag;\n";
    s << "    uint32_t offset, data;\n";
    s << "    const uint32_t lane_id = threadIdx.x & (PROGPOW_LANES - 1);\n";
    s << "    offset = __shfl_sync(0xFFFFFFFFu, mix[0], loop % PROGPOW_LANES, PROGPOW_LANES);\n";
    s << "    offset %= PROGPOW_DAG_ELEMENTS;\n";
    s << "    offset = offset * PROGPOW_LANES + (lane_id ^ loop) % PROGPOW_LANES;\n";
    s << "    data_dag = g_dag[offset];\n";
    s << "    if (hack_false) __threadfence_block();\n";

    const int max_ops = p.cnt_cache > p.cnt_math ? p.cnt_cache : p.cnt_math;
    for (int i = 0; i < max_ops; ++i)
    {
        if (i < p.cnt_cache)
        {
            const uint32_t src = st.next_src();
            const uint32_t dst = st.next_dst();
            const uint32_t sel = st.next_rng();
            s << "    // cache " << i << "\n";
            s << "    data = c_dag[" << lane(src) << " % PROGPOW_CACHE_WORDS];\n";
            s << "    " << merge(lane(dst), "data", sel);
        }
        if (i < p.cnt_math)
        {
            const uint32_t src_rnd = st.next_rng() % ((uint32_t)p.num_regs * ((uint32_t)p.num_regs - 1));
            const uint32_t src1 = src_rnd % (uint32_t)p.num_regs;
            uint32_t src2 = src_rnd / (uint32_t)p.num_regs;
            if (src2 >= src1) ++src2;
            const uint32_t sel1 = st.next_rng();
            const uint32_t dst = st.next_dst();
            const uint32_t sel2 = st.next_rng();
            s << "    // math " << i << "\n";
            s << "    " << math("data", lane(src1), lane(src2), sel1);
            s << "    " << merge(lane(dst), "data", sel2);
        }
    }

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

// Fixed device prefix: keccak_f800 / kiss99 / fnv1a / fill_mix. The numeric
// PROGPOW_* defines are emitted separately (PROGPOW_REGS varies per variant).
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

// Emit the keccak seal snippet for the device kernel. `indent` prefixes each line.
// `off` is the state fill offset (10 = seed phase, 16 = final phase).
std::string gen_seal(const pp_params& p, const char* indent, int off)
{
    std::stringstream s;
    if (p.seal_mode == PP_SEAL_SEEDWORDS)
        s << indent << "for (int i = " << off << "; i < 25; ++i) state[i] = sw[i - " << off << "];\n";
    else  // PP_SEAL_VANILLA: zero the fill range, then set the two padding words.
    {
        s << indent << "for (int i = " << off << "; i < 25; ++i) state[i] = 0u;\n";
        if (off == 10)
            s << indent << "state[10] = 0x00000001u; state[18] = 0x80008081u;\n";
        else
            s << indent << "state[17] = 0x00000001u; state[24] = 0x80008081u;\n";
    }
    return s.str();
}

// Warp-cooperative search entry (16 threads share a nonce). Structure follows
// kawpow_search; the keccak seal is emitted per variant.
std::string gen_device_entry(const pp_params& p)
{
    std::stringstream s;
    s << "struct ppmulti_result { uint32_t found; uint32_t nonce_lo; uint32_t mix[8]; uint32_t final[8]; };\n\n";
    s << "extern \"C\" __global__ void progpow_search(\n"
      << "    const uint32_t* __restrict__ header, uint64_t start_nonce,\n"
      << "    const dag_t* __restrict__ g_dag,\n"
      << "    uint64_t target, ppmulti_result* __restrict__ result, uint32_t hack_false)\n{\n";
    s << "    __shared__ uint32_t c_dag[PROGPOW_CACHE_WORDS];\n";
    s << "    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;\n";
    s << "    const uint64_t nonce = start_nonce + gid;\n";
    s << "    const uint32_t lane_id = threadIdx.x & (PROGPOW_LANES - 1);\n";

    if (p.seal_mode == PP_SEAL_SEEDWORDS)
    {
        s << "    const uint32_t sw[15] = {";
        for (int i = 0; i < 15; ++i)
        {
            char buf[16];
            snprintf(buf, sizeof(buf), "0x%08Xu", p.seed_words[i]);
            s << (i ? "," : "") << buf;
        }
        s << "};\n";
    }

    s << "    for (uint32_t word = threadIdx.x * PROGPOW_DAG_LOADS; word < PROGPOW_CACHE_WORDS;\n"
      << "         word += blockDim.x * PROGPOW_DAG_LOADS) {\n"
      << "        dag_t load = g_dag[word / PROGPOW_DAG_LOADS];\n"
      << "        #pragma unroll\n"
      << "        for (int i = 0; i < PROGPOW_DAG_LOADS; ++i) c_dag[word + i] = load.s[i];\n"
      << "    }\n"
      << "    __syncthreads();\n\n";

    // Initial keccak seed for this thread's own nonce.
    s << "    uint32_t state2[8];\n";
    s << "    {\n";
    s << "        uint32_t state[25];\n";
    s << "        #pragma unroll\n";
    s << "        for (int i = 0; i < 8; ++i) state[i] = header[i];\n";
    s << "        state[8] = (uint32_t)nonce;\n";
    s << "        state[9] = (uint32_t)(nonce >> 32);\n";
    s << gen_seal(p, "        ", 10);
    s << "        keccak_f800(state);\n";
    s << "        #pragma unroll\n";
    s << "        for (int i = 0; i < 8; ++i) state2[i] = state[i];\n";
    s << "    }\n\n";

    s << "    uint32_t digest[8];\n";
    s << "    #pragma unroll 1\n";
    s << "    for (uint32_t h = 0; h < PROGPOW_LANES; ++h) {\n";
    s << "        uint32_t mix[PROGPOW_REGS];\n";
    s << "        const uint32_t s0 = __shfl_sync(0xFFFFFFFFu, state2[0], h, PROGPOW_LANES);\n";
    s << "        const uint32_t s1 = __shfl_sync(0xFFFFFFFFu, state2[1], h, PROGPOW_LANES);\n";
    s << "        fill_mix(s0, s1, lane_id, mix);\n\n";
    s << "        #pragma unroll 1\n";
    s << "        for (uint32_t loop = 0; loop < PROGPOW_CNT_DAG; ++loop)\n";
    s << "            progPowLoop(loop, mix, g_dag, c_dag, hack_false);\n\n";
    s << "        uint32_t digest_lane = FNV_OFFSET_BASIS;\n";
    s << "        #pragma unroll\n";
    s << "        for (int i = 0; i < PROGPOW_REGS; ++i) FNV1A(digest_lane, mix[i]);\n\n";
    s << "        uint32_t dt[8];\n";
    s << "        #pragma unroll\n";
    s << "        for (int i = 0; i < 8; ++i) dt[i] = FNV_OFFSET_BASIS;\n";
    s << "        for (int i = 0; i < PROGPOW_LANES; i += 8)\n";
    s << "            #pragma unroll\n";
    s << "            for (int j = 0; j < 8; ++j)\n";
    s << "                FNV1A(dt[j], __shfl_sync(0xFFFFFFFFu, digest_lane, i + j, PROGPOW_LANES));\n\n";
    s << "        if (h == lane_id)\n";
    s << "            #pragma unroll\n";
    s << "            for (int i = 0; i < 8; ++i) digest[i] = dt[i];\n";
    s << "    }\n\n";

    // Final keccak for this thread's nonce.
    s << "    uint32_t state[25];\n";
    s << "    #pragma unroll\n";
    s << "    for (int i = 0; i < 8; ++i) state[i] = state2[i];\n";
    s << "    #pragma unroll\n";
    s << "    for (int i = 8; i < 16; ++i) state[i] = digest[i - 8];\n";
    s << gen_seal(p, "    ", 16);
    s << "    keccak_f800(state);\n\n";
    s << "    const uint64_t res = ((uint64_t)cuda_swab32(state[0]) << 32) | cuda_swab32(state[1]);\n";
    s << "    if (res > target) return;\n\n";
    s << "    if (atomicExch(&result->found, 1u) == 0u) {\n";
    s << "        result->nonce_lo = gid;\n";
    s << "        #pragma unroll\n";
    s << "        for (int i = 0; i < 8; ++i) { result->mix[i] = digest[i]; result->final[i] = state[i]; }\n";
    s << "    }\n}\n";
    return s.str();
}

} // anonymous namespace

std::string ppmulti_progpow_source(const pp_params& p, uint64_t period, uint32_t num_items)
{
    std::stringstream s;
    s << "#define PROGPOW_DAG_ELEMENTS " << num_items << "u\n";
    s << "#define PROGPOW_LANES        " << PP_LANES << "\n";
    s << "#define PROGPOW_REGS         " << p.num_regs << "\n";
    s << "#define PROGPOW_CNT_DAG      " << p.cnt_dag << "\n";
    s << "#define PROGPOW_DAG_LOADS    " << PP_DAG_LOADS << "\n";
    s << "#define PROGPOW_CACHE_WORDS  4096\n";
    s << "#define FNV_OFFSET_BASIS     0x811c9dc5u\n";
    s << kDevicePrefix;
    s << "\n// ProgPoW program (" << (p.name ? p.name : "progpow") << ") for period " << period << "\n";
    s << gen_progpowloop(p, period);
    s << gen_device_entry(p);
    return s.str();
}

// ---- JIT compile + cache ----------------------------------------------------

ppmulti_jit::~ppmulti_jit()
{
    if (cached_module_)
        cuModuleUnload(cached_module_);
}

bool ppmulti_jit::get(uint64_t period, uint32_t num_items, CUfunction* fn)
{
    if (cached_module_ && cached_period_ == period)
    {
        *fn = cached_fn_;
        return true;
    }

    const std::string src = ppmulti_progpow_source(params_, period, num_items);
    ++compiles_;

    nvrtcProgram prog;
    nvrtcResult nr = nvrtcCreateProgram(&prog, src.c_str(), "progpow.cu", 0, nullptr, nullptr);
    if (nr != NVRTC_SUCCESS)
    {
        fprintf(stderr, "ppmulti_jit: nvrtcCreateProgram: %s\n", nvrtcGetErrorString(nr));
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
        fprintf(stderr, "ppmulti_jit: nvrtcCompileProgram failed:\n%s\n", log.data());
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
        fprintf(stderr, "ppmulti_jit: cuModuleLoadData: %s\n", es ? es : "?");
        return false;
    }
    CUfunction f = nullptr;
    cr = cuModuleGetFunction(&f, mod, "progpow_search");
    if (cr != CUDA_SUCCESS)
    {
        const char* es = nullptr; cuGetErrorString(cr, &es);
        fprintf(stderr, "ppmulti_jit: cuModuleGetFunction: %s\n", es ? es : "?");
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
