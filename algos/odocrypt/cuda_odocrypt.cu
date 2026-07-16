/*
 * Odocrypt (DigiByte, algo "odo") — CUDA scan driver + device hash.
 *
 *   hash = first 32 bytes of KeccakP800_12( OdoCrypt(key).Encrypt( header||0x01 ) )
 *   key  = nTime - (nTime % ODO_SHAPECHANGE_INTERVAL)
 *
 * The cipher tables depend only on the epoch key (not the nonce), so they are
 * built on the host once per epoch and uploaded to the GPU. Per nonce the
 * kernel runs the 84-round SPN + Keccak-p[800] and prefilters on the top hash
 * word; the host re-verifies each candidate with the reference odo_hash_host.
 */

#include <stdint.h>
#include <string.h>

#include "miner.h"
#include "cuda_helper.h"
#include "odocrypt.h"

// ---- device tables (uploaded per epoch) -----------------------------------
// S-boxes are data-dependent lookups -> keep them in global memory (L2-cached);
// constant memory would serialize divergent reads within a warp. The remaining
// tables are accessed uniformly, so they live in constant memory.
__device__ uint8_t  d_Sbox1[ODO_SMALL_SBOX_COUNT][1 << ODO_SMALL_SBOX_WIDTH];
__device__ uint16_t d_Sbox2[ODO_LARGE_SBOX_COUNT][1 << ODO_LARGE_SBOX_WIDTH];

__constant__ uint64_t c_pmask[2][ODO_PBOX_SUBROUNDS][ODO_STATE_SIZE / 2];
__constant__ int      c_prot[2][ODO_PBOX_SUBROUNDS - 1][ODO_STATE_SIZE / 2];
__constant__ int      c_rot[ODO_ROTATION_COUNT];
__constant__ uint16_t c_rkey[ODO_ROUNDS];
__constant__ uint32_t c_header[19];   // be32enc'd block header words 0..18
__constant__ uint32_t c_target[8];

static uint32_t *d_resNonce[MAX_GPUS] = { 0 };
static uint32_t *h_resNonce[MAX_GPUS] = { 0 };

// ---- device cipher ---------------------------------------------------------

__device__ __forceinline__ uint64_t dev_rot64( uint64_t x, int r )
{
   return r == 0 ? x : ( x << r ) ^ ( x >> ( 64 - r ) );
}

__device__ void dev_apply_pbox( uint64_t state[ODO_STATE_SIZE], int p )
{
   #pragma unroll
   for ( int i = 0; i < ODO_PBOX_SUBROUNDS - 1; i++ )
   {
      #pragma unroll
      for ( int k = 0; k < ODO_STATE_SIZE / 2; k++ )
      {
         uint64_t swp = c_pmask[p][i][k] & ( state[2*k] ^ state[2*k+1] );
         state[2*k]   ^= swp;
         state[2*k+1] ^= swp;
      }
      uint64_t next[ODO_STATE_SIZE];
      #pragma unroll
      for ( int x = 0; x < ODO_STATE_SIZE; x++ )
         next[( ODO_PBOX_M * x ) % ODO_STATE_SIZE] = state[x];
      #pragma unroll
      for ( int x = 0; x < ODO_STATE_SIZE; x++ ) state[x] = next[x];
      #pragma unroll
      for ( int k = 0; k < ODO_STATE_SIZE / 2; k++ )
         state[2*k] = dev_rot64( state[2*k], c_prot[p][i][k] );
   }
   #pragma unroll
   for ( int k = 0; k < ODO_STATE_SIZE / 2; k++ )
   {
      uint64_t swp = c_pmask[p][ODO_PBOX_SUBROUNDS-1][k] & ( state[2*k] ^ state[2*k+1] );
      state[2*k]   ^= swp;
      state[2*k+1] ^= swp;
   }
}

__device__ void dev_apply_sboxes( uint64_t state[ODO_STATE_SIZE] )
{
   const uint64_t MASK1 = ( 1 << ODO_SMALL_SBOX_WIDTH ) - 1;
   const uint64_t MASK2 = ( 1 << ODO_LARGE_SBOX_WIDTH ) - 1;
   int smallIdx = 0;
   #pragma unroll
   for ( int i = 0; i < ODO_STATE_SIZE; i++ )
   {
      uint64_t next = 0;
      int pos = 0, largeIdx = i;
      #pragma unroll
      for ( int j = 0; j < ODO_SMALL_SBOX_COUNT / ODO_STATE_SIZE; j++ )
      {
         next |= (uint64_t)d_Sbox1[smallIdx][( state[i] >> pos ) & MASK1] << pos;
         pos += ODO_SMALL_SBOX_WIDTH;
         next |= (uint64_t)d_Sbox2[largeIdx][( state[i] >> pos ) & MASK2] << pos;
         pos += ODO_LARGE_SBOX_WIDTH;
         smallIdx++;
      }
      state[i] = next;
   }
}

__device__ void dev_apply_rotations( uint64_t state[ODO_STATE_SIZE] )
{
   uint64_t next[ODO_STATE_SIZE];
   #pragma unroll
   for ( int i = 0; i < ODO_STATE_SIZE; i++ )
      next[i] = state[( i + 1 ) % ODO_STATE_SIZE];
   #pragma unroll
   for ( int i = 0; i < ODO_STATE_SIZE; i++ )
      #pragma unroll
      for ( int j = 0; j < ODO_ROTATION_COUNT; j++ )
         next[i] ^= dev_rot64( state[i], c_rot[j] );
   #pragma unroll
   for ( int i = 0; i < ODO_STATE_SIZE; i++ ) state[i] = next[i];
}

// Keccak-p[800], 12 rounds, on a 25 x uint32 state.
__constant__ uint32_t kc_rc[22] =
{
   0x00000001, 0x00008082, 0x0000808a, 0x80008000, 0x0000808b, 0x80000001,
   0x80008081, 0x00008009, 0x0000008a, 0x00000088, 0x80008009, 0x8000000a,
   0x8000808b, 0x0000008b, 0x00008089, 0x00008003, 0x00008002, 0x00000080,
   0x0000800a, 0x8000000a, 0x80008081, 0x00008080
};
__constant__ int kc_rho[25] =
{
   0,  1, 30, 28, 27,  4, 12,  6, 23, 20,  3, 10, 11, 25,  7,
   9, 13, 15, 21,  8, 18,  2, 29, 24, 14
};
#define KIDX(x,y)  ( ( (x) % 5 ) + 5 * ( (y) % 5 ) )

__device__ void dev_keccakp800_12( uint32_t A[25] )
{
   for ( int round = 22 - 12; round < 22; round++ )
   {
      uint32_t C[5], D[5], B[25];
      #pragma unroll
      for ( int x = 0; x < 5; x++ )
         C[x] = A[KIDX(x,0)] ^ A[KIDX(x,1)] ^ A[KIDX(x,2)] ^ A[KIDX(x,3)] ^ A[KIDX(x,4)];
      #pragma unroll
      for ( int x = 0; x < 5; x++ )
         D[x] = ROTL32( C[(x+1)%5], 1 ) ^ C[(x+4)%5];
      #pragma unroll
      for ( int x = 0; x < 5; x++ )
         #pragma unroll
         for ( int y = 0; y < 5; y++ )
            A[KIDX(x,y)] ^= D[x];
      #pragma unroll
      for ( int i = 0; i < 25; i++ )
         A[i] = kc_rho[i] ? ROTL32( A[i], kc_rho[i] ) : A[i];
      #pragma unroll
      for ( int x = 0; x < 5; x++ )
         #pragma unroll
         for ( int y = 0; y < 5; y++ )
            B[KIDX( y, 2*x + 3*y )] = A[KIDX(x,y)];
      #pragma unroll
      for ( int y = 0; y < 5; y++ )
         #pragma unroll
         for ( int x = 0; x < 5; x++ )
            A[KIDX(x,y)] = B[KIDX(x,y)] ^ ( ( ~B[KIDX(x+1,y)] ) & B[KIDX(x+2,y)] );
      A[0] ^= kc_rc[round];
   }
}

__global__ void odocrypt_gpu_hash( uint32_t threads, uint32_t startNonce, uint32_t *resNonce )
{
   const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;
   if ( thread >= threads ) return;
   const uint32_t nonce = startNonce + thread;

   // Build the 10 x uint64 state from the big-endian header words + nonce.
   uint64_t state[ODO_STATE_SIZE];
   #pragma unroll
   for ( int i = 0; i < ODO_STATE_SIZE - 1; i++ )
      state[i] = (uint64_t)c_header[2*i] | ( (uint64_t)c_header[2*i+1] << 32 );
   state[ODO_STATE_SIZE-1] = (uint64_t)c_header[18] | ( (uint64_t)nonce << 32 );

   // premix
   uint64_t total = 0;
   #pragma unroll
   for ( int i = 0; i < ODO_STATE_SIZE; i++ ) total ^= state[i];
   total ^= total >> 32;
   #pragma unroll
   for ( int i = 0; i < ODO_STATE_SIZE; i++ ) state[i] ^= total;

   // 84 SPN rounds
   for ( int round = 0; round < ODO_ROUNDS; round++ )
   {
      dev_apply_pbox( state, 0 );
      dev_apply_sboxes( state );
      dev_apply_pbox( state, 1 );
      dev_apply_rotations( state );
      uint16_t rk = c_rkey[round];
      #pragma unroll
      for ( int i = 0; i < ODO_STATE_SIZE; i++ ) state[i] ^= (uint64_t)( ( rk >> i ) & 1 );
   }

   // pack into the 25-lane Keccak state: words 0..19 from state, 0x01 at byte 80.
   uint32_t A[25];
   #pragma unroll
   for ( int i = 0; i < ODO_STATE_SIZE; i++ )
   {
      A[2*i]   = (uint32_t)state[i];
      A[2*i+1] = (uint32_t)( state[i] >> 32 );
   }
   A[20] = 1u;
   A[21] = A[22] = A[23] = A[24] = 0u;

   dev_keccakp800_12( A );

   // prefilter on the most-significant hash word (A[7]); host re-verifies.
   if ( A[7] <= c_target[7] )
      atomicMin( resNonce, nonce );
}

// ---- host driver -----------------------------------------------------------

extern "C" void odocrypt_init( OdoCrypt *c, uint32_t key );
extern "C" void odo_hash_host( const OdoCrypt *c, void *output, const void *input );

static THREAD OdoCrypt  h_ctx;
static THREAD uint32_t  h_ctx_key = 0;
static THREAD bool      h_ctx_ready = false;

static void odocrypt_upload_tables( const OdoCrypt *c )
{
   CUDA_SAFE_CALL( cudaMemcpyToSymbol( d_Sbox1, c->Sbox1, sizeof c->Sbox1, 0, cudaMemcpyHostToDevice ) );
   CUDA_SAFE_CALL( cudaMemcpyToSymbol( d_Sbox2, c->Sbox2, sizeof c->Sbox2, 0, cudaMemcpyHostToDevice ) );

   uint64_t pmask[2][ODO_PBOX_SUBROUNDS][ODO_STATE_SIZE / 2];
   int      prot[2][ODO_PBOX_SUBROUNDS - 1][ODO_STATE_SIZE / 2];
   for ( int p = 0; p < 2; p++ )
   {
      memcpy( pmask[p], c->Permutation[p].mask, sizeof pmask[p] );
      memcpy( prot[p],  c->Permutation[p].rotation, sizeof prot[p] );
   }
   CUDA_SAFE_CALL( cudaMemcpyToSymbol( c_pmask, pmask, sizeof pmask, 0, cudaMemcpyHostToDevice ) );
   CUDA_SAFE_CALL( cudaMemcpyToSymbol( c_prot,  prot,  sizeof prot,  0, cudaMemcpyHostToDevice ) );
   CUDA_SAFE_CALL( cudaMemcpyToSymbol( c_rot,   c->Rotations, sizeof c->Rotations, 0, cudaMemcpyHostToDevice ) );
   CUDA_SAFE_CALL( cudaMemcpyToSymbol( c_rkey,  c->RoundKey,  sizeof c->RoundKey,  0, cudaMemcpyHostToDevice ) );
}

// Known-answer test (validated against DigiByte Core). input[i]=i*7+1,
// key=0x12345678 rounded to the epoch boundary.
static const uint8_t odo_kat[32] =
{
   0x28,0x66,0xb2,0xe8,0xff,0x9a,0xdb,0x62,0xfe,0x16,0x00,0x79,0x29,0x51,0x62,0xca,
   0x46,0x24,0x3b,0xae,0xe9,0xd6,0xab,0x7e,0xbc,0x87,0xe1,0x96,0x7f,0xd4,0xbc,0x7c
};

static void odo_self_test( int thr_id )
{
   uint8_t in[ODO_DIGEST_SIZE], h[32];
   for ( int i = 0; i < ODO_DIGEST_SIZE; i++ ) in[i] = (uint8_t)( i * 7 + 1 );
   const uint32_t key = 0x12345678u - ( 0x12345678u % ODO_SHAPECHANGE_INTERVAL );
   OdoCrypt c;
   odocrypt_init( &c, key );
   odo_hash_host( &c, h, in );
   if ( memcmp( h, odo_kat, 32 ) == 0 )
      gpulog( LOG_INFO, thr_id, "odocrypt host self-test OK" );
   else
      gpulog( LOG_ERR, thr_id, "odocrypt host self-test MISMATCH" );
}

int scanhash_odocrypt( int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done )
{
   uint32_t *pdata = work->data;
   uint32_t *ptarget = work->target;
   const uint32_t first_nonce = pdata[19];
   const int dev_id = device_map[thr_id];

   uint32_t _ALIGN(64) endiandata[20];
   for ( int i = 0; i < 19; i++ )
      be32enc( &endiandata[i], pdata[i] );

   uint32_t throughput = cuda_default_throughput( thr_id, 1U << 20 );
   throughput = min( throughput, max_nonce - first_nonce );
   throughput &= ~0xffU; // multiple of 256 (block size)
   if ( throughput < 256 ) throughput = 256;

   static THREAD bool init = false;
   if ( !init )
   {
      CUDA_SAFE_CALL( cudaSetDevice( dev_id ) );
      if ( opt_cudaschedule == -1 ) {
         cudaDeviceReset();
         cudaSetDeviceFlags( cudaDeviceScheduleBlockingSync );
      }
      odo_self_test( thr_id );
      CUDA_SAFE_CALL( cudaMalloc( &d_resNonce[thr_id], 2 * sizeof(uint32_t) ) );
      CUDA_SAFE_CALL( cudaMallocHost( &h_resNonce[thr_id], 2 * sizeof(uint32_t) ) );
      gpulog( LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads",
              throughput2intensity( throughput ), throughput );
      init = true;
   }

   // Build the per-epoch cipher tables on key change and upload them.
   const uint32_t ntime = endiandata[17];
   const uint32_t key = ntime - ( ntime % ODO_SHAPECHANGE_INTERVAL );
   if ( !h_ctx_ready || h_ctx_key != key )
   {
      odocrypt_init( &h_ctx, key );
      odocrypt_upload_tables( &h_ctx );
      h_ctx_key = key;
      h_ctx_ready = true;
   }

   CUDA_SAFE_CALL( cudaMemcpyToSymbol( c_header, endiandata, 19 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice ) );
   CUDA_SAFE_CALL( cudaMemcpyToSymbol( c_target, ptarget, 8 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice ) );

   const dim3 block( 256 );
   const dim3 grid( ( throughput + block.x - 1 ) / block.x );
   const uint32_t Htarg = ptarget[7];
   uint32_t n = first_nonce;

   do
   {
      *h_resNonce[thr_id] = UINT32_MAX;
      CUDA_SAFE_CALL( cudaMemcpy( d_resNonce[thr_id], h_resNonce[thr_id], sizeof(uint32_t), cudaMemcpyHostToDevice ) );

      odocrypt_gpu_hash <<< grid, block >>> ( throughput, n, d_resNonce[thr_id] );

      CUDA_SAFE_CALL( cudaMemcpy( h_resNonce[thr_id], d_resNonce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost ) );

      if ( *h_resNonce[thr_id] != UINT32_MAX )
      {
         uint32_t _ALIGN(64) vhash[8];
         const uint32_t cand = *h_resNonce[thr_id];
         endiandata[19] = cand;
         odo_hash_host( &h_ctx, vhash, endiandata );
         if ( vhash[7] <= Htarg && fulltest( vhash, ptarget ) )
         {
            work->nonces[0] = cand;
            work_set_target_ratio( work, vhash );
            *hashes_done = n - first_nonce + throughput;
            work->valid_nonces = 1;
            pdata[19] = cand;
            return 1;
         }
         gpulog( LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", cand );
      }

      if ( (uint64_t)throughput + n >= max_nonce ) { n = max_nonce; break; }
      n += throughput;

   } while ( !work_restart[thr_id].restart );

   *hashes_done = n - first_nonce;
   pdata[19] = n;
   return 0;
}
