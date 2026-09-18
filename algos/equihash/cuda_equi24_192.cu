// Equihash 192/7 instantiation of the equi24 solver.
//
// 192/7 shares 144/5's geometry exactly (DIGITBITS = 192/8 = 144/6 = 24), so
// RESTBITS, BUCKBITS, NBUCKETS, the partition scheme and the 32-bit reference
// word are all identical. Only WK (7 rounds vs 5) and PROOFSIZE (128 vs 32)
// differ, and both are compile-time.
//
// Built with -DEQ_WN=192 -DEQ_WK=7; the namespace and every exported symbol are
// keyed on EQ_WN (see EQ24_API), so this object links alongside the 144/5 one.
#include "cuda_equi24.cu"
