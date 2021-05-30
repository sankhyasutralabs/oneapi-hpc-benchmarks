#include "diffuse.h"
#include "config.h"
#include <stdlib.h>
#include <immintrin.h>

inline size_t
idx(size_t x, size_t y, size_t z, size_t npx, size_t npy, size_t npz)
{
  return x + npx * (y + npy * z);
}

// Tnew(x,y,z) = (1-6*cfl)*T(x,y,z) + cfl*(T(x-1,y,z) + T(x+1,y,z)
//                                       + T(x,y-1,z) + T(x,y+1,z)
//                                       + T(x,y,z-1) + T(x,y,z+1))
void
diffuse(REAL* T, REAL* Tnew, size_t npx, size_t npy, size_t npz, size_t np, double cfl)
{
  __m512d cfl_zmm = _mm512_set1_pd(cfl);
  __m512d one_minus_6cfl_zmm = _mm512_set1_pd(1. - 6. * cfl);
  __m512d zmm, sum;
  for (size_t z = np; z <= npz-(np+1); z++) {
    for (size_t y = np; y <= npy-(np+1); y++) {
      for (size_t x = np; x <= npx-(np+1); x+=8) {
        sum = _mm512_loadu_pd(&T[idx(x-1,y,z,npx,npy,npz)]);
        zmm = _mm512_loadu_pd(&T[idx(x+1,y,z,npx,npy,npz)]);
        sum = _mm512_add_pd(sum, zmm);
        zmm = _mm512_loadu_pd(&T[idx(x,y-1,z,npx,npy,npz)]);
        sum = _mm512_add_pd(sum, zmm);
        zmm = _mm512_loadu_pd(&T[idx(x,y+1,z,npx,npy,npz)]);
        sum = _mm512_add_pd(sum, zmm);
        zmm = _mm512_loadu_pd(&T[idx(x,y,z-1,npx,npy,npz)]);
        sum = _mm512_add_pd(sum, zmm);
        zmm = _mm512_loadu_pd(&T[idx(x,y,z+1,npx,npy,npz)]);
        sum = _mm512_add_pd(sum, zmm);
        sum = _mm512_mul_pd(sum, cfl_zmm);
        zmm = _mm512_loadu_pd(&T[idx(x,y,z,npx,npy,npz)]);
        sum = _mm512_fmadd_pd(zmm, one_minus_6cfl_zmm, sum);
        _mm512_storeu_pd(&Tnew[idx(x,y,z,npx,npy,npz)], sum);
      }
    }
  }
  return;
}

