#include "diffuse.hpp"
#include "config.hpp"
#include <cstdlib>
#include <immintrin.h>

inline size_t
idx(size_t x, size_t y, size_t z, size_t npx, size_t npy, size_t npz)
{
  return x + npx * (y + npy * z);
}

void
diffuse_d3q7(VARTYPE* T, VARTYPE* Tnew, size_t npx, size_t npy, size_t npz, size_t np, VARTYPE cfl)
{
  for (size_t z = np; z <= npz-(np+1); z++) {
    for (size_t y = np; y <= npy-(np+1); y++) {
      for (size_t x = np; x <= npx-(np+1); x++) {
        Tnew[idx(x,y,z,npx,npy,npz)] = (1. - 6. * cfl) * T[idx(x,y,z,npx,npy,npz)]
                                       + cfl * (T[idx(x-1,y,z,npx,npy,npz)] + T[idx(x+1,y,z,npx,npy,npz)]
                                              + T[idx(x,y-1,z,npx,npy,npz)] + T[idx(x,y+1,z,npx,npy,npz)]
                                              + T[idx(x,y,z-1,npx,npy,npz)] + T[idx(x,y,z+1,npx,npy,npz)]);
      }
    }
  }
  return;
}

void
diffuse_blocks_d3q7(VARTYPE* T, VARTYPE* Tnew, size_t npx, size_t npy, size_t npz, size_t np, size_t nbx, size_t nby, size_t nbz, VARTYPE cfl)
{
  const size_t bsize = npx * npy * npz;
  for (size_t b = 0; b < (nbx * nby * nbz); b++) {
    diffuse_d3q7(&T[bsize * b], &Tnew[bsize * b], npx, npy, npz, np, cfl);
  }
  return;
}

void
diffuse_d3q7_avx512(VARTYPE* T, VARTYPE* Tnew, size_t npx, size_t npy, size_t npz, size_t np, VARTYPE cfl)
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

void
diffuse_blocks_d3q7_avx512(VARTYPE* T, VARTYPE* Tnew, size_t npx, size_t npy, size_t npz, size_t np, size_t nbx, size_t nby, size_t nbz, VARTYPE cfl)
{
  const size_t bsize = npx * npy * npz;
  for (size_t b = 0; b < (nbx * nby * nbz); b++) {
    diffuse_d3q7_avx512(&T[bsize * b], &Tnew[bsize * b], npx, npy, npz, np, cfl);
  }
  return;
}
