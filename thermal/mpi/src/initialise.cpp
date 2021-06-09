#include "initialise.hpp"
#include "config.hpp"
#include <cstdlib>

inline size_t
idx(size_t x, size_t y, size_t z, size_t npx, size_t npy, size_t npz)
{
  return x + npx * (y + npy * z);
}

void
initialise(VARTYPE* T, size_t npx, size_t npy, size_t npz, size_t np, VARTYPE Tbulk, VARTYPE Tbc)
{
  for (size_t z = 0; z <= npz-1; z++) {
    for (size_t y = 0; y <= npy-1; y++) {
      for (size_t x = 0; x <= npx-1; x++) {
        T[idx(x,y,z,npx,npy,npz)] = Tbc;
      }
    }
  }
  for (size_t z = np; z <= npz-(np+1); z++) {
    for (size_t y = np; y <= npy-(np+1); y++) {
      for (size_t x = np; x <= npx-(np+1); x++) {
        T[idx(x,y,z,npx,npy,npz)] = Tbulk;
      }
    }
  }
  return;
}

void
initialise_blocks(VARTYPE* T, size_t npx, size_t npy, size_t npz, size_t np, size_t nbx, size_t nby, size_t nbz, VARTYPE Tbulk, VARTYPE Tbc)
{
  const size_t bsize = npx * npy * npz;
  for (size_t b = 0; b < (nbx * nby * nbz); b++) {
    initialise(&T[bsize * b], npx, npy, npz, np, Tbulk, Tbc);
  }
  return;
}
