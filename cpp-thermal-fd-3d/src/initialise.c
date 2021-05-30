#include "initialise.h"
#include "config.h"
#include <stdlib.h>

inline size_t
idx(size_t x, size_t y, size_t z, size_t npx, size_t npy, size_t npz)
{
  return x + npx * (y + npy * z);
}

void
initialise(REAL* T, size_t npx, size_t npy, size_t npz, size_t np, double Tbulk, double Tbc)
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
