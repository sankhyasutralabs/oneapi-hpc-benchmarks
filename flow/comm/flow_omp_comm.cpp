#include <chrono>
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <tuple>
#include <stdlib.h>

#ifndef VARTYPE
  #define VARTYPE double
#endif

#define NG 4UL
#define NM 8UL

#define BETA2  1.
#define ITHETA 3.
#define W000   (64./216.)
#define W100   (16./216.)
#define W110   ( 4./216.)
#define W111   ( 1./216.)

inline size_t
idx(size_t m, size_t x, size_t y, size_t z, size_t g, size_t npx, size_t npy, size_t npz)
{
  return m + NM * (x + npx * (y + npy * (z + npz * g)));
}

void
fill_moments(VARTYPE* f, VARTYPE* rho1, VARTYPE* ux1, VARTYPE* uy1, VARTYPE* uz1)
{
  VARTYPE rho = 0;
  VARTYPE ux  = 0;
  VARTYPE uy  = 0;
  VARTYPE uz  = 0;
  VARTYPE fval;

  fval = f[0 + NM * 0]; rho += fval; ux += +1*fval; uy += +1*fval; uz += -1*fval;
  fval = f[1 + NM * 0]; rho += fval; ux += +0*fval; uy += +1*fval; uz += -1*fval;
  fval = f[2 + NM * 0]; rho += fval; ux += -1*fval; uy += +1*fval; uz += -1*fval;
  fval = f[3 + NM * 0]; rho += fval; ux += +1*fval; uy += +0*fval; uz += -1*fval;
  fval = f[4 + NM * 0]; rho += fval; ux += +0*fval; uy += +0*fval; uz += -1*fval;
  fval = f[5 + NM * 0]; rho += fval; ux += -1*fval; uy += +0*fval; uz += -1*fval;
  fval = f[6 + NM * 0]; rho += fval; ux += +1*fval; uy += -1*fval; uz += -1*fval;

  fval = f[0 + NM * 1]; rho += fval; ux += -1*fval; uy += -1*fval; uz += +1*fval;
  fval = f[1 + NM * 1]; rho += fval; ux += +0*fval; uy += -1*fval; uz += +1*fval;
  fval = f[2 + NM * 1]; rho += fval; ux += +1*fval; uy += -1*fval; uz += +1*fval;
  fval = f[3 + NM * 1]; rho += fval; ux += -1*fval; uy += +0*fval; uz += +1*fval;
  fval = f[4 + NM * 1]; rho += fval; ux += +0*fval; uy += +0*fval; uz += +1*fval;
  fval = f[5 + NM * 1]; rho += fval; ux += +1*fval; uy += +0*fval; uz += +1*fval;
  fval = f[6 + NM * 1]; rho += fval; ux += -1*fval; uy += +1*fval; uz += +1*fval;

  fval = f[0 + NM * 2]; rho += fval; ux += +0*fval; uy += -1*fval; uz += -1*fval;
  fval = f[1 + NM * 2]; rho += fval; ux += -1*fval; uy += -1*fval; uz += -1*fval;
  fval = f[2 + NM * 2]; rho += fval; ux += +0*fval; uy += +0*fval; uz += +0*fval;
  fval = f[3 + NM * 2]; rho += fval; ux += -1*fval; uy += +0*fval; uz += +0*fval;
  fval = f[4 + NM * 2]; rho += fval; ux += +1*fval; uy += -1*fval; uz += +0*fval;
  fval = f[5 + NM * 2]; rho += fval; ux += +0*fval; uy += -1*fval; uz += +0*fval;
  fval = f[6 + NM * 2]; rho += fval; ux += -1*fval; uy += -1*fval; uz += +0*fval;

  fval = f[0 + NM * 3]; rho += fval; ux += +0*fval; uy += +1*fval; uz += +1*fval;
  fval = f[1 + NM * 3]; rho += fval; ux += +1*fval; uy += +1*fval; uz += +1*fval;
  fval = f[3 + NM * 3]; rho += fval; ux += +1*fval; uy += +0*fval; uz += +0*fval;
  fval = f[4 + NM * 3]; rho += fval; ux += -1*fval; uy += +1*fval; uz += +0*fval;
  fval = f[5 + NM * 3]; rho += fval; ux += +0*fval; uy += +1*fval; uz += +0*fval;
  fval = f[6 + NM * 3]; rho += fval; ux += +1*fval; uy += +1*fval; uz += +0*fval;

  *rho1 = rho;
  *ux1 = ux / rho;
  *uy1 = uy / rho;
  *uz1 = uz / rho;
  
  return;
}

void
fill_feq(VARTYPE rho, VARTYPE ux, VARTYPE uy, VARTYPE uz, VARTYPE* f)
{
  const VARTYPE udotu = (ux*ux + uy*uy + uz*uz) * ITHETA;
  VARTYPE cdotu;

  // g0
  cdotu = (+1*ux + +1*uy + -1*uz) * ITHETA;
  f[0 + NM * 0] = rho * W111 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (+0*ux + +1*uy + -1*uz) * ITHETA;
  f[1 + NM * 0] = rho * W110 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (-1*ux + +1*uy + -1*uz) * ITHETA;
  f[2 + NM * 0] = rho * W111 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (+1*ux + +0*uy + -1*uz) * ITHETA;
  f[3 + NM * 0] = rho * W110 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (+0*ux + +0*uy + -1*uz) * ITHETA;
  f[4 + NM * 0] = rho * W100 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (-1*ux + +0*uy + -1*uz) * ITHETA;
  f[5 + NM * 0] = rho * W110 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (+1*ux + -1*uy + -1*uz) * ITHETA;
  f[6 + NM * 0] = rho * W111 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  // g1
  cdotu = (-1*ux + -1*uy + +1*uz) * ITHETA;
  f[0 + NM * 1] = rho * W111 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (+0*ux + -1*uy + +1*uz) * ITHETA;
  f[1 + NM * 1] = rho * W110 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (+1*ux + -1*uy + +1*uz) * ITHETA;
  f[2 + NM * 1] = rho * W111 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (-1*ux + +0*uy + +1*uz) * ITHETA;
  f[3 + NM * 1] = rho * W110 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (+0*ux + +0*uy + +1*uz) * ITHETA;
  f[4 + NM * 1] = rho * W100 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (+1*ux + +0*uy + +1*uz) * ITHETA;
  f[5 + NM * 1] = rho * W110 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (-1*ux + +1*uy + +1*uz) * ITHETA;
  f[6 + NM * 1] = rho * W111 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  // g2
  cdotu = (+0*ux + -1*uy + -1*uz) * ITHETA;
  f[0 + NM * 2] = rho * W110 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (-1*ux + -1*uy + -1*uz) * ITHETA;
  f[1 + NM * 2] = rho * W111 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (+0*ux + +0*uy + +0*uz) * ITHETA;
  f[2 + NM * 2] = rho * W000 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (-1*ux + +0*uy + +0*uz) * ITHETA;
  f[3 + NM * 2] = rho * W100 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (+1*ux + -1*uy + +0*uz) * ITHETA;
  f[4 + NM * 2] = rho * W110 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (+0*ux + -1*uy + +0*uz) * ITHETA;
  f[5 + NM * 2] = rho * W100 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (-1*ux + -1*uy + +0*uz) * ITHETA;
  f[6 + NM * 2] = rho * W110 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  // g3
  cdotu = (+0*ux + +1*uy + +1*uz) * ITHETA;
  f[0 + NM * 3] = rho * W110 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (+1*ux + +1*uy + +1*uz) * ITHETA;
  f[1 + NM * 3] = rho * W111 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (+1*ux + +0*uy + +0*uz) * ITHETA;
  f[3 + NM * 3] = rho * W100 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (-1*ux + +1*uy + +0*uz) * ITHETA;
  f[4 + NM * 3] = rho * W110 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (+0*ux + +1*uy + +0*uz) * ITHETA;
  f[5 + NM * 3] = rho * W100 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  cdotu = (+1*ux + +1*uy + +0*uz) * ITHETA;
  f[6 + NM * 3] = rho * W110 * (1 + cdotu - 0.5 * udotu + 0.5 * cdotu * cdotu);

  return;
}

void
collide_d3q27(VARTYPE* T, size_t npx, size_t npy, size_t npz, size_t np)
{
  VARTYPE feq[NM * NG] = {0};
  VARTYPE rho, ux, uy, uz;
  for (size_t z = np; z <= npz-(np+1); z++) {
    for (size_t y = np; y <= npy-(np+1); y++) {
      for (size_t x = np; x <= npx-(np+1); x++) {
        for (size_t g = 0; g < NG; g++) {
          for (size_t m = 0; m < NM; m++) {
            feq[m + NM * g] = T[idx(m,x,y,z,g,npx,npy,npz)];
          }
        }
        fill_moments(feq, &rho, &ux, &uy, &uz);
        fill_feq(rho, ux, uy, uz, feq);
        for (size_t g = 0; g < NG; g++) {
          for (size_t m = 0; m < NM; m++) {
            T[idx(m,x,y,z,g,npx,npy,npz)] = (1. - BETA2) * T[idx(m,x,y,z,g,npx,npy,npz)]
                                            + BETA2 * feq[m + NM * g];
          }
        }
      }
    }
  }
  return;
}

void
collide_blocks_d3q27(VARTYPE* T, size_t npx, size_t npy, size_t npz, size_t np, size_t nbx, size_t nby, size_t nbz)
{
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();
    std::size_t blocks_per_thread = (nbx*nby*nbz)/nthreads;

    std::size_t rem_blocks = 0;
    if(tid == nthreads - 1) {
      rem_blocks = (nbx*nby*nbz) % (nthreads);
    }

    const size_t bsize = NM * npx * npy * npz * NG;

    for (size_t bz = 0; bz < nbz; bz++) {
      for (size_t by = 0; by < nby; by++) {
        for (size_t bx = 0; bx < nbx; bx++) {
          const size_t bidx  = (bx + nbx * (by + nby * bz));

          if(bidx >= tid*blocks_per_thread and bidx < (tid+1)*blocks_per_thread + rem_blocks )
          {
            collide_d3q27(&T[bsize * bidx], npx, npy, npz, np);
          }
        }
      }
    }
  }
  return;
}

void
advect_d3q27(VARTYPE* T, size_t npx, size_t npy, size_t npz, size_t np)
{
  for (size_t z = np; z <= npz-(np+1); z++) {
    for (size_t y = np; y <= npy-(np+1); y++) {
      for (size_t x = np; x <= npx-(np+1); x++) {
        T[idx(0,x,y,z,0,npx,npy,npz)] = T[idx(0,x-1,y-1,z+1,0,npx,npy,npz)];
        T[idx(1,x,y,z,0,npx,npy,npz)] = T[idx(1,x+0,y-1,z+1,0,npx,npy,npz)];
        T[idx(2,x,y,z,0,npx,npy,npz)] = T[idx(2,x+1,y-1,z+1,0,npx,npy,npz)];
        T[idx(3,x,y,z,0,npx,npy,npz)] = T[idx(3,x-1,y+0,z+1,0,npx,npy,npz)];
        T[idx(4,x,y,z,0,npx,npy,npz)] = T[idx(4,x+0,y+0,z+1,0,npx,npy,npz)];
        T[idx(5,x,y,z,0,npx,npy,npz)] = T[idx(5,x+1,y+0,z+1,0,npx,npy,npz)];
        T[idx(6,x,y,z,0,npx,npy,npz)] = T[idx(6,x-1,y+1,z+1,0,npx,npy,npz)];
      }
    }
  }
  for (size_t z = npz-(np+1); z >= np; z--) {
    for (size_t y = npy-(np+1); y >= np; y--) {
      for (size_t x = npx-(np+1); x >= np; x--) {
        T[idx(0,x,y,z,1,npx,npy,npz)] = T[idx(0,x+1,y+1,z-1,1,npx,npy,npz)];
        T[idx(1,x,y,z,1,npx,npy,npz)] = T[idx(1,x+0,y+1,z-1,1,npx,npy,npz)];
        T[idx(2,x,y,z,1,npx,npy,npz)] = T[idx(2,x-1,y+1,z-1,1,npx,npy,npz)];
        T[idx(3,x,y,z,1,npx,npy,npz)] = T[idx(3,x+1,y+0,z-1,1,npx,npy,npz)];
        T[idx(4,x,y,z,1,npx,npy,npz)] = T[idx(4,x+0,y+0,z-1,1,npx,npy,npz)];
        T[idx(5,x,y,z,1,npx,npy,npz)] = T[idx(5,x-1,y+0,z-1,1,npx,npy,npz)];
        T[idx(6,x,y,z,1,npx,npy,npz)] = T[idx(6,x+1,y-1,z-1,1,npx,npy,npz)];
      }
    }
  }
  for (size_t z = np; z <= npz-(np+1); z++) {
    for (size_t y = np; y <= npy-(np+1); y++) {
      for (size_t x = np; x <= npx-(np+1); x++) {
        T[idx(0,x,y,z,2,npx,npy,npz)] = T[idx(0,x+0,y+1,z+1,2,npx,npy,npz)];
        T[idx(1,x,y,z,2,npx,npy,npz)] = T[idx(1,x+1,y+1,z+1,2,npx,npy,npz)];
        T[idx(2,x,y,z,2,npx,npy,npz)] = T[idx(2,x+0,y+0,z+0,2,npx,npy,npz)];
        T[idx(3,x,y,z,2,npx,npy,npz)] = T[idx(3,x+1,y+0,z+0,2,npx,npy,npz)];
        T[idx(4,x,y,z,2,npx,npy,npz)] = T[idx(4,x-1,y+1,z+0,2,npx,npy,npz)];
        T[idx(5,x,y,z,2,npx,npy,npz)] = T[idx(5,x+0,y+1,z+0,2,npx,npy,npz)];
        T[idx(6,x,y,z,2,npx,npy,npz)] = T[idx(6,x+1,y+1,z+0,2,npx,npy,npz)];
      }
    }
  }
  for (size_t z = npz-(np+1); z >= np; z--) {
    for (size_t y = npy-(np+1); y >= np; y--) {
      for (size_t x = npx-(np+1); x >= np; x--) {
        T[idx(0,x,y,z,3,npx,npy,npz)] = T[idx(0,x+0,y-1,z-1,3,npx,npy,npz)];
        T[idx(1,x,y,z,3,npx,npy,npz)] = T[idx(1,x-1,y-1,z-1,3,npx,npy,npz)];
        T[idx(2,x,y,z,3,npx,npy,npz)] = T[idx(2,x+0,y+0,z+0,3,npx,npy,npz)];
        T[idx(3,x,y,z,3,npx,npy,npz)] = T[idx(3,x-1,y+0,z+0,3,npx,npy,npz)];
        T[idx(4,x,y,z,3,npx,npy,npz)] = T[idx(4,x+1,y-1,z+0,3,npx,npy,npz)];
        T[idx(5,x,y,z,3,npx,npy,npz)] = T[idx(5,x+0,y-1,z+0,3,npx,npy,npz)];
        T[idx(6,x,y,z,3,npx,npy,npz)] = T[idx(6,x-1,y-1,z+0,3,npx,npy,npz)];
      }
    }
  }
  return;
}

void
advect_blocks_d3q27(VARTYPE* T, size_t npx, size_t npy, size_t npz, size_t np, size_t nbx, size_t nby, size_t nbz)
{
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();
    std::size_t blocks_per_thread = (nbx*nby*nbz)/nthreads;

    std::size_t rem_blocks = 0;
    if(tid == nthreads - 1) {
      rem_blocks = (nbx*nby*nbz) % (nthreads);
    }

    const size_t bsize = NM * npx * npy * npz * NG;

    for (size_t bz = 0; bz < nbz; bz++) {
      for (size_t by = 0; by < nby; by++) {
        for (size_t bx = 0; bx < nbx; bx++) {
          const size_t bidx  = (bx + nbx * (by + nby * bz));

          if(bidx >= tid*blocks_per_thread and bidx < (tid+1)*blocks_per_thread + rem_blocks )
          {
            VARTYPE* b = &T[bsize * bidx]; 
            
            // Communicate
            size_t plus_ngb_b, neg_ngb_b, b_ngb_id; 

            // X Comm
            if( bx == 0 ) // first block setting periodic in -ve dir
              neg_ngb_b = nbx - 1;
            else
              neg_ngb_b = bx - 1;

            if( bx == nbx - 1 ) // last block setting periodic in +ve dir
              plus_ngb_b = 0;
            else
              plus_ngb_b = bx + 1;

            b_ngb_id = (neg_ngb_b + nbx * (by + nby * bz));
            VARTYPE* b_neg_ngb = &T[bsize * b_ngb_id];
            for (size_t z = 0; z < npz; z++) {
              for (size_t y = 0; y < npy; y++) {

                for (size_t g = 0; g < NG; g++) {
                  for (size_t m = 0; m < NM; m++) {
                    b[idx(m,0,y,z,g,npx,npy,npz)] = b_neg_ngb[idx(m,npx-(np+1),y,z,g,npx,npy,npz)];
                  }
                }

              }
            }

            b_ngb_id = (plus_ngb_b + nbx * (by + nby * bz));
            VARTYPE* b_plus_ngb = &T[bsize * b_ngb_id];
            for (size_t z = 0; z < npz; z++) {
              for (size_t y = 0; y < npy; y++) {

                for (size_t g = 0; g < NG; g++) {
                  for (size_t m = 0; m < NM; m++) {
                    b[idx(m,npx-1,y,z,g,npx,npy,npz)] = b_plus_ngb[idx(m,np,y,z,g,npx,npy,npz)];
                  }
                }

              }
            }
            // Y Comm
            if( by == 0 ) // first block setting periodic in -ve dir
              neg_ngb_b = nby - 1;
            else
              neg_ngb_b = by - 1;

            if( by == nby - 1 ) // last block setting periodic in +ve dir
              plus_ngb_b = 0;
            else
              plus_ngb_b = by + 1;

            b_ngb_id = (bx + nbx * (neg_ngb_b + nby * bz));
            b_neg_ngb = &T[bsize * b_ngb_id];

            for (size_t z = 0; z < npz; z++) {
              for (size_t x = 0; x < npx; x++) {

                for (size_t g = 0; g < NG; g++) {
                  for (size_t m = 0; m < NM; m++) {
                    b[idx(m,x,0,z,g,npx,npy,npz)] = b_neg_ngb[idx(m,x,npy-(np+1),z,g,npx,npy,npz)];
                  }
                }

              }
            }

            b_ngb_id = (bx + nbx * (plus_ngb_b + nby * bz));
            b_plus_ngb = &T[bsize * b_ngb_id];

            for (size_t z = 0; z < npz; z++) {
              for (size_t x = 0; x < npx; x++) {

                  for (size_t g = 0; g < NG; g++) {
                    for (size_t m = 0; m < NM; m++) {
                      b[idx(m,x,npy-1,z,g,npx,npy,npz)] = b_plus_ngb[idx(m,x,np,z,g,npx,npy,npz)];
                    }
                  }

              }
            }
            // Z comm
            if( bz == 0 ) // first block setting periodic in -ve dir
              neg_ngb_b = nbz - 1;
            else
              neg_ngb_b = bz - 1;

            if( bz == nbz - 1 ) // last block setting periodic in +ve dir
              plus_ngb_b = 0;
            else
              plus_ngb_b = bz + 1;

            b_ngb_id = (bx + nbx * (by + nby * neg_ngb_b));
            b_neg_ngb = &T[bsize * b_ngb_id];

            for (size_t y = 0; y < npy; y++) {
              for (size_t x = 0; x < npx; x++) {

                for (size_t g = 0; g < NG; g++) {
                  for (size_t m = 0; m < NM; m++) {
                    b[idx(m,x,y,0,g,npx,npy,npz)] = b_neg_ngb[idx(m,x,y,npz-(np+1),g,npx,npy,npz)];
                  }
                }

              }
            }

            b_ngb_id = (bx + nbx * (by + nby * plus_ngb_b));
            b_plus_ngb = &T[bsize * b_ngb_id];

            for (size_t y = 0; y < npy; y++) {
              for (size_t x = 0; x < npx; x++) {

                for (size_t g = 0; g < NG; g++) {
                  for (size_t m = 0; m < NM; m++) {
                    b[idx(m,x,y,npz-1,g,npx,npy,npz)] = b_plus_ngb[idx(m,x,y,np,g,npx,npy,npz)];
                  }
                }

              }
            }

            advect_d3q27(&T[bsize * bidx], npx, npy, npz, np);
          }
        }
      }
    }
  }
  return;
}

void
initialise(VARTYPE* T, size_t npx, size_t npy, size_t npz, size_t np, VARTYPE rho, VARTYPE ux, VARTYPE uy, VARTYPE uz)
{
  VARTYPE feq[NM * NG] = {0};
  for (size_t z = 0; z <= npz-1; z++) {
    for (size_t y = 0; y <= npy-1; y++) {
      for (size_t x = 0; x <= npx-1; x++) {
        fill_feq(rho, ux, uy, uz, feq);
        for (size_t g = 0; g < NG; g++) {
          for (size_t m = 0; m < NM; m++) {
            T[idx(m,x,y,z,g,npx,npy,npz)] = feq[m + NM * g];
          }
        }
      }
    }
  }
  return;
}

void
initialise_blocks(VARTYPE* T, size_t npx, size_t npy, size_t npz, size_t np, size_t nbx, size_t nby, size_t nbz, VARTYPE rho, VARTYPE ux, VARTYPE uy, VARTYPE uz)
{
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();
    std::size_t blocks_per_thread = (nbx*nby*nbz)/nthreads;

    std::size_t rem_blocks = 0;
    if(tid == nthreads - 1) {
      rem_blocks = (nbx*nby*nbz) % (nthreads);
    }

    const size_t bsize = NM * npx * npy * npz * NG;
    for (size_t bz = 0; bz < nbz; bz++) {
      for (size_t by = 0; by < nby; by++) {
        for (size_t bx = 0; bx < nbx; bx++) {
          const size_t bidx  = (bx + nbx * (by + nby * bz));

          if(bidx >= tid*blocks_per_thread and bidx < (tid+1)*blocks_per_thread + rem_blocks )
          {
            initialise(&T[bsize * bidx], npx, npy, npz, np, rho, ux, uy, uz);
          }
        }
      }
    }
  }
  return;
}

std::tuple<double, double, VARTYPE, VARTYPE, VARTYPE, VARTYPE>
run(size_t nx, size_t ny, size_t nz, size_t nbx, size_t nby, size_t nbz, size_t nt)
{
  const size_t np = 1;
  const size_t npx = nx + 2 * np;
  const size_t npy = ny + 2 * np;
  const size_t npz = nz + 2 * np;

  const size_t alloc_bytes = NM * (npx * npy * npz) * NG * (nbx * nby * nbz) * sizeof(VARTYPE);
  const size_t moffset = 512 - 1;
  VARTYPE* uT = (VARTYPE*)malloc(alloc_bytes + moffset);
  VARTYPE *T = (VARTYPE*)(((size_t)uT + moffset) & ~moffset);
  
  initialise_blocks(T, npx, npy, npz, np, nbx, nby, nbz, 1., 0., 0., 0.);

  double collide_time = 0, advect_time = 0;
  for (size_t t = 0; t < nt; t++) {
    auto tic1 = std::chrono::high_resolution_clock::now();
    collide_blocks_d3q27(T, npx, npy, npz, np, nbx, nby, nbz);
    auto tic2 = std::chrono::high_resolution_clock::now();
    advect_blocks_d3q27(T, npx, npy, npz, np, nbx, nby, nbz);
    auto tic3 = std::chrono::high_resolution_clock::now();
    auto elapsed_time = (std::chrono::duration<double, std::nano>(
                       tic2 - tic1).count())*1E-9;
    collide_time += elapsed_time;

    elapsed_time = (std::chrono::duration<double, std::nano>(
                       tic3 - tic2).count())*1E-9;

    advect_time += elapsed_time;
  }

  VARTYPE feq[NM * NG];
  for (size_t g = 0; g < NG; g++) {
    for (size_t m = 0; m < NM; m++) {
      feq[m + NM * g] = T[idx(m,np,np,np,g,npx,npy,npz)];
    }
  }
  VARTYPE rho, ux, uy, uz;
  fill_moments(feq, &rho, &ux, &uy, &uz);

  free(uT);

  return std::make_tuple(collide_time, advect_time, rho, ux, uy, uz);
}

int
main(int argc, char* argv[])
{
  const size_t nx  = atoi(argv[1]);
  const size_t ny  = atoi(argv[2]);
  const size_t nz  = atoi(argv[3]);
  const size_t nbx = atoi(argv[4]);
  const size_t nby = atoi(argv[5]);
  const size_t nbz = atoi(argv[6]);
  const size_t nt  = atoi(argv[7]);

  auto res = run(nx, ny, nz, nbx, nby, nbz, nt);
  std::cout << nx << ", ";
  std::cout << ny << ", ";
  std::cout << nz << ", ";
  std::cout << nbx << ", ";
  std::cout << nby << ", ";
  std::cout << nbz << ", ";
  std::cout << nt << ", ";
  std::cout << std::get<0>(res) << ", "; // collide time
  std::cout << std::get<1>(res) << ", "; // advect time
  std::cout << std::get<2>(res) << ", "; // rho
  std::cout << std::get<3>(res) << ", "; // ux
  std::cout << std::get<4>(res) << ", "; // uy
  std::cout << std::get<5>(res) << ", "; // uz
  std::cout << std::endl;

  return 0;
}