#include <chrono>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <omp.h>
#include <utility>

#ifndef VARTYPE
  #define VARTYPE double
#endif

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
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();
    std::size_t blocks_per_thread = (nbx*nby*nbz)/nthreads;
    const size_t bsize = npx * npy * npz;

    std::size_t rem_blocks = 0;
    if(tid == nthreads - 1) {
      rem_blocks = (nbx*nby*nbz) % (nthreads);
    }

    for (size_t bz = 0; bz < nbz; bz++) {
      for (size_t by = 0; by < nby; by++) {
        for (size_t bx = 0; bx < nbx; bx++) {
          const size_t bidx  = (bx + nbx * (by + nby * bz));
          VARTYPE* b = &T[bsize * bidx]; 
          VARTYPE* bnew = &Tnew[bsize * bidx];

          if(bidx >= tid*blocks_per_thread and bidx < (tid+1)*blocks_per_thread + rem_blocks ) {
            // Communicate
            size_t plus_ngb_b, neg_ngb_b, b_ngb_id; 

            // X Comm
            if( bx == 0 )
              neg_ngb_b = nbx - 1;
            else
              neg_ngb_b = bx - 1;

            if( bx == nbx - 1 )
              plus_ngb_b = 0;
            else
              plus_ngb_b = bx + 1;

            b_ngb_id = (neg_ngb_b + nbx * (by + nby * bz));
            VARTYPE* b_neg_ngb = &T[bsize * b_ngb_id];

            for (size_t z = 0; z < npz; z++) {
              for (size_t y = 0; y < npy; y++) {
                if(bx != 0){
                  b[idx(0,y,z,npx,npy,npz)] = b_neg_ngb[idx(npx-(np+1),y,z,npx,npy,npz)];
                }
              }
            }

            b_ngb_id = (plus_ngb_b + nbx * (by + nby * bz));
            VARTYPE* b_plus_ngb = &T[bsize * b_ngb_id];

            for (size_t z = 0; z < npz; z++) {
              for (size_t y = 0; y < npy; y++) {
                if(bx != (nbx -1)){
                  b[idx(npx-1,y,z,npx,npy,npz)] = b_plus_ngb[idx(np,y,z,npx,npy,npz)];
                }
              }
            }

            // Y Comm
            if( by == 0 )
              neg_ngb_b = nby - 1;
            else
              neg_ngb_b = by - 1;

            if( by == nby - 1 )
              plus_ngb_b = 0;
            else
              plus_ngb_b = by + 1;

            b_ngb_id = (bx + nbx * (neg_ngb_b + nby * bz));
            b_neg_ngb = &T[bsize * b_ngb_id];

            for (size_t z = 0; z < npz; z++) {
              for (size_t x = 0; x < npx; x++) {
                if (by != 0) {
                  b[idx(x,0,z,npx,npy,npz)] = b_neg_ngb[idx(x,npy-(np+1),z,npx,npy,npz)];
                } 
              }
            }

            b_ngb_id = (bx + nbx * (plus_ngb_b + nby * bz));
            b_plus_ngb = &T[bsize * b_ngb_id];
            for (size_t z = 0; z < npz; z++) {
              for (size_t x = 0; x < npx; x++) {
                if (by != (nby -1)) {
                  b[idx(x,npy-1,z,npx,npy,npz)] = b_plus_ngb[idx(x,np,z,npx,npy,npz)];
                }
              }
            }

            // Z comm
            if( bz == 0 )
              neg_ngb_b = nbz - 1;
            else
              neg_ngb_b = bz - 1;

            if( bz == nbz - 1 )
              plus_ngb_b = 0;
            else
              plus_ngb_b = bz + 1;

            b_ngb_id = (bx + nbx * (by + nby * neg_ngb_b));
            b_neg_ngb = &T[bsize * b_ngb_id];

            for (size_t y = 0; y < npy; y++) {
              for (size_t x = 0; x < npx; x++) {
                if (bz != 0) {
                  b[idx(x,y,0,npx,npy,npz)] = b_neg_ngb[idx(x,y,npz-(np+1),npx,npy,npz)];
                }
              }
            }

            b_ngb_id = (bx + nbx * (by + nby * plus_ngb_b));
            b_plus_ngb = &T[bsize * b_ngb_id];

            for (size_t y = 0; y < npy; y++) {
              for (size_t x = 0; x < npx; x++) {
                if (bz != (nbz -1)) {
                  b[idx(x,y,npz-1,npx,npy,npz)] = b_plus_ngb[idx(x,y,np,npx,npy,npz)];
                }
              }
            }

            diffuse_d3q7(b, bnew, npx, npy, npz, np, cfl);
          }
        }
      }
    }
  }
  return;
}

void
initialise_blocks(VARTYPE* T, VARTYPE* Tnew, size_t npx, size_t npy, size_t npz, size_t np, size_t nbx, size_t nby, size_t nbz, VARTYPE Tbulk, VARTYPE Tbc)
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

    const size_t bsize = npx * npy * npz;

    for (int bz = 0; bz < nbz; bz++) {
      for (int by = 0; by < nby; by++) {
        for (int bx = 0; bx < nbx; bx++) {
          const size_t bidx  = (bx + nbx * (by + nby * bz));
          VARTYPE* b = &T[bsize * bidx]; 
          VARTYPE* bnew = &Tnew[bsize * bidx];

          // if(bidx % nthreads == tid) {
          if(bidx >= tid*blocks_per_thread and bidx < (tid+1)*blocks_per_thread + rem_blocks ) {
            for (size_t z = 0; z <= npz-1; z++) {
              for (size_t y = 0; y <= npy-1; y++) {
                for (size_t x = 0; x <= npx-1; x++) {
                  b[idx(x,y,z,npx,npy,npz)] = Tbulk;
                  bnew[idx(x,y,z,npx,npy,npz)] = Tbulk;

                  if(bx == nbx-1){
                    if(x == npx-1) {
                      b[idx(x,y,z,npx,npy,npz)] = Tbc;
                      bnew[idx(x,y,z,npx,npy,npz)] = Tbc;
                    }
                  }

                  if(bx == 0){
                    if(x == 0) {
                      b[idx(x,y,z,npx,npy,npz)] = Tbc;
                      bnew[idx(x,y,z,npx,npy,npz)] = Tbc;
                    }
                  }

                  if(by == nby-1){
                    if(y == npy-1) {
                      b[idx(x,y,z,npx,npy,npz)] = Tbc;
                      bnew[idx(x,y,z,npx,npy,npz)] = Tbc;
                    }
                  }

                  if(by == 0){
                    if(y == 0) {
                      b[idx(x,y,z,npx,npy,npz)] = Tbc;
                      bnew[idx(x,y,z,npx,npy,npz)] = Tbc;
                    }
                  }

                  if(bz == nbz-1){
                    if(z == npz-1) {
                      b[idx(x,y,z,npx,npy,npz)] = Tbc;
                      bnew[idx(x,y,z,npx,npy,npz)] = Tbc;
                    }
                  }

                  if(bz == 0){
                    if(z == 0) {
                      b[idx(x,y,z,npx,npy,npz)] = Tbc;
                      bnew[idx(x,y,z,npx,npy,npz)] = Tbc;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return;
}

std::pair<float, VARTYPE>
run(size_t nx, size_t ny, size_t nz, size_t nbx, size_t nby, size_t nbz, size_t nt)
{
  const size_t np = 1;
  const size_t npx = nx + 2 * np;
  const size_t npy = ny + 2 * np;
  const size_t npz = nz + 2 * np;

  const size_t alloc_bytes = (npx * npy * npz) * (nbx * nby * nbz) * sizeof(VARTYPE);
  VARTYPE* T = (VARTYPE*)malloc(alloc_bytes);
  VARTYPE* Tnew = (VARTYPE*)malloc(alloc_bytes);
    
  initialise_blocks(T, Tnew, npx, npy, npz, np, nbx, nby, nbz, 10., 100.);

  double total_time = 0.0;
  auto start_time = std::chrono::high_resolution_clock::now();

  for (size_t t = 0; t < nt; t++) {
    diffuse_blocks_d3q7(T, Tnew, npx, npy, npz, np, nbx, nby, nbz, 0.1);

    std::swap(T, Tnew);
  }

  total_time += std::chrono::duration<double, std::nano>(
                 std::chrono::high_resolution_clock::now() - start_time).count();
  total_time *= 1E-9; //nano to seconds


  VARTYPE sample_val = T[idx(np,np,np,npx,npy,npz)];

  free(T);
  free(Tnew);

  return std::make_pair(total_time, sample_val);
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
  std::cout << res.first << ", ";
  std::cout << res.second;
  std::cout << std::endl;

  return 0;
}