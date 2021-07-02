#include <mpi.h>
#include <cstdlib>
#include <iostream>
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
  const size_t bsize = NM * npx * npy * npz * NG;
  for (size_t b = 0; b < (nbx * nby * nbz); b++) {
    collide_d3q27(&T[bsize * b], npx, npy, npz, np);
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
  const size_t bsize = NM * npx * npy * npz * NG;
  for (size_t b = 0; b < (nbx * nby * nbz); b++) {
    advect_d3q27(&T[bsize * b], npx, npy, npz, np);
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
  const size_t bsize = NM * npx * npy * npz * NG;
  for (size_t b = 0; b < (nbx * nby * nbz); b++) {
    initialise(&T[bsize * b], npx, npy, npz, np, rho, ux, uy, uz);
  }
  return;
}

std::tuple<float, float, VARTYPE, VARTYPE, VARTYPE, VARTYPE>
run(size_t nx, size_t ny, size_t nz, size_t nbx, size_t nby, size_t nbz, size_t nt, MPI_Comm mpi_comm)
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

  float collide_time = 0, advect_time = 0;
  for (size_t t = 0; t < nt; t++) {
    MPI_Barrier(mpi_comm);
    float tic1 = MPI_Wtime();
    collide_blocks_d3q27(T, npx, npy, npz, np, nbx, nby, nbz);
    MPI_Barrier(mpi_comm);
    float tic2 = MPI_Wtime();
    advect_blocks_d3q27(T, npx, npy, npz, np, nbx, nby, nbz);
    MPI_Barrier(mpi_comm);
    float tic3 = MPI_Wtime();
    collide_time += tic2 - tic1;
    advect_time += tic3 - tic2;
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
  MPI_Init(&argc, &argv);
  int mpi_rank, mpi_size;
  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  MPI_Comm_rank(mpi_comm, &mpi_rank);
  MPI_Comm_size(mpi_comm, &mpi_size);

  const size_t nx  = atoi(argv[1]);
  const size_t ny  = atoi(argv[2]);
  const size_t nz  = atoi(argv[3]);
  const size_t nbx = atoi(argv[4]);
  const size_t nby = atoi(argv[5]);
  const size_t nbz = atoi(argv[6]);
  const size_t nt  = atoi(argv[7]);

  if ((nx % 8) != 0) {
    if (0 == mpi_rank) {
      std::cerr << "error: nx must be a multiple of 8" << std::endl;
    }
    MPI_Abort(mpi_comm, 2);
    exit(1);
  }

  auto res = run(nx, ny, nz, nbx, nby, nbz, nt, mpi_comm);
  if (0 == mpi_rank) {
    std::cout << nx << ", ";
    std::cout << ny << ", ";
    std::cout << nz << ", ";
    std::cout << nbx << ", ";
    std::cout << nby << ", ";
    std::cout << nbz << ", ";
    std::cout << nt << ", ";
    std::cout << mpi_size << ", ";
    std::cout << std::get<0>(res) << ", "; // collide time
    std::cout << std::get<1>(res) << ", "; // advect time
    std::cout << std::get<2>(res) << ", "; // rho
    std::cout << std::get<3>(res) << ", "; // ux
    std::cout << std::get<4>(res) << ", "; // uy
    std::cout << std::get<5>(res) << ", "; // uz
    std::cout << std::endl;
  }

  MPI_Finalize();
  return 0;
}
