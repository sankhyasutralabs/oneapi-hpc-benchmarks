#include <mpi.h>
#include <cstdlib>
#include <iostream>
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
  const size_t bsize = npx * npy * npz;
  for (size_t b = 0; b < (nbx * nby * nbz); b++) {
    diffuse_d3q7(&T[bsize * b], &Tnew[bsize * b], npx, npy, npz, np, cfl);
  }
  return;
}

void
initialise(VARTYPE* T, size_t npx, size_t npy, size_t npz, size_t np, VARTYPE Tbulk, VARTYPE Tbc)
{
  for (size_t z = 0; z <= npz-1; z++) {
    for (size_t y = 0; y <= npy-1; y++) {
      for (size_t x = 0; x <= npx-1; x++) {
        T[idx(x,y,z,npx,npy,npz)] = Tbc;
        if (z >= np and z <= npz-(np+1)) {
          if (y >= np and y <= npy-(np+1)) {
            if (x >= np and x <= npx-(np+1)) {
              T[idx(x,y,z,npx,npy,npz)] = Tbulk;
            }
          }
        }
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

std::pair<double, VARTYPE>
run(size_t nx, size_t ny, size_t nz, size_t nbx, size_t nby, size_t nbz, size_t nt, MPI_Comm mpi_comm)
{
  const size_t np = 1;
  const size_t npx = nx + 2 * np;
  const size_t npy = ny + 2 * np;
  const size_t npz = nz + 2 * np;

  const size_t alloc_bytes = (npx * npy * npz) * (nbx * nby * nbz) * sizeof(VARTYPE);
  VARTYPE* T = (VARTYPE*)malloc(alloc_bytes);
  VARTYPE* Tnew = (VARTYPE*)malloc(alloc_bytes);
  
  initialise_blocks(T, npx, npy, npz, np, nbx, nby, nbz, 10., 100.);
  initialise_blocks(Tnew, npx, npy, npz, np, nbx, nby, nbz, 10., 100.);

  MPI_Barrier(mpi_comm);
  double tic = MPI_Wtime();
  for (size_t t = 0; t < nt; t++) {
    diffuse_blocks_d3q7(T, Tnew, npx, npy, npz, np, nbx, nby, nbz, 0.1);
    MPI_Barrier(mpi_comm);
    std::swap(T, Tnew);
  }
  MPI_Barrier(mpi_comm);
  double toc = MPI_Wtime();

  VARTYPE sample_val = T[idx(np,np,np,npx,npy,npz)];

  free(T);
  free(Tnew);

  return std::make_pair(toc - tic, sample_val);
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
    std::cout << res.first << ", ";
    std::cout << res.second;
    std::cout << std::endl;
  }

  MPI_Finalize();
  return 0;
}
