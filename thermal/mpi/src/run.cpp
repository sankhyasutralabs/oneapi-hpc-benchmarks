#include "initialise.hpp"
#include "diffuse.hpp"
#include <mpi.h>
#include <cstdlib>
#include <utility>

inline size_t
idx(size_t x, size_t y, size_t z, size_t npx, size_t npy, size_t npz)
{
  return x + npx * (y + npy * z);
}

void
swap(VARTYPE** a, VARTYPE** b)
{
  VARTYPE* tmp = *a;
  *a = *b;
  *b = tmp;
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
    swap(&T, &Tnew);
  }
  MPI_Barrier(mpi_comm);
  double toc = MPI_Wtime();

  VARTYPE sample_val = T[idx(np,np,np,npx,npy,npz)];

  free(T);
  free(Tnew);

  return std::make_pair(toc - tic, sample_val);
}

