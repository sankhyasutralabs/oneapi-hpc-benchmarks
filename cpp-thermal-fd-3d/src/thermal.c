#include "initialise.h"
#include "diffuse.h"
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

inline size_t
idx(size_t x, size_t y, size_t z, size_t npx, size_t npy, size_t npz)
{
  return x + npx * (y + npy * z);
}

void
swap(REAL** a, REAL** b)
{
  REAL* tmp = *a;
  *a = *b;
  *b = tmp;
  return;
}

int
main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  int mpi_rank, mpi_size;
  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  MPI_Comm_rank(mpi_comm, &mpi_rank);
  MPI_Comm_size(mpi_comm, &mpi_size);

  const size_t nx = atoi(argv[1]);
  const size_t ny = atoi(argv[2]);
  const size_t nz = atoi(argv[3]);
  const size_t nt = atoi(argv[4]);
  const size_t np = 1;
  const size_t npx = nx + 2 * np;
  const size_t npy = ny + 2 * np;
  const size_t npz = nz + 2 * np;

  const size_t alloc_bytes = npx * npy * npz * sizeof(REAL);
  REAL* T = (REAL*)malloc(alloc_bytes);
  REAL* Tnew = (REAL*)malloc(alloc_bytes);

  initialise(T, npx, npy, npz, np, 0., 100.);
  initialise(Tnew, npx, npy, npz, np, 0., 100.);

  MPI_Barrier(mpi_comm);
  double tic = MPI_Wtime();
  for (size_t t = 0; t < nt; t++) {
    diffuse(T, Tnew, npx, npy, npz, np, 0.1);
    swap(&T, &Tnew);
  }
  MPI_Barrier(mpi_comm);
  double toc = MPI_Wtime();
  double elapsed = toc - tic;

  if (0 == mpi_rank) {
    fprintf(stdout, "%d, %lu, %lu, %lu, %f, %f\n", mpi_size, nx, ny, nz, elapsed, T[idx(np,np,np,npx,npy,npz)]);
  }

  free(T);
  free(Tnew);

  MPI_Finalize();
  return 0;
}
