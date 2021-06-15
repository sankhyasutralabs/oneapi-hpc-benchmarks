#include <mpi.h>
#include <CL/sycl.hpp>
#include <dpc_common.hpp>
#include <tuple>
#include <cstdlib>
#include <iostream>

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

std::tuple<double, double, VARTYPE, VARTYPE, VARTYPE, VARTYPE>
run(size_t nx, size_t ny, size_t nz, size_t nbx, size_t nby, size_t nbz, size_t nt, MPI_Comm mpi_comm, sycl::device& d)
{
  const size_t np = 1;
  const size_t npx = nx + 2 * np;
  const size_t npy = ny + 2 * np;
  const size_t npz = nz + 2 * np;

  sycl::property_list properties{ sycl::property::queue::in_order() };
  sycl::queue q(d, dpc_common::exception_handler, properties);

  const size_t alloc_bytes = NM * (npx * npy * npz) * NG * (nbx * nby * nbz) * sizeof(VARTYPE);
  VARTYPE* T = (VARTYPE*)sycl::malloc_device(alloc_bytes, q);
  VARTYPE* Tnew = (VARTYPE*)sycl::malloc_device(alloc_bytes, q);

  sycl::range<3> threads = { npx * nbx, npy * nby, npz * nbz };
  sycl::range<3> wg = { npx, npy, npz };
  const size_t bsize = NM * (npx * npy * npz) * NG;

  // initialise
  auto initialise_blocks = [&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<3>(threads, wg), [=](sycl::nd_item<3> it) {
      const size_t bx = it.get_group(0);
      const size_t by = it.get_group(1);
      const size_t bz = it.get_group(2);
      const size_t bidx  = (bx + nbx * (by + nby * bz));
      VARTYPE* b = &T[bsize * bidx]; 
      VARTYPE* bnew = &Tnew[bsize * bidx];

      const int x = it.get_local_id(0);
      const int y = it.get_local_id(1);
      const int z = it.get_local_id(2);

      VARTYPE feq[NM * NG] = {0};
      VARTYPE rho = 1., ux = 0., uy = 0., uz = 0.;
      fill_feq(rho, ux, uy, uz, feq);
      for (size_t g = 0; g < NG; g++) {
        for (size_t m = 0; m < NM; m++) {
          b[idx(m,x,y,z,g,npx,npy,npz)] = feq[m + NM * g];
          bnew[idx(m,x,y,z,g,npx,npy,npz)] = feq[m + NM * g];
        }
      }
    });
  };

  try {
    q.submit(initialise_blocks);
    q.wait();
  } catch (sycl::exception const& ex) {
    std::cerr << "dpcpp error: " << ex.what() << std::endl;
  }

  // collide
  auto collide_blocks_d3q27 = [&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<3>(threads, wg), [=](sycl::nd_item<3> it) {
      const size_t bx = it.get_group(0);
      const size_t by = it.get_group(1);
      const size_t bz = it.get_group(2);
      const size_t bidx  = (bx + nbx * (by + nby * bz));
      VARTYPE* b = &T[bsize * bidx]; 

      const int x = it.get_local_id(0);
      const int y = it.get_local_id(1);
      const int z = it.get_local_id(2);

      if (z >= np and z <= npz-(np+1)) {
        if (y >= np and y <= npy-(np+1)) {
          if (x >= np and x <= npx-(np+1)) {
            VARTYPE feq[NM * NG] = {0};
            VARTYPE rho, ux, uy, uz;
            for (size_t g = 0; g < NG; g++) {
              for (size_t m = 0; m < NM; m++) {
                feq[m + NM * g] = b[idx(m,x,y,z,g,npx,npy,npz)];
              }
            }
            fill_moments(feq, &rho, &ux, &uy, &uz);
            fill_feq(rho, ux, uy, uz, feq);
            for (size_t g = 0; g < NG; g++) {
              for (size_t m = 0; m < NM; m++) {
                b[idx(m,x,y,z,g,npx,npy,npz)] = (1. - BETA2) * b[idx(m,x,y,z,g,npx,npy,npz)]
                                                + BETA2 * feq[m + NM * g];
              }
            }
          }
        }
      }
    });
  };

  // advect
  auto advect_blocks_d3q27 = [&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<3>(threads, wg), [=](sycl::nd_item<3> it) {
      const size_t bx = it.get_group(0);
      const size_t by = it.get_group(1);
      const size_t bz = it.get_group(2);
      const size_t bidx  = (bx + nbx * (by + nby * bz));
      VARTYPE* b = &T[bsize * bidx]; 
      VARTYPE* bnew = &Tnew[bsize * bidx];

      const int x = it.get_local_id(0);
      const int y = it.get_local_id(1);
      const int z = it.get_local_id(2);

      if (z >= np and z <= npz-(np+1)) {
        if (y >= np and y <= npy-(np+1)) {
          if (x >= np and x <= npx-(np+1)) {
            bnew[idx(0,x,y,z,0,npx,npy,npz)] = b[idx(0,x-1,y-1,z+1,0,npx,npy,npz)];
            bnew[idx(1,x,y,z,0,npx,npy,npz)] = b[idx(1,x+0,y-1,z+1,0,npx,npy,npz)];
            bnew[idx(2,x,y,z,0,npx,npy,npz)] = b[idx(2,x+1,y-1,z+1,0,npx,npy,npz)];
            bnew[idx(3,x,y,z,0,npx,npy,npz)] = b[idx(3,x-1,y+0,z+1,0,npx,npy,npz)];
            bnew[idx(4,x,y,z,0,npx,npy,npz)] = b[idx(4,x+0,y+0,z+1,0,npx,npy,npz)];
            bnew[idx(5,x,y,z,0,npx,npy,npz)] = b[idx(5,x+1,y+0,z+1,0,npx,npy,npz)];
            bnew[idx(6,x,y,z,0,npx,npy,npz)] = b[idx(6,x-1,y+1,z+1,0,npx,npy,npz)];

            bnew[idx(0,x,y,z,1,npx,npy,npz)] = b[idx(0,x+1,y+1,z-1,1,npx,npy,npz)];
            bnew[idx(1,x,y,z,1,npx,npy,npz)] = b[idx(1,x+0,y+1,z-1,1,npx,npy,npz)];
            bnew[idx(2,x,y,z,1,npx,npy,npz)] = b[idx(2,x-1,y+1,z-1,1,npx,npy,npz)];
            bnew[idx(3,x,y,z,1,npx,npy,npz)] = b[idx(3,x+1,y+0,z-1,1,npx,npy,npz)];
            bnew[idx(4,x,y,z,1,npx,npy,npz)] = b[idx(4,x+0,y+0,z-1,1,npx,npy,npz)];
            bnew[idx(5,x,y,z,1,npx,npy,npz)] = b[idx(5,x-1,y+0,z-1,1,npx,npy,npz)];
            bnew[idx(6,x,y,z,1,npx,npy,npz)] = b[idx(6,x+1,y-1,z-1,1,npx,npy,npz)];
    
            bnew[idx(0,x,y,z,2,npx,npy,npz)] = b[idx(0,x+0,y+1,z+1,2,npx,npy,npz)];
            bnew[idx(1,x,y,z,2,npx,npy,npz)] = b[idx(1,x+1,y+1,z+1,2,npx,npy,npz)];
            bnew[idx(2,x,y,z,2,npx,npy,npz)] = b[idx(2,x+0,y+0,z+0,2,npx,npy,npz)];
            bnew[idx(3,x,y,z,2,npx,npy,npz)] = b[idx(3,x+1,y+0,z+0,2,npx,npy,npz)];
            bnew[idx(4,x,y,z,2,npx,npy,npz)] = b[idx(4,x-1,y+1,z+0,2,npx,npy,npz)];
            bnew[idx(5,x,y,z,2,npx,npy,npz)] = b[idx(5,x+0,y+1,z+0,2,npx,npy,npz)];
            bnew[idx(6,x,y,z,2,npx,npy,npz)] = b[idx(6,x+1,y+1,z+0,2,npx,npy,npz)];
    
            bnew[idx(0,x,y,z,3,npx,npy,npz)] = b[idx(0,x+0,y-1,z-1,3,npx,npy,npz)];
            bnew[idx(1,x,y,z,3,npx,npy,npz)] = b[idx(1,x-1,y-1,z-1,3,npx,npy,npz)];
            bnew[idx(2,x,y,z,3,npx,npy,npz)] = b[idx(2,x+0,y+0,z+0,3,npx,npy,npz)];
            bnew[idx(3,x,y,z,3,npx,npy,npz)] = b[idx(3,x-1,y+0,z+0,3,npx,npy,npz)];
            bnew[idx(4,x,y,z,3,npx,npy,npz)] = b[idx(4,x+1,y-1,z+0,3,npx,npy,npz)];
            bnew[idx(5,x,y,z,3,npx,npy,npz)] = b[idx(5,x+0,y-1,z+0,3,npx,npy,npz)];
            bnew[idx(6,x,y,z,3,npx,npy,npz)] = b[idx(6,x-1,y-1,z+0,3,npx,npy,npz)];
          }
        }
      }
    });
  };

  double collide_time = 0, advect_time = 0;
  try {
    for (size_t t = 0; t < nt; t++) {
      MPI_Barrier(mpi_comm);
      q.wait();
      double tic1 = MPI_Wtime();
      q.submit(collide_blocks_d3q27);
      q.wait();
      MPI_Barrier(mpi_comm);
      q.wait();
      double tic2 = MPI_Wtime();
      q.submit(advect_blocks_d3q27);
      q.wait();
      std::swap(T, Tnew);
      MPI_Barrier(mpi_comm);
      double tic3 = MPI_Wtime();
      collide_time += tic2 - tic1;
      advect_time += tic3 - tic2;
    }
  } catch (sycl::exception const& ex) {
    std::cerr << "dpcpp error: " << ex.what() << std::endl;
  }

  VARTYPE sample_val;
  q.memcpy(&sample_val, &T[np + npx * (np + npy * np)], sizeof(VARTYPE));
  VARTYPE feq[NM * NG];
  for (size_t g = 0; g < NG; g++) {
    for (size_t m = 0; m < NM; m++) {
      q.memcpy(&feq[m + NM * g], &T[idx(m,np,np,np,g,npx,npy,npz)], sizeof(VARTYPE));
    }
  }
  VARTYPE rho, ux, uy, uz;
  fill_moments(feq, &rho, &ux, &uy, &uz);

  sycl::free(T, q);
  sycl::free(Tnew, q);

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

  sycl::default_selector d_selector;
  sycl::device d = sycl::device(d_selector);

  const size_t nx  = atoi(argv[1]);
  const size_t ny  = atoi(argv[2]);
  const size_t nz  = atoi(argv[3]);
  const size_t nbx = atoi(argv[4]);
  const size_t nby = atoi(argv[5]);
  const size_t nbz = atoi(argv[6]);
  const size_t nt  = atoi(argv[7]);

  auto res = run(nx, ny, nz, nbx, nby, nbz, nt, mpi_comm, d);
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
