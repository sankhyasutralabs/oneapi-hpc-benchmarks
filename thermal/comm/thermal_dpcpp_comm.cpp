#include <mpi.h>
#include <CL/sycl.hpp>
#include <dpc_common.hpp>
#include <utility>
#include <cstdlib>
#include <iostream>
#include <chrono>

#ifndef VARTYPE
  #define VARTYPE double
#endif

inline size_t
idx(size_t x, size_t y, size_t z, size_t npx, size_t npy, size_t npz)
{
  return x + npx * (y + npy * z);
}

std::pair<float, VARTYPE>
run(size_t nx, size_t ny, size_t nz, size_t nbx, size_t nby, size_t nbz, size_t nt, MPI_Comm mpi_comm, sycl::device& d)
{
  const size_t np = 1;
  const size_t npx = nx + 2 * np;
  const size_t npy = ny + 2 * np;
  const size_t npz = nz + 2 * np;

  sycl::property_list properties{ sycl::property::queue::in_order() };
  sycl::queue q(d, dpc_common::exception_handler, properties);

  const size_t alloc_bytes = (npx * npy * npz) * (nbx * nby * nbz) * sizeof(VARTYPE);
  VARTYPE* T = (VARTYPE*)sycl::malloc_device(alloc_bytes, q);
  VARTYPE* Tnew = (VARTYPE*)sycl::malloc_device(alloc_bytes, q);

  sycl::range<3> threads = { npz * nbz, npy * nby, npx * nbx };
  sycl::range<3> wg = { npz, npy, npx };
  const size_t bsize = (npx * npy * npz);

  // initialise
  auto initialise = [=](sycl::nd_item<3> it) {
    const size_t bx = it.get_group(2);
    const size_t by = it.get_group(1);
    const size_t bz = it.get_group(0);
    const size_t bidx  = (bx + nbx * (by + nby * bz));
    VARTYPE* b = &T[bsize * bidx]; 
    VARTYPE* bnew = &Tnew[bsize * bidx];

    const int x = it.get_local_id(2);
    const int y = it.get_local_id(1);
    const int z = it.get_local_id(0);

    const VARTYPE Tbc = 100.;
    const VARTYPE Tbulk = 10.;

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

    // if (z >= np and z <= npz-(np+1)) {
    //   if (y >= np and y <= npy-(np+1)) {
    //     if (x >= np and x <= npx-(np+1)) {
    //       b[idx(x,y,z,npx,npy,npz)] = Tbulk;
    //       bnew[idx(x,y,z,npx,npy,npz)] = Tbulk;
    //     }
    //   }
    // }
  };

  auto initialise_blocks = [&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<3>(threads, wg), initialise);
  };

  try {
    q.submit(initialise_blocks);
    q.wait();
  } catch (sycl::exception const& ex) {
    std::cerr << "dpcpp error: " << ex.what() << std::endl;
  }

  auto diffuse_blocks_d3q7 = [&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<3>(threads, wg), [=](sycl::nd_item<3> it) {
    const size_t bx = it.get_group(2);
    const size_t by = it.get_group(1);
    const size_t bz = it.get_group(0);
    const size_t bidx  = (bx + nbx * (by + nby * bz));
    VARTYPE* b = &T[bsize * bidx]; 
    VARTYPE* bnew = &Tnew[bsize * bidx];

    const int x = it.get_local_id(2);
    const int y = it.get_local_id(1);
    const int z = it.get_local_id(0);

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
    if (0 == x and bx != 0) {
      b[idx(x,y,z,npx,npy,npz)] = b_neg_ngb[idx(npx-(np+1),y,z,npx,npy,npz)];
    }

    b_ngb_id = (plus_ngb_b + nbx * (by + nby * bz));
    VARTYPE* b_plus_ngb = &T[bsize * b_ngb_id];
    if (npx-1 == x and bx != (nbx -1)) {
      b[idx(x,y,z,npx,npy,npz)] = b_plus_ngb[idx(np,y,z,npx,npy,npz)];
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
    if (0 == y and by != 0) {
      b[idx(x,y,z,npx,npy,npz)] = b_neg_ngb[idx(x,npy-(np+1),z,npx,npy,npz)];
    }

    b_ngb_id = (bx + nbx * (plus_ngb_b + nby * bz));
    b_plus_ngb = &T[bsize * b_ngb_id];
    if (npy-1 == y and by != (nby -1)) {
      b[idx(x,y,z,npx,npy,npz)] = b_plus_ngb[idx(x,np,z,npx,npy,npz)];
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
    if (0 == z and bz != 0) {
      b[idx(x,y,z,npx,npy,npz)] = b_neg_ngb[idx(x,y,npz-(np+1),npx,npy,npz)];
    }

    b_ngb_id = (bx + nbx * (by + nby * plus_ngb_b));
    b_plus_ngb = &T[bsize * b_ngb_id];
    if (npz-1 == z and bz != (nbz -1)) {
      b[idx(x,y,z,npx,npy,npz)] = b_plus_ngb[idx(x,y,np,npx,npy,npz)];
    }

      if (z >= np and z <= npz-(np+1)) {
        if (y >= np and y <= npy-(np+1)) {
          if (x >= np and x <= npx-(np+1)) {
            const VARTYPE cfl = 0.1;
            bnew[idx(x,y,z,npx,npy,npz)] = (1. - 6.0 * cfl) * b[idx(x,y,z,npx,npy,npz)]
                                         + cfl * (b[idx(x-1,y,z,npx,npy,npz)] + b[idx(x+1,y,z,npx,npy,npz)]
                                                + b[idx(x,y-1,z,npx,npy,npz)] + b[idx(x,y+1,z,npx,npy,npz)]
                                                + b[idx(x,y,z-1,npx,npy,npz)] + b[idx(x,y,z+1,npx,npy,npz)]);
          }
        }
      }
    });
  };
  MPI_Barrier(mpi_comm);

  double total_time = 0.0;
  auto start_time = std::chrono::high_resolution_clock::now();

  try {
    for (size_t t = 0; t < nt; t++) {
      q.submit(diffuse_blocks_d3q7);
      q.wait();
      MPI_Barrier(mpi_comm);

      std::swap(T, Tnew);
    }
  } catch (sycl::exception const& ex) {
    std::cerr << "dpcpp error: " << ex.what() << std::endl;
  }
  q.wait();
  MPI_Barrier(mpi_comm);
  total_time += std::chrono::duration<double, std::nano>(
                 std::chrono::high_resolution_clock::now() - start_time).count();
  total_time *= 1E-9; //nano to seconds

  VARTYPE sample_val;
  q.memcpy(&sample_val, &T[np + npx * (np + npy * np)], sizeof(VARTYPE));

  sycl::free(T, q);
  sycl::free(Tnew, q);

  return std::make_pair(total_time, sample_val);
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
    std::cout << res.first << ", ";
    std::cout << res.second;
    std::cout << std::endl;
  }

  MPI_Finalize();
  return 0;
}