#include <iostream>
#include <chrono>

#ifndef REAL
  #define REAL double
#endif

#ifndef NT
  #define NT 10
#endif

#ifndef NX
  #define NX 100
#endif

#define NY NX
#define NZ NX

inline size_t
idx(const size_t x, const size_t y, const size_t z)
{
  return x + (NX + 2) * (y + (NY + 2) * z);
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
main()
{
  const REAL cfl = 0.1;

  REAL* T    = (REAL*)malloc((NX + 2) * (NY + 2) * (NZ + 2) * sizeof(REAL));
  REAL* Tnew = (REAL*)malloc((NX + 2) * (NY + 2) * (NZ + 2) * sizeof(REAL));

  // initialize padding points to T = 100
  // and domain points to T = 0, the faces
  // of the domain will act as Dirichlet
  // boundary condition fixed at T = 100
  for (size_t z = 0; z <= NZ+1; z++) {
    for (size_t y = 0; y <= NY+1; y++) {
      for (size_t x = 0; x <= NX+1; x++) {
        T[idx(x,y,z)] = 100.0;
        Tnew[idx(x,y,z)] = 100.0;
      }
    }
  }
  for (size_t z = 1; z <= NZ; z++) {
    for (size_t y = 1; y <= NY; y++) {
      for (size_t x = 1; x <= NX; x++) {
        T[idx(x,y,z)] = 0.0;
        Tnew[idx(x,y,z)] = 0.0;
      }
    }
  }

  auto tic = std::chrono::high_resolution_clock::now();
  // updates
  for (size_t t = 0; t < NT; t++) {
    for (size_t z = 1; z <= NZ; z++) {
      for (size_t y = 1; y <= NY; y++) {
        for (size_t x = 1; x <= NX; x++) {
          Tnew[idx(x,y,z)] = T[idx(x,y,z)] + cfl * (T[idx(x-1,y,z)] + T[idx(x+1,y,z)]
                                                  + T[idx(x,y-1,z)] + T[idx(x,y+1,z)]
                                                  + T[idx(x,y,z-1)] + T[idx(x,y,z+1)]
                                                  - 6. * T[idx(x,y,z)]);
        }
      }
    }
    swap(&T, &Tnew);
  }
  auto toc = std::chrono::high_resolution_clock::now();
  auto elapsed = toc - tic;
  auto duration = std::chrono::duration<double, std::nano>(elapsed).count() * 1e-9;

  std::cout << NX << ", " << duration << ", " << T[idx(1,1,1)] << std::endl;

  free(T);
  free(Tnew);
  return 0;
}
