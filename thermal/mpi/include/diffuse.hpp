#pragma once

#include "config.hpp"
#include <cstdlib>

void diffuse_d3q7(VARTYPE* T, VARTYPE* Tnew, size_t npx, size_t npy, size_t npz, size_t np, VARTYPE cfl);
void diffuse_blocks_d3q7(VARTYPE* T, VARTYPE* Tnew, size_t npx, size_t npy, size_t npz, size_t np, size_t nbx, size_t nby, size_t nbz, VARTYPE cfl);
void diffuse_d3q7_avx512(VARTYPE* T, VARTYPE* Tnew, size_t npx, size_t npy, size_t npz, size_t np, VARTYPE cfl);
void diffuse_blocks_d3q7_avx512(VARTYPE* T, VARTYPE* Tnew, size_t npx, size_t npy, size_t npz, size_t np, size_t nbx, size_t nby, size_t nbz, VARTYPE cfl);
