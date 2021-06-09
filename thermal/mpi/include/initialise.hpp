#pragma once

#include "config.hpp"
#include <cstdlib>

// initialize padding points to T = Tbc
// and domain points to T = Tbulk, the
// faces of the domain will act as Dirichlet
// boundary condition fixed at T = Tbc
void initialise(VARTYPE* T, size_t npx, size_t npy, size_t npz, size_t np, VARTYPE Tbulk, VARTYPE Tbc);
void initialise_blocks(VARTYPE* T, size_t npx, size_t npy, size_t npz, size_t np, size_t nbx, size_t nby, size_t nbz, VARTYPE Tbulk, VARTYPE Tbc);
