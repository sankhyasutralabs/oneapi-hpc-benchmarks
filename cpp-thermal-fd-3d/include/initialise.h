#pragma once

#include "config.h"
#include <stdlib.h>

// initialize padding points to T = Tbc
// and domain points to T = Tbulk, the
// faces of the domain will act as Dirichlet
// boundary condition fixed at T = Tbc
void initialise(REAL* T, size_t npx, size_t npy, size_t npz, size_t np, double Tbulk, double Tbc);
