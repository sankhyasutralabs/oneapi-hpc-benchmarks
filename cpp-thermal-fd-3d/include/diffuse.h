#pragma once

#include "config.h"
#include <stdlib.h>

void diffuse(REAL* T, REAL* Tnew, size_t npx, size_t npy, size_t npz, size_t np, double cfl);
