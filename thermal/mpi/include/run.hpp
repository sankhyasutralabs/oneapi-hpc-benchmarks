#pragma once

#include "config.hpp"
#include <mpi.h>
#include <cstdlib>
#include <utility>

std::pair<double, VARTYPE>
run(size_t nx, size_t ny, size_t nz, size_t nbx, size_t nby, size_t nbz, size_t nt, MPI_Comm mpi_comm);
