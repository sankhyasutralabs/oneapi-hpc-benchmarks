#!/bin/bash

source /opt/intel/oneapi/setvars.sh 
#source /opt/intel/oneAPI/latest/setvars.sh 

rm -rf bin
mkdir -p bin

CXX=mpiicpc
CXXFLAGS="-std=c++11 -Wall -O3 -mcmodel=large -xCORE-AVX512 -qopt-zmm-usage=high -qopt-streaming-stores always"
$CXX $CXXFLAGS thermal_mpi.cpp -o bin/thermal_mpi

CXX=mpiicpc
CXXFLAGS="-fsycl -std=c++11 -Wall -O3 -mcmodel=large -xCORE-AVX512 -qopt-zmm-usage=high -fno-sycl-id-queries-fit-in-int" # -qopt-streaming-stores always
I_MPI_CXX=dpcpp $CXX $CXXFLAGS thermal_dpcpp.cpp -o bin/thermal_dpcpp
