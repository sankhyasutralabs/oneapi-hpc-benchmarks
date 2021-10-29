#!/bin/bash

source /opt/intel/oneapi/setvars.sh 
#source /opt/intel/oneAPI/latest/setvars.sh 

rm -rf bin
mkdir -p bin

CXX=mpiicpc
CXXFLAGS="-qopenmp -std=c++17 -Wall -O3 -mcmodel=large -xCORE-AVX512 -qopt-zmm-usage=high -qopt-streaming-stores always -DVARTYPE=double"
$CXX $CXXFLAGS thermal_omp_comm.cpp -o bin/thermal_omp

CXX=mpiicpc
CXXFLAGS="-fsycl -std=c++17 -Wall -O3 -mcmodel=large -xCORE-AVX512 -qopt-zmm-usage=high -fno-sycl-id-queries-fit-in-int -DVARTYPE=double" # -qopt-streaming-stores always
I_MPI_CXX=dpcpp $CXX $CXXFLAGS thermal_dpcpp_comm.cpp -o bin/thermal_dpcpp_comm
