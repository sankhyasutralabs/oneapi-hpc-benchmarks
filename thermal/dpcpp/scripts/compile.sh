#!/bin/bash

#source /opt/intel/oneAPI/latest/setvars.sh 
source /opt/intel/oneapi/setvars.sh 
CXX=mpiicpc
CXXFLAGS="-fsycl -std=c++11 -Wall -O3 -mcmodel=large -xCORE-AVX512 -qopt-zmm-usage=high -fno-sycl-id-queries-fit-in-int" # -qopt-streaming-stores always

rm -rf obj
rm -rf lib
rm -rf bin

mkdir -p obj
mkdir -p lib
mkdir -p bin

I_MPI_CXX=dpcpp $CXX $CXXFLAGS thermal_dpcpp.cpp -o bin/thermal_dpcpp
