#!/bin/bash

source /opt/intel/oneapi/setvars.sh 
#source /opt/intel/oneAPI/latest/setvars.sh 

rm -rf bin
mkdir -p bin

CXX=mpiicpc
CXXFLAGS="-std=c++11 -Wall -O3 -mcmodel=large -xCORE-AVX512 -qopt-zmm-usage=high -qopt-streaming-stores always"
$CXX $CXXFLAGS flow_mpi.cpp -o bin/flow_mpi
