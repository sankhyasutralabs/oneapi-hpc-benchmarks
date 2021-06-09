#!/bin/bash

#CXX=mpic++
#CXXFLAGS="-std=c++11 -Wall -O3 -mcmodel=large -march=skylake-avx512"
  
source /opt/intel/oneapi/setvars.sh 
#source /opt/intel/oneAPI/latest/setvars.sh 
#module load intel-icc/18.0
CXX=mpiicpc
CXXFLAGS="-std=c++11 -Wall -O3 -mcmodel=large -xCORE-AVX512 -qopt-zmm-usage=high -qopt-streaming-stores always"

rm -rf obj
rm -rf lib
rm -rf bin

mkdir -p obj
mkdir -p lib
mkdir -p bin

# avx512 vectorized mpi thermal
$CXX $CXXFLAGS -I include -c src/initialise.cpp -o obj/initialise.o
$CXX $CXXFLAGS -I include -c src/diffuse.cpp -o obj/diffuse.o
$CXX $CXXFLAGS -I include -c src/run.cpp -o obj/run.o
ar rcs lib/libthermal.a obj/initialise.o obj/diffuse.o obj/run.o
$CXX $CXXFLAGS -I include src/thermal_mpi.cpp -o bin/thermal_mpi -L lib -lthermal
