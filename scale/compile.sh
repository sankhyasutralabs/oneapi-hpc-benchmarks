#!/bin/bash

source /opt/intel/oneapi/setvars.sh 
#source /opt/intel/oneAPI/latest/setvars.sh 

rm -rf bin
mkdir -p bin

# Compile

CXXFLAGS="-fsycl -std=c++17 -Wall -O3 -mcmodel=large -xCORE-AVX512 -qopt-zmm-usage=high -fno-sycl-id-queries-fit-in-int"
dpcpp $CXXFLAGS scale_dpcpp.cpp -o bin/scale_dpcpp