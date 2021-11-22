#!/bin/bash

dpcpp -fsycl -std=c++17 -Wall -O3 -mcmodel=large -xCORE-AVX512 -qopt-zmm-usage=high -fno-sycl-id-queries-fit-in-int nbody_dpcpp_all_interactions.cpp -o n_body_dpcpp

DPCPP_CPU_PLACES=numa_domains ./n_body_dpcpp 576000 1