#!/bin/bash

source /opt/intel/oneapi/setvars.sh 
#source /opt/intel/oneAPI/latest/setvars.sh 

# Run

SYCL_DEVICE_FILTER=opencl:cpu DPCPP_CPU_PLACES=numa_domains ./bin/scale_dpcpp