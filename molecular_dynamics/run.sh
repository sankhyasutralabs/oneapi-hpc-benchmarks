#!/bin/bash
# compile
icpc -qopenmp -mcmodel large -shared-intel -O3 -xCORE-AVX512 -qopt-zmm-usage=high -qopt-streaming-stores always -o n_body_omp nbody_omp_all_interactions.cpp

# Run
export OMP_DISPLAY_ENV=true
export OMP_DISPLAY_AFFINITY=true
export OMP_SCHEDULE=static
export OMP_DYNAMIC=false
#export OMP_STACKSIZE=30M        #L3 cache per process is 60M
export OMP_STACKSIZE=27M        #L3 cache per process is 54M
export OMP_MAX_ACTIVE_LEVELS=1  #OMP_NESTED=FALSE is deprecated
#export OMP_NUM_THREADS=80
#export OMP_THREAD_LIMIT=160
export OMP_NUM_THREADS=72
export OMP_THREAD_LIMIT=144
export OMP_PROC_BIND=true
#export GOMP_CPU_AFFINITY=0-79:1
export GOMP_CPU_AFFINITY=0-71:1

./n_body_omp 576000 1