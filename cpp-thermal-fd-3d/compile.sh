#!/bin/bash

source /opt/intel/compiler/latest/bin/compilervars.sh intel64
source /opt/intel/impi/latest/bin/compilervars.sh intel64

rm -rf obj
rm -rf lib
rm -rf bin

mkdir -p obj
mkdir -p lib
mkdir -p bin

# avx512 vectorized mpi thermal
mpiicc -std=c99 -Wall -O3 -mcmodel=large -xCORE-AVX512 -qopt-zmm-usage=high -qopt-streaming-stores always -I include -c src/initialise.c -o obj/initialise.o
mpiicc -std=c99 -Wall -O3 -mcmodel=large -xCORE-AVX512 -qopt-zmm-usage=high -qopt-streaming-stores always -I include -c src/diffuse.c -o obj/diffuse.o
ar rcs lib/libthermal.a obj/initialise.o obj/diffuse.o
mpiicc -std=c99 -Wall -O3 -mcmodel=large -xCORE-AVX512 -qopt-zmm-usage=high -qopt-streaming-stores always -I include src/thermal.c -o bin/thermal -L lib -lthermal

# serial thermal
icpc -std=c++11 -Wall -O3 -mcmodel=large -xCORE-AVX512 -qopt-zmm-usage=high -qopt-streaming-stores always src/thermal_serial.cpp -o bin/thermal_serial -DNX=1000 -DNT=20
