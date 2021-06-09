# Benchmarks

Tested on:

- Skylake
  - 2S Intel Xeon 6148
  - 12 x 16 GB DDR4 2666 MT/s
  - Theoretical Memory Bandwidth = 256 GB/s
- Ice Lake
  - 2S Intel Xeon 8360Y
  - 16 x 16 GB DDR4 3200 MT/s
  - Theoretical Memory Bandwidth = 410 GB/s

## Vectorized MPI Code

- Using `src/thermal.c`
- compiled using icc (ICC) 19.1.3.304 20200925
- run using MPICH version 3.3.2
- `nt = 10` steps
- `nx * ny * nz ~ 64 million`

### Compile and Run

```
bash scripts/compile.sh

# on skylake with all cores
mpirun -n 40 -bind-to core:1 ./bin/thermal <nx> <ny> <nz> 10

# on icelake with all cores
mpirun -n 72 -bind-to core:1 ./bin/thermal <nx> <ny> <nz> 10
```

### Walltimes (seconds)

|nx|  ny|    nz| Skylake|  Ice Lake|
|---:|---:|---:|-------:|--------:|
| 40|  40| 40000| 3.31| 3.76|
| 80|  80| 10000| 3.20| 3.50|
| 80| 100|  8000| 3.20| 3.49|
|160| 100|  4000| 3.15| 3.31|
|160| 200|  2000| 3.26| 3.29|
|200| 100|  3200| 3.13| 3.25|
|200| 200|  1600| 3.38| 3.31|
|200| 400|   800| 4.30| 3.92|
|240| 111|  2400| 3.15| 3.21|
|320| 400|   500| 4.89| 4.83|
|320| 200|  1000| 3.88| 3.36|
|400| 400|   400| 4.97| 4.93|

### Comparision

#### Best Case for Skylake

- `nx=200, ny=100, nz=3200`
- takes `64 x 8 bytes = 512 MB` per core
- `512 MB x 40 cores = 20.48 GB` data
- took `3.13 s / 10 steps = 0.313 s` per iteration
- assuming `m` memory operations, effective bandwidth used = `20.48 m / 0.313` = 65.43 m GB/s`
- theoretical bandwidth = 256 GB/s
- effective number of operations `m = 256 / 65.43 = 3.9 ops`

#### Best Case for Ice Lake

- `nx=240, ny=111, nz=2400`
- takes `63.96 x 8 bytes = 511.5 MB` per core
- `511.5 MB x 72 cores = 36.8 GB` data
- took `3.21 s / 10 steps = 0.321 s` per iteration
- assuming `m` memory operations, effective bandwidth used = `36.8 m / 0.321` = 114.6 m GB/s`
- theoretical bandwidth = 410 GB/s
- effective number of operations `m = 410 / 114.6 = 3.5 ops`

## Serial Code

- Using `src/thermal_serial.cpp`
- ICC18 : icpc (ICC) 18.0.3 20180410
- ICC19 : icpc (ICC) 19.1.3.304 20200925

### Compile and Run:

```
bash scripts/compile.sh

# on both skylake and icelake
./bin/thermal_serial
```

Walltime (seconds):


| NX   | Skylake + ICC18| Skylake + ICC19| Ice Lake + ICC18| Ice Lake + ICC19|
| ---: | -----: | ------:| -----: | -----: |
|  100 | 0.038  | 0.035  | 0.038  | 0.043  |
|  200 | 0.311  | 0.296  | 0.330  | 0.212  |
|  400 | 2.445  | 2.369  | 2.571  | 2.923  |
|  800 | 22.755 | 19.737 | 21.075 | 23.373 |
| 1000 | 50.140 | 44.030 | 42.650 | 45.773 |
| 1200 | 86.844 | 78.315 | 74.550 | 79.091 |
| 1400 | 143.856| 129.457| 120.111| 126.451|
| 1600 | 222.504| 196.551| 179.432| 188.568|
