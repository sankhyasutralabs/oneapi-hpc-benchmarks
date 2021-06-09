# Benchmarks

Tested on:

- 2S Intel Xeon 8360Y Ice Lake
- 16 x 16 GB DDR4 3200 MT/s
- Theoretical Memory Bandwidth = 410 GB/s

# Thermal Solver

## C++ with MPI

On each MPI process, simulate a grid of `(nbx, nby, nbz)` blocks
with each block containing `(nx,ny,nz)` grid points and with
an additional layer of 1 padding point on each face of blocks.
Each simulation runs `nt` iterations of the diffuse kernel.

```
mpirun -n 72 --bind-to core:1 ./bin/thermal_mpi nx ny nz nbx nby nbz nt
```

### Case: One large block

```
cd thermal/mpi
bash scripts/compile.sh

mpirun -n 72 --bind-to core:1 ./bin/thermal_mpi 256 256 256 1 1 1 10
```

#### Result

- Data Traffic = `(256 x 256 x 256) points per block x (1 x 1 x 1) blocks x 8 bytes x 72 processes` = **9.66 GB**
- took walltime of 1.02 seconds for 10 iterations = **0.102 s** per iteration
- assuming `m` memory operations, effective bandwidth used = `9.66 m / 0.157 = 94.7 m GB/s`
- STREAM Triad bandwidth = 320 GB/s
- effective number of operations `m = 320 / 94.7` = **3.38 ops**


### Case: Many small blocks

```
cd thermal/mpi
bash scripts/compile.sh

mpirun -n 72 --bind-to core:1 ./bin/thermal_mpi 8 8 8 32 32 32 10
```

#### Result

- Data Traffic = `(8 x 8 x 8) points per block x (32 x 32 x 32) blocks x 8 bytes x 72 processes` = **9.66 GB**
- took walltime of 1.57 seconds for 10 iterations = **0.157 s** per iteration
- assuming `m` memory operations, effective bandwidth used = `9.66 m / 0.157 = 61.5 m GB/s`
- STREAM Triad bandwidth = 320 GB/s
- effective number of operations `m = 320 / 61.5` = **5.2 ops**

## DPC++

A grid with `(nbx,nby,nbz)` blocks, each block containing
`(nx,ny,nz)` grid points, is distributed across all threads.

```
cd thermal/dpcpp
bash scripts/compile.sh

./bin/thermal_dpcpp 8 8 8 135 135 135 10
```

### Result

- Data Traffic = `(8 x 8 x 8) points per block x (135 x 135 x 135) blocks x 8 bytes` = **10.07 GB**
- took walltime of 2.68 seconds for 10 iterations = **0.268 s** per iteration
- assuming `m` memory operations, effective bandwidth used = `10.07 m / 0.268 = 37.5 m GB/s`
- STREAM Triad bandwidth = 320 GB/s
- effective number of operations `m = 320 / 37.5` = **8.533 ops**
