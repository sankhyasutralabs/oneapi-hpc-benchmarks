# Flow Solver Benchmarks

Tested on:

- 2S Intel Xeon 8360Y Ice Lake
- 16 x 16 GB DDR4 3200 MT/s
- Theoretical Memory Bandwidth = 410 GB/s

## C++ with MPI

On each MPI process, simulate a grid of `(nbx, nby, nbz)` blocks
with each block containing `(nx,ny,nz)` grid points and with
an additional layer of 1 padding point on each face of blocks.
Each point has `NM x NG = 8 x 4 = 32` variables corresponding to
the f-distribution variables of a D3Q27 lattice Boltzmann model.
Each simulation runs `nt` iterations of the collide and advect kernel.

```
mpirun -n 72 --bind-to core:1 ./bin/flow_mpi nx ny nz nbx nby nbz nt
```

### Case: One large block

```
cd flow
bash compile.sh

mpirun -n 72 --bind-to core:1 ./bin/flow_mpi 256 256 8 1 1 1 10
```

#### Result

- Data Traffic = `32 x (256 x 256 x 8) points per block x (1 x 1 x 1) blocks x 8 bytes x 72 processes` = **9.66 GB**
- collide took walltime of 0.66 seconds for 10 iterations = **0.066 s** per iteration
- advect took walltime of 0.96 seconds for 10 iterations = **0.096 s** per iteration
- assuming `m` memory operations, effective bandwidth used = `data size x m / time per iteration`
- STREAM Triad bandwidth = 320 GB/s
- effective number of operations used in **collide** `m = 320 / (9.66 / 0.066)` = **2.18 ops**
- effective number of operations used in **advect** `m = 320 / (9.66 / 0.096)` = **3.19 ops**


### Case: Few medium blocks

```
cd flow
bash compile.sh

mpirun -n 72 --bind-to core:1 ./bin/flow_mpi 32 32 32 4 4 1 10
```

#### Result

- Data Traffic = `32 x (32 x 32 x 32) points per block x (4 x 4 x 1) blocks x 8 bytes x 72 processes` = **9.66 GB**
- collide took walltime of 0.68 seconds for 10 iterations = **0.068 s** per iteration
- advect took walltime of 0.71 seconds for 10 iterations = **0.071 s** per iteration
- assuming `m` memory operations, effective bandwidth used = `data size x m / time per iteration`
- STREAM Triad bandwidth = 320 GB/s
- effective number of operations used in **collide** `m = 320 / (9.66 / 0.068)` = **2.25 ops**
- effective number of operations used in **advect** `m = 320 / (9.66 / 0.071)` = **2.34 ops**


### Case: Many small blocks

```
cd flow
bash compile.sh

mpirun -n 72 --bind-to core:1 ./bin/flow_mpi 8 8 8 32 32 1 10
```

#### Result

- Data Traffic = `32 x (8 x 8 x 8) points per block x (32 x 32 x 1) blocks x 8 bytes x 72 processes` = **9.66 GB**
- collide took walltime of 0.81 seconds for 10 iterations = **0.081 s** per iteration
- advect took walltime of 0.87 seconds for 10 iterations = **0.087 s** per iteration
- assuming `m` memory operations, effective bandwidth used = `data size x m / time per iteration`
- STREAM Triad bandwidth = 320 GB/s
- effective number of operations used in **collide** `m = 320 / (9.66 / 0.081)` = **2.70 ops**
- effective number of operations used in **advect** `m = 320 / (9.66 / 0.087)` = **2.90 ops**

## DPC++

Double memory advection and pointer swap.

```
cd flow
bash compile.sh

./bin/flow_dpcpp 8 8 8 32 32 72 10
```

#### Result

- Data Traffic = `32 x (8 x 8 x 8) points per block x (32 x 32 x 72) blocks x 8 bytes` = **9.66 GB**
- collide took walltime of 1.04 seconds for 10 iterations = **0.104 s** per iteration
- advect took walltime of 1.82 seconds for 10 iterations = **0.182 s** per iteration
- assuming `m` memory operations, effective bandwidth used = `data size x m / time per iteration`
- STREAM Triad bandwidth = 320 GB/s
- effective number of operations used in **collide** `m = 320 / (9.66 / 0.104)` = **3.44 ops**
- effective number of operations used in **advect** `m = 320 / (9.66 / 0.182)` = **6.02 ops**
