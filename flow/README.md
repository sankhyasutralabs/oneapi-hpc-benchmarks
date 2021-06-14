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
bash scripts/compile.sh

mpirun -n 72 --bind-to core:1 ./bin/flow_mpi 256 256 8 1 1 1 10
```

#### Result

- Data Traffic = `32 x (256 x 256 x 8) points per block x (1 x 1 x 1) blocks x 8 bytes x 72 processes` = **9.66 GB**
- collide took walltime of 3.00 seconds for 10 iterations = **0.300 s** per iteration
- advect took walltime of 0.96 seconds for 10 iterations = **0.096 s** per iteration
- assuming `m` memory operations, effective bandwidth used = `data size x m / time per iteration`
- STREAM Triad bandwidth = 320 GB/s
- effective number of operations used in **collide** `m = 320 / (9.66 / 0.300)` = **9.93 ops**
- effective number of operations used in **advect** `m = 320 / (9.66 / 0.096)` = **3.18 ops**


### Case: Few medium blocks

```
cd flow
bash scripts/compile.sh

mpirun -n 72 --bind-to core:1 ./bin/flow_mpi 32 32 32 4 4 1 10
```

#### Result

- Data Traffic = `32 x (32 x 32 x 32) points per block x (4 x 4 x 1) blocks x 8 bytes x 72 processes` = **9.66 GB**
- collide took walltime of 3.33 seconds for 10 iterations = **0.333 s** per iteration
- advect took walltime of 0.69 seconds for 10 iterations = **0.069 s** per iteration
- assuming `m` memory operations, effective bandwidth used = `data size x m / time per iteration`
- STREAM Triad bandwidth = 320 GB/s
- effective number of operations used in **collide** `m = 320 / (9.66 / 0.333)` = **11.03 ops**
- effective number of operations used in **advect** `m = 320 / (9.66 / 0.069)` = **2.28 ops**


### Case: Many small blocks

```
cd flow
bash scripts/compile.sh

mpirun -n 72 --bind-to core:1 ./bin/flow_mpi 8 8 8 32 32 1 10
```

#### Result

- Data Traffic = `32 x (8 x 8 x 8) points per block x (32 x 32 x 1) blocks x 8 bytes x 72 processes` = **9.66 GB**
- collide took walltime of 3.31 seconds for 10 iterations = **0.331 s** per iteration
- advect took walltime of 0.86 seconds for 10 iterations = **0.086 s** per iteration
- assuming `m` memory operations, effective bandwidth used = `data size x m / time per iteration`
- STREAM Triad bandwidth = 320 GB/s
- effective number of operations used in **collide** `m = 320 / (9.66 / 0.331)` = **10.96 ops**
- effective number of operations used in **advect** `m = 320 / (9.66 / 0.086)` = **2.84 ops**

## C++ with MPI and collide kernel using intrinsics

Improved effective number of operations used in collide kernel

- Case: One large block: **2.08 ops**
- Case: Few medium blocks: **2.15 ops**
- Case: Many small blocks: **2.46 ops**
