# Benchmarks

Tested on:

- 2S Intel Xeon 6148 Skylake
- 2S Intel Xeon 8360Y Ice Lake

Compile:

```
$ icpc --version
icpc (ICC) 18.0.3 20180410
Copyright (C) 1985-2018 Intel Corporation.  All rights reserved.

$ bash compile.sh
```

Run:

```
mpirun -n $nprocs ./thermal
```

Walltime (seconds):

| nprocs | Skylake | Ice Lake |
| -----: | ------: | -------: |
| 1      | 62.35   | 53.99    |

