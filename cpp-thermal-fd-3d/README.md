# Benchmarks

Tested on:

- 2S Intel Xeon 6148 Skylake
- 2S Intel Xeon 8360Y Ice Lake

Compiled Using

- icpc (ICC) 18.0.3 20180410
- icpc (ICC) 19.1.3.304 20200925

Compile and Run:

```
$ bash compile.sh
$ ./thermal
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
