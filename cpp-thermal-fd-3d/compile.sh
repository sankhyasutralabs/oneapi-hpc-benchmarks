#!/bin/bash

icpc -std=c++11 -O3 -mcmodel=large -xCORE-AVX512 -qopt-zmm-usage=high -qopt-streaming-stores always thermal.cpp -o thermal -DNX=1080 -DNT=20
