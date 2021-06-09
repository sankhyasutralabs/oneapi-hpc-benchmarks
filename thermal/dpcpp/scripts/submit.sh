#!/bin/bash

bsub -P S:SankhyaTest -t 10 -p workq scripts/submit.slurm
