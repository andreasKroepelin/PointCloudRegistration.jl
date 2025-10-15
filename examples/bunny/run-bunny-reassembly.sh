#!/bin/bash
#SBATCH --output=out.txt
#SBATCH --cpus-per-task=48

srun julia --project=BunnyReassembly --threads=auto --module=BunnyReassembly $@

