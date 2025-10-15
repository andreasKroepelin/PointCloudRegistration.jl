#!/bin/bash
#SBATCH --output=out.txt
#SBATCH --cpus-per-task=48
#SBATCH --array=0-4

maxscalefactors=(10 6 3 2 1)

srun julia --project=BunnyReassembly --threads=auto --module=BunnyReassembly numrestarts=200 numthresholds=200 maxscalefactor=${maxscalefactors[$SLURM_ARRAY_TASK_ID]}

