#!/bin/bash
#SBATCH --output=out.txt
#SBATCH --cpus-per-task=48
#SBATCH --array=0-4

maxscalefactors=("10" "6" "4" "2" "1.5")

srun julia \
  --project=BunnyReassembly \
  --threads=auto \
  --module=BunnyReassembly \
  numrestarts=500 \
  numthresholds=250 \
  maxscalefactor=${maxscalefactors[$SLURM_ARRAY_TASK_ID]}

