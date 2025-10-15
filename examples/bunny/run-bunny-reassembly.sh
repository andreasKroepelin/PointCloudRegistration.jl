#!/bin/bash
#SBATCH --output=out.txt
#SBATCH --cpus-per-task=48

srun julia --project=BunnyReassembly --threads=auto --module=BunnyReassembly numrestarts=200 numthresholds=200 maxscalefactor=6
srun julia --project=BunnyReassembly --threads=auto --module=BunnyReassembly numrestarts=200 numthresholds=200 maxscalefactor=2
srun julia --project=BunnyReassembly --threads=auto --module=BunnyReassembly numrestarts=200 numthresholds=200 maxscalefactor=.9 minscalefactor=.5

