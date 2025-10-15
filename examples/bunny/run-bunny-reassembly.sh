#!/bin/bash
#SBATCH --output=out.txt
#SBATCH --cpus-per-task=48

julia --project=BunnyReassembly --threads=auto --module=BunnyReassembly $@

