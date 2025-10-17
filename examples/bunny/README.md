# Stanford Bunny Example

In this example, we cut the Stanford Bunny point cloud into two sections with
varying overlap and try to reassemble it.

To run the reassembly experiment, execute
```sh
julia --project=BunnyReassembly --threads=auto --module=BunnyReassembly [args]
```
where the additional arguments can be `key=value` pairs with
- `numrestarts`: how many restarts to perform when registering the two parts for
  one specific combination of thresholds (default: 10)
- `numthresholds`: how many different thresholds to try out from 0 to 1
  (default: 10)
- `maxscalefactor`: the initial scale for the registration annealing, given as
  a multiple of the resolution (default: 4)
- `minscalefactor`: the final scale for the registration annealing, given as
  a multiple of the resolution (default: 1)

This produces a HDF5 file containing the results.

On a HPC cluster with SLURM, you can execute
```sh
sbatch --user-mail=your@email.com --mail-type=ALL run-bunny-reassembly.sh
```

To create the plots run
```sh
julia --project=PlotReassembly --module=PlotReassembly <command> <files>
```
where the command can be
- `success`: plot the boundary of success/failure for all given result files
  into one plot, automatically sorted by initial annealing scale
- `inits`: plot a representation of the initial annealing target for all given
  result files into one plot, automatically sorted by initial annealing scale
- `slices`: create an explanatory plot of what position in the plot from the
  `success` command corresponds to what slicing of the bunny
