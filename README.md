# Point Cloud Registration

<video src="examples/animations/register-julia-logo.mp4"></video>

<video src="examples/animations/register-julia-logo.mp4" type="video/mp4" width="300" height="200" controls>
</video>

![animation of Julia logo](examples/animations/register-julia-logo.mp4)

This package provides the function `register(source, target)` to find an
`AffineMap` consisting of a rotation and a translation that maps a point cloud
`source` to a point cloud `target`.

`source` and `target` are expected to be matrices with the same number of rows,
such that each column represents one point.

`register` has an optional keyword argument `correspondences` with the following
possible values:
- `Val(:unknown)` (the default):
  You know nothing about what point in the source corresponds to what point in
  the target.
  It is not even assumed that there is a one-to-one correspondence.
  Registration happens by maximizing the _kernel correlation_ between the two
  point clouds.
- `Val(:known)`:
  Both point clouds have the same number of points (columns) and you are sure
  that the `i`-th column in `source` corresponds to the `i`-th column in
  `target`.
  Registration happens by minimizing the RMSD via the Kabsch algorithm.
- `Val(:unsure)`:
  Both point clouds have the same number of points (columns) and you assume that
  the `i`-th column in `source` corresponds to the `i`-th column in `target` but
  some of those correspondences might be wrong.
  Registration happens by minimizing the Gelman-McClure loss between the two
  point clouds.
