# `PointCloudRegistration.jl`

<video src="PointCloudRegistration/raw/branch/main/examples/animations/register-julia-logo.mp4" type="video/mp4" width="600" height="400" controls>
</video>

**PointCloudRegistration** is a Julia package for registering two point clouds.
This is also known as aligning or superimposing.

Given two point clouds `source` and `target`, this package can perform
* **rigid registration**, i.e. rotate and translate `source` such that it
  matches `target`, or
* **nonrigid registration**, i.e. shift the points in `source` individually but
  coherently to match `target`.
The animation above shows first a rigid and then a nonrigid registration of the
blue and green point clouds.

Currently, the following four methods are implemented:
* **`register_rmsd`:**
  Rigid registration that minimizes the *root mean square deviation* between
  `source` and `target`; assumes that equal-index points correspond to each
  other and that all pairs of points are equally relevant.
* **`register_gmc`:**
  Rigid registration that minimizes the *Geman-McClure loss* between `source`
  and `target`; assumes that equal-index points correspond to each
  other but caps the influence of far apart pairs of points (outliers).
* **`register_kc`:**
  Rigid registration that maximizes the *kernel correlation* between `source`
  and `target`; assumes no correspondences and is robust against outliers.
* **`register_cpd`:**
  Nonrigid registration that estimates a *coherent point drift* from `source`
  to `target`; internally estimates correspondences and is configurable to
  account for outliers.

## Quickstart

```julia
julia> ]

pkg> add PointCloudRegistration

julia> using PointCloudRegistration

julia> target = cumsum(randn(3, 100), dims=2)
3×100 Matrix{Float64}:
  0.951501   0.86076    1.24536  …  14.4517   14.0935   14.7221
 -0.195826  -0.756038  -1.77237      8.11216   6.41248   7.25313
  0.185991   2.2701     2.06564      8.23514   6.98      7.02232

julia> source = target .+ 1
3×100 Matrix{Float64}:
 1.9515    1.86076    2.24536  …  15.4517   15.0935   15.7221
 0.804174  0.243962  -0.77237      9.11216   7.41248   8.25313
 1.18599   3.2701     3.06564      9.23514   7.98      8.02232


julia> T = register_gmc(source, target) # returns an `AffineMap` from CoordinateTransformations.jl
AffineMap([0.9999999999999999 1.0075742162992266e-16 -1.2502288632455386e-15; 2.524927231180851e-16 1.0 7.771435197962951e-16; 1.1339360284051926e-15 -6.237558778696607e-16 1.0], [-0.9999999999999947, -1.0000000000000067, -1.000000000000007])


julia> T.linear
3×3 StaticArraysCore.SMatrix{3, 3, Float64, 9} with indices SOneTo(3)×SOneTo(3):
 1.0           1.00757e-16  -1.25023e-15
 2.52493e-16   1.0           7.77144e-16
 1.13394e-15  -6.23756e-16   1.0

julia> T.translation
3-element StaticArraysCore.SVector{3, Float64} with indices SOneTo(3):
 -0.9999999999999947
 -1.0000000000000067
 -1.000000000000007

julia> T.(eachcol(source)) ≈ eachcol(target)
true
```
