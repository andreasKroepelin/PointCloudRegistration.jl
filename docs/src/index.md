# PointCloudRegistration.jl

**PointCloudRegistration.jl** is a package for performing rigid and nonrigid
registration of point clouds, also known as *superimposing* or *aligning*.

## What is (non)rigid registration?

For registration, we always think about two [point clouds](#Representing-point-clouds)
`source` and `target` where we want to transform `source` such that it sits
on top of `target` as close as possible.

When doing **rigid registration**, we can only globally rotate and translate
`source` such that it matches `target`.
With **nonrigid registration**, we shift the points in `source` individually but
coherently to match `target`.

## How to do...

This is a quick guide that should help you find the appropriate function from
PointCloudRegistration.jl for your usecase.
We always assume that you have two point clouds `source` and `target`in a [valid
representation](#Representing-point-clouds).

```@setup howto
using PointCloudRegistration
source = vcat((1:9)', zeros(9)')
angle = deg2rad(20)
rotation = [cos(angle) -sin(angle); sin(angle) cos(angle)]
translation = [13., 42.]
target = rotation * source .+ translation
```

### ... rigid registration with *known* correspondences

!!! note "Assumption"
    You know for certain that the ``i``-th point in `source` and the ``i``-th
    point in `target` belong together.
    This especially means that `source` and `target` have the same number of
    points.
    Also, *every* such pair of points can be explained using a global rotation
    and translation.

This is the simplest case.
You can use [`rigid_rmsd`](@ref) to minimize the *Root Mean Square Deviation*:
```@repl howto
using PointCloudRegistration
T = rigid_rmsd(source, target)
T.linear
T.translation
```
### ... rigid registration with *unreliable* correspondences

!!! note "Assumption"
    You assume that the ``i``-th point in `source` and the ``i``-th point in
    `target` belong together.
    This especially means that `source` and `target` have the same number of
    points.
    However, you expect that some of these pairs are *unreliable* in the sense
    that they might be incorrect correspondences or they do correspond but
    cannot be explained by a global rotation and translation.

You can use [`rigid_gmc`](@ref) to minimize the *Geman-McClure loss* which
is a variant of the RMSD that caps the influence of far-apart pairs:
```@repl howto
using PointCloudRegistration
T = rigid_gmc(source, target)
T.linear
T.translation
```

This method works iteratively and has a few parameters you can tweak, see the
documentation of [`rigid_gmc`](@ref).
All of them have sensible defaults, however.

### ... rigid registration with *unknown* correspondences

!!! note "Assumption"
    You know nothing about the correspondence between points in `source` and in
    `target`.
    `source` and `target` need not have the same number of points.
    You might not even expect there to be any correspondences at all.
    All you know is that rotating and translating `source` as a whole should
    match `target` as a whole.

You can use [`rigid_kc`](@ref) to maximize the *kernel correlation* which
can be thought of as a scalar product of the densities induced by `source` and
`target`:
```@repl howto
using PointCloudRegistration
T = rigid_kc(source, target)
T.linear
T.translation
```

Like [`rigid_gmc`](@ref), this method works iteratively and has a few
parameters you can tweak, see the documentation of [`rigid_kc`](@ref).
All of them have sensible defaults, again.

Note that this method has a runtime complexity linear in the number of points
of both point clouds, despite not knowing correspondences.
This is achieved by
* never estimating correspondences internally (opposed to, say, [Iterative
  Closest Point](https://en.wikipedia.org/wiki/Iterative_closest_point)) and
* precomputing relevant quantities related to the `target`.

This precomputation can be cached if you want to register multiple sources to
the same target (see [`prepare_target_kc`](@ref)).

## Representing point clouds

For the purpose of this package, we refer to a point cloud as a set of points,
i.e. a subset of ``\mathbb{R}^N`` for some integer ``N``.
We then also say that it is *``N``-dimensional*.
A point cloud can represent the atom coordinates of a molecule, the surface of
a 3D object, or many other things.

The following representations of an ``N``-dimensional point cloud are valid for
use with this package:
* a matrix with ``N`` rows where each column represents one point
  (i.e. `pointcloud isa AbstractMatrix{<: Real}`),
* a vector of vectors, all of length ``N``, where each element represents one
  point (i.e. `pointcloud isa AbstractVector{<: AbstractVector{<: Real}}`),
* an instance of [`PointCloud`](@ref), the type that this package uses
  internally and every other representation is converted to; provides the most
  control and performance guarantees (i.e. `pointcloud isa PointCloud{N}`).


Let us create two point clouds `source` and `target` where `target` is a
rotated and shifted version of `source`:
```@repl 1
using PointCloudRegistration
source = cumsum(randn(2, 100) .+ [1, 0], dims = 2)
angle = deg2rad(20)
rotation = [cos(angle) -sin(angle); sin(angle) cos(angle)]
translation = [13., 42.]
target = rotation * source .+ translation
using GLMakie # hide
fig = Figure() # hide
ax = Axis(fig[1, 1]; autolimitaspect = 1) # hide
scatter!(ax, source; label = "source") # hide
scatter!(ax, target; label = "target") # hide
axislegend(ax; position = (:top, :left)) # hide
save("source-target.png", fig); # hide
```

![](source-target.png)


How can we recover `rotation` and `translation` from `source` and `target`?


## API

```@docs
PointCloud
PointCloud(points::Any)
PointCloud(points::Any, weights::AbstractVector)
PointCloud(::PointCloud)
rigid_rmsd
rigid_gmc
rigid_kc
prepare_target_kc
```

## Common keyword arguments
This packages implements registration with respect to two non-convex
losses/scores.
While they are inherently different, the respective `rigid_gmc` and
`rigid_kc` functions share a common interface to deal with this
non-convexity, namely certain keyword arguments:

### `scale`
Default: `TargetScales(5)`

Both the Geman-McClure loss and the Kernel Correlation have a scale parameter
that determines to what distances they are sensitive to.
Thus, the scale should eventually take a value that is relevant for the
application at hand.
In the simplest case, you can just set to a **scalar value**.

However, optimizing the rotation and translation can easily get stuck in a
non-global optimum when starting with a scale too small.
Too avoid that, you can specify how the scale should be successively decreased.
For maximum control, you can set `scale` to any **`AbstractVector{<: Real}`**.

If you are unsure what values are sensible to use, two heuristics are
implemented.
```@docs
DownTo
TargetScales
```

### `restarts`
Default: `5`

The non-convex nature of the optimization makes it prone to run into local
optima.
The parameter `restarts` specifies how often to perform a restart from a random
initial transformation.

### `iterations`
Default: `10`

How many optimizaton iterations to perform per restart.
For `rigid_kc`, fewer iterations may happen if convergence occurs.

### `rng`
Default: `Random.default_rng()`

The random number generator to use to generate random initializations for each
restart.
You can specify this keyword argument to ensure reproducible initializations.


