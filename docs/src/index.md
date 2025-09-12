# PointCloudRegistration.jl

**PointCloudRegistration.jl** is a package for performing rigid and nonrigid
registration of point clouds, also known as *superimposing* or *aligning*.

## Point clouds

By point cloud, we refer to a set of points, i.e. a subset of ``\mathbb{R}^N``
for some integer ``N``.
We then also say that it is *``N``-dimensional*.
A point cloud can represent the atom coordinates of a molecule, the surface of
a 3D object, or many other things.

Computationally, we can store them as lists of vectors of equal length.
In fact, this package represents point clouds as a subtype of
`AbstractVector{SVector{N, T}}` via its [`PointCloud`](@ref) type that can
optionally also have weights for the points.
It is often convenient to construct a `PointCloud` from a matrix where every
column is one point.

Let us create two point clouds `source` and `target` where `target` is a
rotated and shifted version of `source`:
```@repl
using PointCloudRegistration
coords = [1.0:5.0 zeros(5)]'
source = PointCloud(coords)
angle = deg2rad(20)
rotation = [cos(angle) -sin(angle); sin(angle) cos(angle)]
translation = [13., 42.]
target = PointCloud(rotation * coords .+ translation)
```

How can we recover `rotation` and `translation` from `source` and `target`?


## API

```@docs
PointCloud
PointCloud(points::Any)
PointCloud(points::Any, weights::AbstractVector)
PointCloud(::PointCloud)
register_rmsd
register_gmc
register_kc
prepare_target_kc
```

## Common keyword arguments
This packages implements registration with respect to two non-convex
losses/scores.
While they are inherently different, the respective `register_gmc` and
`register_kc` functions share a common interface to deal with this
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
For `register_kc`, fewer iterations may happen if convergence occurs.

### `rng`
Default: `Random.default_rng()`

The random number generator to use to generate random initializations for each
restart.
You can specify this keyword argument to ensure reproducible initializations.


