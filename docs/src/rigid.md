# Rigid registration

Rigid registration of two point clouds is the act of finding a rigid
transformation, i.e. a rotation and a translation, that transforms one point
cloud, the _source_, to optimally match the other point cloud, the _target_.

```@docs
rigid_registration(source, target; ordered)
rigid_registration(source, target, algorithm, flip)
```

## Rigid registration algorithms

- [Geman-McClure cost](@ref)
- [Kernel Correlation](@ref)
- [Kabsch](@ref)
- [Iterative Closest Point](@ref)
- [Mean Absolute Deviation](@ref)

### Geman-McClure cost
```@docs
GemanMcClureMM
rigid_registration(source, target, ::GemanMcClureMM)
```

### Kernel Correlation
```@docs
KernelCorrelationMM
rigid_registration(source, target, ::KernelCorrelationMM; target_preparation)
prepare_target_kernelcorrelation
```

### Kabsch
```@docs
Kabsch
rigid_registration(source, target, ::Kabsch)
```

### Iterative Closest Point
```@docs
IterativeClosestPoint
rigid_registration(source, target, ::IterativeClosestPoint)
```

### Mean Absolute Deviation
```@docs
MeanAbsoluteDeviationMM
rigid_registration(source, target, ::MeanAbsoluteDeviationMM)
```

## Dealing with non-convexity

Only [`Kabsch`](@ref) and [`MeanAbsoluteDeviationMM`](@ref) solve _convex_
optimization problems, meaning they are guaranteed to find the global optimum.
The other algorithms perform non-convex optimization and can thus get stuck in
local optima, in principle.
To counteract this issue, three strategies can be employed:
Simulated annealing, multiple restarts, and stochastic batching.

### Scale parameter
Both the Geman-McClure loss and the Kernel Correlation have a scale parameter
that determines up to what distances they are sensitive to.
Thus, the scale should eventually take a value that is relevant for the
application at hand.
In the simplest case, you can just set it to a **scalar value**.

To perform simulated annealing, you can specify how the scale should be
successively decreased.
For maximum control, you can set `scale` to any **`AbstractVector{<: Number}`**.

If you are unsure what values are sensible to use, the following heuristics are
implemented:
```@docs
LogAnnealingToNearestNeighborDistance
LogAnnealingTo
TargetScales
```

### Restarts
For multiple restarts, you can specify if the initial optimization candidates
should be chosen randomly or from a given list.

```@docs
RandomRestarts
FixedRestarts
```

### Batching
In machine learning, the technique of _Stochastic Gradient Descent_ is a
standard strategy to speed up training and avoid local optima.
The idea is that each training iteration uses only a small randomly chosen
subset of the training data.
Similarly, it can help to only use some of the points in every iteration of a
rigid registration algorithm.

```@docs
FullBatch
StochasticBatch
```


## Reflections

By default, rigid registration does not allow _reflecting_ or _flipping_ the
source.
The phyiscal process producing the point cloud data typically determines their
orientation.
Sometimes, however, orientation is not known and has to be estimated as well.
To this end, the [`rigid_registration`](@ref) function accepts an additional
argument that can either be [`WithFlip()`](@ref) or [`NoFlip()`](@ref).

This choice influences the return type of `rigid_registration`.
For `NoFlip()` and `source isa PointCloud{N}`, `target isa PointCloud{N}`, it is
a subtype of
```julia
CoordinateTransformations.AffineMap{
  <: Rotations.RotMatrix{N},
  <: StaticArrays.SVector{N},
}
```
For `WithFlip()` and `source isa PointCloud{N}`, `target isa PointCloud{N}`, it
is a subtype of
```julia
CoordinateTransformations.AffineMap{
  <: PointCloudRegistration.OrthogonalMatrix{N},
  <: StaticArrays.SVector{N},
}
```

```@docs
WithFlip
NoFlip
PointCloudRegistration.OrthogonalMatrix
```
