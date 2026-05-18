# Rigid registration

Rigid registration of two point clouds is the act of finding a rigid
transformation, i.e. a rotation and a translation, that transforms one point
cloud, the _source_, to optimally match the other point cloud, the _target_.

```@docs
rigid_registration(source, target; ordered)
rigid_registration(source, target, algorithm)
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

## Scale parameter
Both the Geman-McClure loss and the Kernel Correlation have a scale parameter
that determines up to what distances they are sensitive to.
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

## Restarts

## Batching

## Performance Tips
