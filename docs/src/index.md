# PointCloudRegistration.jl

## API

```@docs
PointCloud
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


### `accumulator`
Default: `PointCloudRegistration.BestTransformation`

How to accumulate the results of the individual restarts.
This affects the return type of the registration functions.
`accumulator` expects a subtype of
`PointCloudRegistration.AbstractTransformationAccumulator`.
The following two subtypes are provided:
```@docs
PointCloudRegistration.BestTransformation
PointCloudRegistration.AllTransformations
```
