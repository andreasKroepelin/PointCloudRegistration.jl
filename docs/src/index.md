# PointCloudRegistration.jl

## API

```@autodocs
Modules = [PointCloudRegistration]
Order = [:function, :type]
Private = false
```

## Common keyword arguments
This packages implements registration with respect to two non-convex
losses/scores.
While they are inherently different, the respective `register_gmc` and
`register_kc` functions share a common interface to deal with this
non-convexity, namely certain keyword arguments:

### `scale`

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
```@docs; canonical=false
DownTo
DefaultAnnealing
```

