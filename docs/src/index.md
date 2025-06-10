# PointCloudRegistration.jl

## API

```@autodocs
Modules = [PointCloudRegistration]
Order = [:function, :type]
Private = false
```

## Dealing with non-convexity
This packages implements registration with respect to two non-convex
losses/scores.
While they are inherently different, the respective `register_*` functions share
a common interface to deal with this non-convexity, namely certain keyword
arguments:

### `scale`
