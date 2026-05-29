# API Overview

## Registration

The two main functions provided by this package are [`rigid_registration`](@ref)
and [`nonrigid_registration`](@ref).
They both work in a similar way:
```julia
transformation = [non]rigid_registration(source, target, algorithm)
```
where `source` and `target` can be given in a form described in the previous
section ([Representing point clouds](@ref)).
They must have matching dimensions (both 2D or both 3D and so forth).

The type of the returned `transformation` depends on the concrete registration.
However, it always has the property that `transformation(source)` provides the
registered source.
(To avoid potential type piracy, this only works when `source isa PointCloud`.)

The third argument, `algorithm`, determines how the registration is performed.
See the following pages for available choices.

## Utilities

PointCloudRegistration.jl also offers some convenient utility functions:
* [Thinning](@ref): 
* [Conversion from density](@ref):
