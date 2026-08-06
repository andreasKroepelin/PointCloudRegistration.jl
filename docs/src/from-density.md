# Conversion from density

When the objects you want to register are only available as _density arrays_
such as images or voxel data, you can convert them to point clouds using the
[`density2pointcloud`](@ref) function.

If offset and scale of the coordinates are arbitrary anyways, you can simply
use any `AbstractArray` for the density.
For more control about the coordinates, use a `DimArray` from
[DimensionalData.jl](https://rafaqz.github.io/DimensionalData.jl/stable/).

Typically, the resulting point clouds will have way more points then is
necessary or practical for the task at hand.
See Section [Thinning](@ref) for solutions.

```@docs
density2pointcloud(::AbstractArray)
density2pointcloud(::DimensionalData.DimArray)
```
