module DimensionalDataExt
using PointCloudRegistration
import PointCloudRegistration: density2pointcloud
using StaticArrays
using DimensionalData

"""
    density2pointcloud(density::DimArray{T, N}) where {T, N}

Creates an `N`-dimensional point cloud that represents the `density` by placing
a point at the axes values of every entry of `density` with the corresponding
value as the point's weight.

# Example
```julia
julia> density = DimArray(
        reshape(11:19, 3, 3),
        (X = range(42, 64; length=3), Y = [-10.0, 5.0, 100.0])
    )
┌ 3×3 DimArray{Int64, 2} ┐
├────────────────────────┴─────────────────── dims ┐
  ↓ X Sampled{Float64} 42.0:11.0:64.0 ForwardOrdered Regular Points,
  → Y Sampled{Float64} [-10.0, …, 100.0] ForwardOrdered Irregular Points
└──────────────────────────────────────────────────┘
  ↓ →  -10.0   5.0  100.0
 42.0   11    14     17
 53.0   12    15     18
 64.0   13    16     19

julia> density2pointcloud(density)
2-dimensional point cloud with 9 points of eltype Float64
  42.0   53.0   64.0  …   42.0   53.0   64.0
 -10.0  -10.0  -10.0     100.0  100.0  100.0
and weights
 9-element UnitRange{Int64}
 11  12  13  14  15  16  17  18  19

```
"""
function density2pointcloud(density::DimArray)
    points = vec(map(SVector, Iterators.product(dims(density)...)))
    weights = vec(density)
    PointCloud(points, weights)
end

end
