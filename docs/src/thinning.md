# Thinning

In the typical regime of two- or three-dimensional point clouds, rigid
registration actually needs to estimate only three (rotation angle and two
translation components) or six (three Euler angles and three translation
components) parameters, respectively.
Point clouds with, say, millions of points therefore often provide an excessive
amount of data for the rigid registration problem.
At the same time, the registration algorithms take longer for larger point
clouds.

Consequently, it can make sense to _thin_ point clouds before performing
registration.
This package provides the three functions [`thin_to_distance`](@ref),
[`thin_to_grid`](@ref), and [`thin_to_number`](@ref) for this task.

```@docs
thin_to_distance
thin_to_grid
thin_to_number
```

Additionally, one may want to _drop_ points in a point cloud that have a low
weight.
For this, there are the function [`drop_threshold`](@ref),
[`drop_proportion`](@ref), and [`drop_quantile`](@ref).

```@docs
drop_threshold
drop_proportion
drop_quantile
```
