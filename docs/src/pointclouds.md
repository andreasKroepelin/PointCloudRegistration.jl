# Representing point clouds

For the purpose of this package, we refer to a point cloud as a set of points,
i.e. a subset of ``\mathbb{R}^D`` for some integer ``D``.
We then also say that it is *``D``-dimensional*.
A point cloud can represent the atom coordinates of a molecule, the surface of
a 3D object, or many other things.

The following representations of an ``D``-dimensional point cloud are valid for
use with this package:
* a matrix with ``D`` rows where each column represents one point
  (i.e. `pointcloud isa AbstractMatrix{<: Real}`),
* a vector of vectors, all of length ``D``, where each element represents one
  point (i.e. `pointcloud isa AbstractVector{<: AbstractVector{<: Real}}`),
* an instance of [`PointCloud`](@ref), the type that this package uses
  internally and every other representation is converted to; provides the most
  control and performance guarantees (i.e. `pointcloud isa PointCloud{D}`).

```@docs
PointCloud
PointCloud(points::Any)
PointCloud(points::Any, weights::AbstractVector)
PointCloud(::PointCloud)
```
