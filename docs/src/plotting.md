# Plotting

This package provides some recipes for the [Makie.jl](https://makie.org/)
plotting package.

## Point clouds

While you can simply call `scatter(pointcloud.points)`, this does not reflect
the weights and can thus greatly misrepresent the data.
Instead, you can use [`pointcloudplotflat`](@ref) and
[`pointcloudplotmesh`](@ref) for a more appropriate visualization.

Note that these recipes are defined as the default plot types for
`PointCloud{2}` and `PointCloud{3}`, respectively, so you can also just use
`plot(pointcloud)`.

```@docs
pointcloudplotflat
pointcloudplotmesh
```

## Displacement

For non-rigid registration methods returning a
[`PointCloudRegistration.Displacement`](@ref), you can plot this result using
Makie's [`Makie.arrows2d`](@extref) and [`Makie.arrows3d`](@extref) like so:
```julia
displacement = nonrigid_registration(source, target, EarthMover(...))
arrows2d(displacement)
```

## Scaffold

The [`DistancePreserving`](@ref) non-rigid registration algorithm is concerned
with distances between neighboring points in the source point cloud.
These can be seen as some kind of scaffold that constrains the deformation of
the source.
Use [`scaffoldplot`](@ref) to visualize it.

```@docs
scaffoldplot
```
