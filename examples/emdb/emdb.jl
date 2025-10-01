using Revise
using MRCFile
using GLMakie
using PointCloudRegistration
using PointCloudRegistration.StaticArrays
using Base.Iterators

mrc = read("/home/andreas/Downloads/emd_21619.map.gz", MRCData);

points_3 = map(SVector{3, Float32}, Iterators.product(voxelaxes(header(mrc))...))
points = reshape(points_3, :)

weights = vec(mrc.data)
