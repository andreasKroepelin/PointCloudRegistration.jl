using Revise
using MRCFile
using GLMakie
using PointCloudRegistration
using PointCloudRegistration.StaticArrays
using Base.Iterators

mrc = read("/home/andreas/Downloads/emd_21619.map.gz", MRCData);

points_3 = map(SVector{3, Float32}, Iterators.product(voxelaxes(header(mrc))...))
points = reshape(points_3, :)

weights = max.(zero(eltype(mrc.data)), vec(mrc.data))

pc_dense = thin_droplowweight(PointCloud(points, weights), .01)

maxw = maximum(pc_dense.weights)
scatter(pc_dense.points; color = tuple.(:blue, pc_dense.weights ./ maxw))

pc = thin_dpmeans(pc_dense, 10f0)

maxw = maximum(pc.weights)
scatter(pc.points; color = tuple.(:blue, pc.weights ./ maxw))
