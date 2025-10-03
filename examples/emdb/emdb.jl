using Revise
using MRCFile
using GLMakie
using PointCloudRegistration
using PointCloudRegistration.StaticArrays
using Base.Iterators

mrc = read("/home/andreas/Downloads/emd_21619.map.gz", MRCData);

points_3 =
    map(SVector{3, Float32}, Iterators.product(voxelaxes(header(mrc))...));
points = reshape(points_3, :)

weights = max.(zero(eltype(mrc.data)), vec(mrc.data))

pc_naive = PointCloud(points, weights)

pc_dense = thin_droplowweight(pc_naive, 0.01)

length(pc_dense.points) / length(pc_naive.points)

maxw = maximum(pc_dense.weights)
scatter(pc_dense.points; color = tuple.(:blue, pc_dense.weights ./ maxw))

function report_iteration(; iteration, relchange, numclusters, additions)
    println("Iteration $iteration -- relative change $relchange -- $numclusters clusters ($additions newly added).")
end

pc = thin_dpmeans(pc_dense, 10.0f0; report_iteration)

maxw = maximum(pc.weights)
scatter(pc.points; color = tuple.(:blue, pc.weights ./ maxw))

pc_thin = thin_droplowweight(pc, 0.01)
maxw = maximum(pc_thin.weights)
scatter(pc_thin.points; color = tuple.(:blue, pc_thin.weights ./ maxw))
