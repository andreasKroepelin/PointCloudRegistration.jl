module PointCloudRegistration

using StaticArrays
using FillArrays
using Accessors
using CoordinateTransformations
using Rotations
using Distances
using NearestNeighbors
using LinearAlgebra
using Random
using Statistics
using StatsBase
using PrecompileTools: @compile_workload
using ArgCheck

export PointCloud,
    register_rmsd,
    register_gmc,
    register_kc,
    register_cpd,
    prepare_target_kc,
    guess_correspondences,
    BestTransformation,
    AllTransformations,
    DownTo,
    TargetScales

public VecOfSVec

include("pointcloud.jl")
include("common.jl")
include("annealing.jl")
include("transformation_utils.jl")
include("restarts.jl")
include("accumulation.jl")
include("kabsch.jl")
include("robust.jl")
include("kernel_correlation.jl")
include("coherent_point_drift.jl")
include("guess_correspondences.jl")

@compile_workload begin
    X2 = rand(2, 10)
    X3 = rand(3, 10)
    Y2 = PointCloud(rand(2, 10), rand(10))
    Y3 = PointCloud(rand(3, 10), rand(10))

    for (Y, X) in ((Y2, X2), (Y3, X3))
        register_rmsd(Y, X)
        register_gmc(Y, X)
        register_kc(Y, X)
    end
end

end # module PointCloudRegistration
