module PointCloudRegistration

using StaticArrays
using HybridArrays
using FillArrays
using MappedArrays
using Accessors
using CoordinateTransformations
using Rotations
using Distances
using NearestNeighbors
using LinearAlgebra
using FFTW
using Random
using Logging
using Statistics
using StatsBase
using PrecompileTools: @compile_workload
using ProgressLogging

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
include("kabsch.jl")
include("transformation_utils.jl")
include("robust.jl")
include("kernel_correlation.jl")
include("coherent_point_drift.jl")
include("guess_correspondences.jl")

#=
@compile_workload begin
    X2 = rand(2, 10)
    X3 = rand(3, 10)

    for X in (X2, X3)
        register_rmsd(X, X)
        register_gmc(X, X)
        register_kc(X, X)
    end
end
=#

end # module PointCloudRegistration
