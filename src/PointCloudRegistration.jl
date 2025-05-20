module PointCloudRegistration

using StaticArrays
using HybridArrays
using FillArrays
using MappedArrays
using Accessors
using CoordinateTransformations
using Rotations
using Distances
using LinearAlgebra
using FFTW
using Random
using Logging
using Statistics
using PrecompileTools: @compile_workload
using ProgressLogging

export PointCloud,
    register_rmsd,
    register_gmc,
    register_kc,
    prepare_target_kc,
    BestTransformation,
    AllTransformations,
    DownTo,
    DefaultAnnealing

include("pointcloud.jl")
include("common.jl")
include("kabsch.jl")
include("init.jl")
include("robust.jl")
include("kernel_correlation.jl")

@compile_workload begin
    X2 = rand(2, 10)
    X3 = rand(3, 10)

    for X in (X2, X3)
        register_rmsd(X, X)
        register_gmc(X, X)
        register_kc(X, X)
    end
end

end # module PointCloudRegistration
