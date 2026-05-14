module PointCloudRegistration

using StaticArrays
using FillArrays
using Accessors
using CoordinateTransformations
using Rotations
using Distances
using NearestNeighbors
using SpecialFunctions
using SparseArrays
using LinearAlgebra
using Random
using Statistics
using StatsBase
using PrecompileTools: @compile_workload
using ArgCheck

export PointCloud,
    rigid_registration,
    GemanMcClureMM,
    KernelCorrelationMM,
    prepare_target_kernelcorrelation,
    Kabsch,
    rigid_mad,
    rigid_icp,
    nonrigid_cpd,
    nonrigid_sinkhorn,
    nonrigid_divfree,
    nonrigid_kc_springs,
    nonrigid_distancepreserving,
    correspondences,
    displacements,
    apply_displacements,
    prepare_source_distancepreserving,
    guess_correspondences,
    DownTo,
    TargetScales,
    RandomRestarts,
    FixedRestarts,
    NoSmm,
    Smm,
    GeneralizedLogNormalRegularizer,
    thin_to_distance,
    thin_to_number,
    thin_to_grid,
    drop_threshold,
    drop_proportion,
    drop_quantile,
    density2pointcloud

public VecOfSVec

include("pointcloud.jl")
include("common.jl")
include("annealing.jl")
include("stochastic_majorization_minimization.jl")
include("transformation_utils.jl")
include("restarts.jl")
include("accumulation.jl")
include("adam.jl")
include("convergence.jl")
include("rigid.jl")
include("kabsch.jl")
include("robust.jl")
include("kernel_correlation.jl")
include("iterative_closest_point.jl")
include("nonrigid.jl")
include("coherent_point_drift.jl")
include("distancepreserving.jl")
include("guess_correspondences.jl")
include("thinning.jl")
include("assets.jl")

@compile_workload begin
    X2 = rand(2, 10)
    X3 = rand(3, 10)
    Y2 = PointCloud(rand(2, 10), rand(10))
    Y3 = PointCloud(rand(3, 10), rand(10))

    for (Y, X) in ((Y2, X2), (Y3, X3))
        # rigid_rmsd(Y, X)
        # rigid_gmc(Y, X)
        # rigid_kc(Y, X)
    end
end

end # module PointCloudRegistration
