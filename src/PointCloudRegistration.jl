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
    IterativeClosestPoint,
    MeanAbsoluteDeviationMM,
    nonrigid_registration,
    CoherentPointDrift,
    DistancePreserving,
    nonrigid_sinkhorn,
    nonrigid_divfree,
    nonrigid_kc_springs,
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
include("rigid/annealing.jl")
include("rigid/stochastic_majorization_minimization.jl")
include("rigid/transformation_utils.jl")
include("rigid/restarts.jl")
include("rigid/accumulation.jl")
include("adam.jl")
include("convergence.jl")
include("rigid/rigid.jl")
include("rigid/kabsch.jl")
include("rigid/robust.jl")
include("rigid/kernel_correlation.jl")
include("rigid/iterative_closest_point.jl")
include("nonrigid/nonrigid.jl")
include("nonrigid/coherent_point_drift.jl")
include("nonrigid/distancepreserving.jl")
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
