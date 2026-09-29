module PointCloudRegistration

using StaticArrays
using FillArrays
using StructArrays
using Accessors
using CoordinateTransformations
using Rotations
using Distances
using NearestNeighbors
using SpecialFunctions
using LinearAlgebra
using LinearAlgebra: norm_sqr
using Random
using Statistics
using StatsBase
using PrecompileTools: @compile_workload
using ArgCheck

export PointCloud,
    WeightedPoint,
    points,
    weights,
    # = = = rigid registration = = =
    WithFlip,
    NoFlip,
    rigid_registration,
    rigidly_registered,
    GemanMcClureMM,
    KernelCorrelationMM,
    prepare_target_kernelcorrelation,
    Kabsch,
    IterativeClosestPoint,
    MeanAbsoluteDeviationMM,
    # = = = non-rigid registration = = =
    nonrigid_registration,
    nonrigidly_registered,
    CoherentPointDrift,
    prepare_source_coherentpointdrift,
    DistancePreserving,
    prepare_source_distancepreserving,
    EarthMover,
    DivergenceFree,
    # = = = annealing = = =
    LogAnnealingTo,
    LogAnnealingToNearestNeighborDistance,
    TargetScales,
    # = = = restarts = = =
    RandomRestarts,
    FixedRestarts,
    # = = = batching = = =
    FullBatch,
    StochasticBatch,
    # = = = correspondences = = =
    Ordered,
    Correspondences,
    matching_labels,
    Unknown,
    compatible_triangles,
    # = = = thinning
    thin_to_distance,
    thin_to_number,
    thin_to_grid,
    drop_threshold,
    drop_proportion,
    drop_quantile,
    # = = = density = = =
    density2pointcloud,
    # = = = plotting = = =
    pointcloudplotflat,
    pointcloudplotmesh,
    scaffoldplot,
    pointcloudplotflat!,
    pointcloudplotmesh!,
    scaffoldplot!

public VecOfSVec,
    PointType,
    SumOfWeightsType,
    Displacement,
    CpdDisplacement,
    DivFreeDisplacement,
    OrthogonalMatrix

include("pointcloud.jl")
include("common.jl")
include("rigid/rigid.jl")
include("rigid/correspondences.jl")
include("rigid/annealing.jl")
include("rigid/stochastic_majorization_minimization.jl")
include("rigid/orthogonal_matrix.jl")
include("rigid/transformation_utils.jl")
include("rigid/restarts.jl")
include("rigid/accumulation.jl")
include("adam.jl")
include("convergence.jl")
include("rigid/kabsch.jl")
include("rigid/robust.jl")
include("rigid/kernel_correlation.jl")
include("rigid/iterative_closest_point.jl")
include("nonrigid/nonrigid.jl")
include("nonrigid/coherent_point_drift.jl")
include("nonrigid/distancepreserving.jl")
# include("guess_correspondences.jl")
include("thinning.jl")
include("assets.jl")
include("plotting.jl")

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
