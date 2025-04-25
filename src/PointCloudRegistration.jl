module PointCloudRegistration

using StaticArrays
using HybridArrays
using FillArrays
using MappedArrays
using Accessors
using CoordinateTransformations
using Distances
using LinearAlgebra
using FFTW
using Random
using Logging
using Statistics
using Bumper

export PointCloud, register, prepare_target_kernel_correlation

include("pointcloud.jl")
include("common.jl")
include("kabsch.jl")
include("init.jl")
include("robust.jl")
include("kernel_correlation.jl")

function register(source, target; correspondences = Val(:unknown), kwargs...)
    if correspondences isa Val{:unknown}
        register_no_correspondences(source, target; kwargs...)
    elseif correspondences isa Val{:known}
        register_naively(source, target; kwargs...)
    elseif correspondences isa Val{:unsure}
        register_robustly(source, target; kwargs...)
    elseif correspondences isa Symbol
        register(source, target; correspondences = Val(correspondences), kwargs...)
    else
        throw(ArgumentError("`correspondences` must be one of `:unknown`, `:known`, or `:unsure`."))
    end
end

end # module PointCloudRegistration
