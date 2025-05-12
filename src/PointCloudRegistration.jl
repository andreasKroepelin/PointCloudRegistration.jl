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

export PointCloud, register_rmsd, register_gmc, register_kc, prepare_target_kc

include("pointcloud.jl")
include("common.jl")
include("kabsch.jl")
include("init.jl")
include("robust.jl")
include("kernel_correlation.jl")

end # module PointCloudRegistration
