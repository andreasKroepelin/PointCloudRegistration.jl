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

export register_robustly, register_no_correspondences

include("common.jl")
include("kabsch.jl")
include("init.jl")
include("robust.jl")
include("kernel_correlation.jl")

end # module PointCloudRegistration
