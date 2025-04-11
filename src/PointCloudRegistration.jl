module PointCloudRegistration

using StaticArrays
using HybridArrays
using FillArrays
using Accessors
using CoordinateTransformations
using Distances
using LinearAlgebra
using FFTW
using Random
using Bumper

include("common.jl")
include("kabsch.jl")
include("init.jl")
include("sqdist_based.jl")
include("kernel_correlation.jl")

end # module PointCloudRegistration
