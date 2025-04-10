module PointCloudRegistration

using StaticArrays
using Accessors
using CoordinateTransformations
using Distances
using LinearAlgebra
using FFTW
using Random
using Bumper

include("kabsch.jl")
include("init.jl")
include("sqdist_based.jl")
include("kernel_correlation.jl")

end # module PointCloudRegistration
