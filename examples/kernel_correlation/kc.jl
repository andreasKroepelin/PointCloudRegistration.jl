using Revise
using HybridArrays
using HybridArrays.StaticArrays
using BenchmarkTools
using Random
using LinearAlgebra
using Makie, GLMakie
using ProgressMeter
using The2DShapeStructureDataset

using PointCloudRegistration

includet("logging.jl")
includet("plotting.jl")

d = 2
n = 100
# X = HybridMatrix{d, StaticArrays.Dynamic()}(cumsum(hcat(randn(d, n) .+ .5, randn(d, n) .+ [-.5,.0,]); dims = 2))
X = shape_sample_outline("apple-1", 100)
prepd_X = PointCloudRegistration.prepare_target(X; scale = 0.01);
# rotation = PointCloudRegistration.rand_rotation(Random.default_rng(), Val(d), eltype(X))
# translation = 100 * ones(SVector{d, Float64})
# Y = rotation' * (X[:, shuffle(axes(X, 2))[begin:2:end]] .- translation)
# Y .+= 3 * randn(size(Y))
Y = shape_sample_outline("apple-9", 100) .+ 10
register_no_correspondences(Y, prepd_X; restarts = 10)

rows = 2
cols = 5

al = AnalysisLogger([])
Logging.disable_logging(LogLevel(-2001))
T = with_logger(al) do
    register_no_correspondences(
        Y,
        prepd_X;
        iterations = 200,
        restarts = rows * cols - 1,
    )
end
Logging.disable_logging(Logging.Debug)

show_restarts(al, rows, cols, d)
