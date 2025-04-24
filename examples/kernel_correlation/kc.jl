using Revise
using HybridArrays
using HybridArrays.StaticArrays
using BenchmarkTools
using Random
using LinearAlgebra
using Makie, GLMakie
using ProgressMeter
using The2DShapeStructureDataset
using BioStructures

using PointCloudRegistration

includet("logging.jl")
includet("plotting.jl")

d = 2
n = 100
# X = HybridMatrix{d, StaticArrays.Dynamic()}(cumsum(hcat(randn(d, n) .+ .5, randn(d, n) .+ [-.5,.0,]); dims = 2))
X = shape_coords("apple-1")
# X = PointCloud(coordarray(retrievepdb("1su4", dir = tempdir()), calphaselector))
prepd_X = prepare_target_kernel_correlation(X);
# rotation = PointCloudRegistration.rand_rotation(Random.default_rng(), Val(d), eltype(X))
# translation = 100 * ones(SVector{d, Float64})
# Y = rotation' * (X[:, shuffle(axes(X, 2))[begin:2:end]] .- translation)
# Y .+= 3 * randn(size(Y))
# Y = shape_coords("apple-9")
Y = PointCloud(coordarray(retrievepdb("1iwo", dir = tempdir())["A"], calphaselector))
transformation = register(Y, prepd_X; restarts = 100)

side_by_side(Y, X, transformation)

rows = 3
cols = 6

al = AnalysisLogger([])
Logging.disable_logging(LogLevel(-2001))
T = with_logger(al) do
    register(
        Y,
        prepd_X;
        iterations = 200,
        restarts = rows * cols - 1,
    )
end
Logging.disable_logging(Logging.Debug)

show_restarts(al, rows, cols, prepd_X)
