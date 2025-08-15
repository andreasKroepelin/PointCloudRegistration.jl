using Revise
using HybridArrays
using HybridArrays.StaticArrays
using BenchmarkTools
using Random
using LinearAlgebra
using Makie, GLMakie
using ProgressMeter
# using The2DShapeStructureDataset
using BioStructures

using PointCloudRegistration

includet("logging.jl")
includet("plotting.jl")

d = 2
n = 100
# X = HybridMatrix{d, StaticArrays.Dynamic()}(cumsum(hcat(randn(d, n) .+ .5, randn(d, n) .+ [-.5,.0,]); dims = 2))
# X = shape_sample_outline("apple-1", 10000) # [:, 20:50] # just the leaves
X = PointCloud(coordarray(retrievepdb("1su4"; dir = tempdir()), calphaselector))
prepd_X = prepare_target_kc(X; scale = 0.01);
# rotation = PointCloudRegistration.rand_rotation(Random.default_rng(), Val(d), eltype(X))
# translation = 100 * ones(SVector{d, Float64})
# Y = rotation' * (X[:, shuffle(axes(X, 2))[begin:2:end]] .- translation)
# Y .+= 3 * randn(size(Y))
# Y = shape_sample_outline("apple-2", 10000) .+ 50
# Y = shape_coords("apple-1")
Y = PointCloud(
    coordarray(retrievepdb("1iwo"; dir = tempdir())["A"], calphaselector),
)
PointCloudRegistration.register_kc_naive(Y, X; restarts = 100)
transformation = register_kc(Y, X; restarts = 100).transformation
transformation = PointCloudRegistration.register_kc_rff(
    Y,
    X;
    features = 50000,
    restarts = 0,
    scale = 1.0,
).transformation

side_by_side(Y, X, transformation)

rows = 2
cols = 2

al = AnalysisLogger([])
Logging.disable_logging(LogLevel(-2001))
T = with_logger(al) do
    register_kc(Y, prepd_X; iterations = 200, restarts = rows * cols - 1)
end
Logging.disable_logging(Logging.Debug)

show_restarts(al, rows, cols, prepd_X)
