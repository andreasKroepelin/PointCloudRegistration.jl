using Revise
using HybridArrays
using HybridArrays.StaticArrays
using BenchmarkTools
using Random
using LinearAlgebra
using Makie, GLMakie
using ProgressMeter

using PointCloudRegistration

d = 3
n = 1_000
X = HybridMatrix{d, StaticArrays.Dynamic()}(rand(d, n))
rotation = rand(SMatrix{3, 3, Float64})
translation = ones(SVector{3, Float64})
Y = rotation' * (X .- translation)
T = register_robustly(Y, X; scale = .01, iterations = 100, restarts = 10)
T.linear
T.translation

shuffled_proportions = 0.:0.1:1
samples = 100
errors = Float64[]
@showprogress for shuffled_proportion in shuffled_proportions
    mean_error = 0.0
    for _ in 1:samples
        permuted_idcs = collect(axes(X, 2))
        shuffle!(@view(permuted_idcs[1:floor(Int, shuffled_proportion * n)]))

        Y = rotation' * (X[:, permuted_idcs] .- translation)

        T = register_robustly(
            Y,
            X;
            scale = 0.01,
            iterations = 100,
            restarts = 10,
            rng = Xoshiro(123),
        )
        mean_error += (
            norm(vec(T.linear) - vec(rotation)) +
            norm(T.translation - translation)
        )
    end
    push!(errors, (mean_error / samples))
end

scatterlines(shuffled_proportions, errors; axis = (;))
