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

shuffled_proportions = 0.9:0.001:1
samples = 100
errors = Float64[]
@showprogress for shuffled_proportion in shuffled_proportions
    mean_error = 0.0
    for _ in 1:samples
        permuted_idcs = collect(axes(X, 2))
        shuffle!(@view(permuted_idcs[1:floor(Int, shuffled_proportion * n)]))

        Y = X[:, permuted_idcs]

        T = PointCloudRegistration.register_robustly(
            Y,
            X;
            minscale = 0.01,
            initialization = PointCloudRegistration.RandomRestarts(
                10,
                Xoshiro(123),
            ),
        )
        mean_error += log10(
            norm(vec(T.linear) - vec(one(T.linear))) +
            norm(T.translation - zero(T.translation)),
        )
    end
    push!(errors, exp10(mean_error / samples))
end

scatterlines(shuffled_proportions, errors; axis = (; yscale = log10))
