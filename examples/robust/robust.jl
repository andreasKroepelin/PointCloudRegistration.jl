using Revise
using HybridArrays
using HybridArrays.StaticArrays
using BenchmarkTools
using Random
using LinearAlgebra
using Makie, GLMakie
using ProgressMeter

using PointCloudRegistration

includet("logging.jl")

d = 3
n = 100
X = HybridMatrix{d, StaticArrays.Dynamic()}(cumsum(randn(d, n) .+ 1; dims = 2))
rotation = PointCloudRegistration.rand_rotation(
    Random.default_rng(),
    Val(d),
    eltype(X),
)
translation = ones(SVector{d, Float64})
permuted_idcs = collect(axes(X, 2))
shuffled_proportion = 0.9
shuffle!(@view(permuted_idcs[1:floor(Int, shuffled_proportion * n)]))
Y = rotation' * (X[:, permuted_idcs] .- translation)
al = AnalysisLogger([])
Logging.disable_logging(LogLevel(-2001))
T = with_logger(al) do
    register_gmc(
        Y,
        X;
        scale = 1.0,
        iterations = 20,
        annealing = 3,
        restarts = 10,
    )
end
Logging.disable_logging(Logging.Debug)

fig = Figure()
slider = Slider(fig[2, 1]; range = eachindex(al.logs))
curr_log = @lift al.logs[$(slider.value)]
curr_rotation = @lift ($curr_log).rotation
curr_translation = @lift ($curr_log).translation
curr_trY = @lift $curr_rotation * Y .+ $curr_translation
# curr_source_mean = @lift $curr_rotation * ($curr_log).source_mean + $curr_translation
# curr_target_mean = @lift ($curr_log).target_mean
correspondences =
    @lift stack(Iterators.flatten(zip(eachcol(X), eachcol($curr_trY))))
weights = @lift ($curr_log).mm_weights .* 5 .+ 0.1
title = @lift string(
    "restart = ",
    ($curr_log).restart,
    " ; iter = ",
    ($curr_log).iter,
)
ax = Axis(fig[1, 1]; autolimitaspect = 1, title)
scatter!(ax, X; color = :tomato)
scatter!(ax, curr_trY; color = :teal)
# scatter!(ax, curr_source_mean, markersize = 20, marker = :hexagon, color = :teal)
# scatter!(ax, curr_target_mean, markersize = 20, marker = :star5, color = :black)
linesegments!(ax, correspondences; color = :grey, linewidth = weights)

ax2, hm = heatmap(fig[3, 1], stack(log -> log.mm_weights, al.logs)')
Colorbar(fig[3, 2], hm)
# ax2 = Axis(fig[3, 1])
# for ws in eachrow(stack(log -> log.mm_weights, al.logs))
#     lines!(ax2, ws)
# end
# vlines!(ax2, slider.value)

shuffled_proportions = 0.0:0.001:1
samples = 2
errors = Float64[]
@showprogress for shuffled_proportion in shuffled_proportions
    mean_error = 0.0
    for _ in 1:samples
        permuted_idcs = collect(axes(X, 2))
        shuffle!(@view(permuted_idcs[1:floor(Int, shuffled_proportion * n)]))

        Y = rotation' * (X[:, permuted_idcs] .- translation)

        T = register_gmc(
            Y,
            X;
            scale = 1.0,
            iterations = 10,
            annealing = 10,
            restarts = 20,
            rng = Xoshiro(123),
        )
        @assert T.linear' * T.linear ≈ one(T.linear)
        mean_error += (norm(vec(T.linear) - vec(rotation)) # +
        # norm(T.translation - translation)
        )
    end
    push!(errors, (mean_error / samples))
end

scatterlines(shuffled_proportions, errors; axis = (;))
