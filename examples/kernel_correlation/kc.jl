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

d = 2
n = 1_000
X = HybridMatrix{d, StaticArrays.Dynamic()}(cumsum(randn(d, n) .+ .01; dims = 2))
rotation = PointCloudRegistration.rand_rotation(Random.default_rng(), X, X)
translation = ones(SVector{d, Float64})
Y = rotation' * (X[:, shuffle(axes(X, 2))[1:n ÷ 10]] .- translation)
al = AnalysisLogger([])
Logging.disable_logging(LogLevel(-2001))
T = with_logger(al) do
    register_no_correspondences(
        Y,
        X;
        scale = 1.0,
        axisalign = true,
        # iterations = 20,
        # annealing = 3,
        restarts = 10,
    )
end
Logging.disable_logging(Logging.Debug)

align_log_idx = findfirst(log -> log.id == :align, al.logs)
align_rotation = if align_log_idx isa Int
    al.logs[align_log_idx].rotation
else
    one(PointCloudRegistration.rotation_type(X, X))
end
mm_logs = filter(log -> log.id == :mm, al.logs)
fig = Figure()
slider = Slider(fig[2, 1]; range = eachindex(mm_logs))
curr_log = @lift mm_logs[$(slider.value)]
curr_rotation = @lift ($curr_log).rotation
curr_translation = @lift ($curr_log).translation
curr_trY = @lift $curr_rotation * Y .+ $curr_translation
curr_target_kde = @lift ($curr_log).target_kde
curr_grid = @lift ($curr_log).grid
curr_xdomain = @lift PointCloudRegistration.domains($curr_grid)[1]
curr_ydomain = @lift PointCloudRegistration.domains($curr_grid)[2]
title = @lift string(
    "restart = ",
    ($curr_log).restart,
    " ; iter = ",
    ($curr_log).iter,
)
ax = Axis(fig[1, 1]; autolimitaspect = 1, title)
my_colormap = Makie.PlotUtils.cgrad([Makie.Colors.alphacolor(colorant"tomato", 0.), colorant"tomato"])
heatmap!(ax, curr_xdomain, curr_ydomain, curr_target_kde, colormap=my_colormap)
# scatter!(ax, align_rotation * X; color = :tomato)
scatter!(ax, curr_trY; color = :teal)




shuffled_proportions = 0.0:0.001:1
samples = 2
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
            scale = 1.0,
            iterations = 10,
            annealing = 10,
            restarts = 10,
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
