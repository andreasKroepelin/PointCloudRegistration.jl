using Revise
using BenchmarkTools
using Random
using StatsBase
using LinearAlgebra
using GLMakie
using GLMakie.Makie.Unitful
using DataFrames
using ProgressMeter
using DelimitedFiles
using PointCloudRegistration
using PointCloudRegistration.Rotations
using PointCloudRegistration.StaticArrays
import PointCloudRegistration as PCReg

# TODO:
# 1. clean up deps
# 2. use 1ake
# 3. "phase plot", i.e. x: norm of translation, y: angle of rotation, then
#    trajectory of increasingly wrong (shuffled) correspondences
# 4. compare RMSD, GMc, KC

pc_1ake = PointCloud(readdlm("../../assets/ake/1ake.csv", ','))

data = let
    # rows = @NamedTuple{angle::typeof(1.0u"deg"), norm::Float64, numwrong::Int, method::String}[]
    rows = @NamedTuple{
        angle::Float64,
        norm::Float64,
        numwrong::Int,
        method::String,
    }[]
    scale = sqrt.(PCReg.annealing_plan(pc_1ake, TargetScales(5)))
    prepd_1ake = prepare_target_kc(pc_1ake; scale)
    @showprogress for num_wrong_idcs in 0:length(pc_1ake.points)
        for _ in 1:100
            wrong_idcs = sample(
                eachindex(pc_1ake.points),
                num_wrong_idcs;
                replace = false,
            )
            idcs = collect(eachindex(pc_1ake.points))
            shuffle!(view(idcs, wrong_idcs))
            pc = pc_1ake[idcs]
            T_rmsd = register_rmsd(pc, pc_1ake)
            T_gmc = register_gmc(pc, pc_1ake; scale)
            T_kc = register_kc(pc, prepd_1ake)
            for (method, T) in (("rmsd", T_rmsd), ("gmc", T_gmc), ("kc", T_kc))
                push!(
                    rows,
                    (;
                        angle = T.linear |>
                                RotMatrix |>
                                rotation_angle |>
                                rad2deg, # |> Base.Fix2(*, 1u"deg"),
                        norm = norm(T.translation),
                        numwrong = num_wrong_idcs,
                        method,
                    ),
                )
            end
        end
    end
    DataFrame(rows)
end

grouped = groupby(data, [:numwrong, :method])
quartile1(x) = quantile(x, 1//4)
quartile3(x) = quantile(x, 3//4)
stats = combine(
    grouped,
    :angle => mean,
    :angle => std,
    :norm => mean,
    :norm => std,
    :angle => quartile1,
    :angle => quartile3,
)

let
    fig = Figure()
    ax = Axis(fig[1, 1]; xscale = log10, yscale = log10)
    for method in ("rmsd", "gmc", "kc")
        selection = subset(stats, :method => (m -> m .== method))
        lines!(
            ax,
            selection.angle_mean,
            selection.norm_mean;
            #=color = selection.numwrong, colormap = :blues=#label = method,
        )
    end
    axislegend(ax)
    fig
end

let
    fig = Figure()
    ax = Axis(
        fig[1, 1];
        yscale = identity,#=yticks = 0:15:180 =##= yticks = [0.1u"deg", 1.0u"deg", 10.0u"deg", 100.0u"deg"]=#
    )
    for method in ("rmsd", "gmc", "kc")
        selection = subset(stats, :method => (m -> m .== method))
        band!(
            ax,
            selection.numwrong,
            selection.angle_quartile1,
            selection.angle_quartile3;
            label = method,
        )
        # scatterlines!(ax, selection.numwrong, selection.angle_mean, label = method)
    end
    axislegend(ax)
    # stats_low = subset(stats, :numwrong => (n -> n .< 195))
    # data_low = subset(data, :numwrong => (n -> n .< 195))
    # scatter!(ax, data_low.numwrong, data_low.angle)
    fig
end

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
    register_gmc(Y, X; scale = 1.0, iterations = 20, annealing = 3, restarts = 10)
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
