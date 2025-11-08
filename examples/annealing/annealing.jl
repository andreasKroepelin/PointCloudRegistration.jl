using Revise
using PointCloudRegistration
using Rotations
using GLMakie
using Format
using Combinatorics
using Random
using LinearAlgebra
using Statistics

# X, Y = PointCloudRegistration.Assets.load_1su4_A_1iwo_A()
X, Y = PointCloudRegistration.Assets.load_1ake_A_4ake_A()
avg_nn_dist = PointCloudRegistration.avg_nn_dist(X)
minscale = .5 * avg_nn_dist

let
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    markersize = 0.5avg_nn_dist
    meshscatter!(ax, X.points; markersize)
    meshscatter!(ax, Y.points; markersize)
    fig
end

struct RestartCollector{T}
    transformations::Vector{T}
end
RestartCollector() = RestartCollector([])
function (rc::RestartCollector)(; transformation, kwargs...)
    push!(rc.transformations, transformation)
end

@time data = let
    nrestarts = 1_000
    iterations = 100
    seed = 2
    ref_al = prepare_target_kc(X; scale = minscale).annealing_levels[end]
    scale_reservoir = range(sqrt(maximum(X.coveigvals)), minscale; length = 6)[begin:end - 1]
    scale_powerset = [[ss; minscale] for ss in powerset(scale_reservoir)]
    successes = fill(0.0, length(scale_powerset))
    successes_smm = fill(0.0, length(scale_powerset))
    ts = Task[]
    for scale in scale_powerset
        t = Threads.@spawn begin
            my_rows = []
            pX = prepare_target_kc(X; scale)
            for do_smm in (false, true)
                restart_collector = RestartCollector()
                smm = if do_smm
                    SomePoints(100, Xoshiro(-seed))
                else
                    AllPoints()
                end
                T = register_kc(
                    Y,
                    pX;
                    restarts = RandomRestarts(nrestarts, Xoshiro(seed)),
                    iterations,
                    stochastic_majorization_minimization = smm,
                    report_restart = restart_collector,
                )
                kc = PointCloudRegistration.eval_kernel_correlation(ref_al, Y, T)
                push!(my_rows, (;T, smm = do_smm, scale, kc, restartTs = restart_collector.transformations))
            end
            my_rows
        end
        push!(ts, t)
    end
    mapreduce(fetch, vcat, ts)
end;

bestT = argmax(r -> r.kc, data).T
invbestT = inv(bestT)

let
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    markersize = 0.5avg_nn_dist
    meshscatter!(ax, X.points; markersize)
    meshscatter!(ax, bestT(Y).points; markersize)
    fig
end

success_rates = map(data) do row
    success_count = count(row.restartTs) do T
        diffT = T ∘ invbestT
        angle = diffT.linear |> RotMatrix |> rotation_angle |> rad2deg
        nrm = norm(diffT.translation)
        angle < 5 && nrm < 1 * avg_nn_dist
    end
    success_count / length(row.restartTs)
end

# let
#     fig = Figure()
#     ax_s = Axis(
#         fig[1, 1:2];
#         dim1_conversion = Makie.CategoricalConversion()#=yscale = log10=#
#     )
#     ax_a = Axis(fig[2, 1:2]; dim1_conversion = Makie.CategoricalConversion())
#     linkxaxes!(ax_s, ax_a)
#     ylims!(ax_a, 0, nothing)
#     sl_min = Slider(fig[3, 1]; range = eachindex(data.minscales))
#     sl_max = Slider(fig[4, 1]; range = eachindex(data.maxscales))
#     Label(fig[3, 2], @lift(format("{:.1f}", data.minscales[$(sl_min.value)];)))
#     Label(fig[4, 2], @lift(format("{:.1f}", data.maxscales[$(sl_max.value)];)))
#     # j = 1
#     # k = 5
#     for i in eachindex(data.numscales)
#         scale_obs = map(sl_min.value, sl_max.value) do j, k
#             logrange(
#                 data.maxscales[k],
#                 data.minscales[j];
#                 length = data.numscales[i],
#             )
#         end
#         is_obs = map(scale -> fill(i, length(scale)), scale_obs)
#         angle_obs = map(sl_min.value, sl_max.value) do j, k
#             Point(i, data.angles[i, j, k])
#         end
#         scatterlines!(ax_s, is_obs, scale_obs)
#         scatter!(ax_a, angle_obs)
#     end
#     fig
# end

let
    fig = Figure()
    ax_sc = Axis(fig[1, 1], ylabel = "annealing scales")
    ax_su = Axis(fig[2, 1], ylabel = "success rate")
    # ax_kc = Axis(fig[3, 1], ylabel = "best KC found")
    linkxaxes!(ax_sc, ax_su)
    # linkxaxes!(ax_sc, ax_kc)
    hidexdecorations!(ax_sc)
    hidexdecorations!(ax_su)
    # hidexdecorations!(ax_kc)
    sorted_idcs = sortperm(success_rates; rev = true)
    plot_idx = 0
    for i in sorted_idcs
        scale = data[i].scale ./ minscale
        plot_idx += 1
        x = fill(plot_idx, length(scale))
        color = if data[i].smm; :cornflowerblue; else :tomato; end
        scatterlines!(ax_sc, x, scale; color, linewidth = 3, markersize = 10)
        barplot!(ax_su, Point(plot_idx, success_rates[i]); color)
        # barplot!(ax_kc, Point(plot_idx, data[i].kc); color)
    end
    fig
end

let
    fig = Figure()
    ax = Axis(fig[1, 1])
    for ci in CartesianIndices(data.kcs)
        (i, j) = Tuple(ci)
        maxscale = data.maxscales[j]
        numscale = data.numscales[i]
        scale = logrange(maxscale, minscale; length = numscale)
        kc = data.kcs[i, j]
        scatter!(ax, Point(length(unique(scale)), kc))
    end
    fig
end
