using Revise
using PointCloudRegistration
using Rotations
using GLMakie
using Format
using Combinatorics
using Random
using LinearAlgebra
using Statistics

# X, Y = PointCloudRegistration.Assets.load_1ysy_A_2ahm_D()
# X, Y = PointCloudRegistration.Assets.load_1su4_A_1iwo_A()
# X, Y = PointCloudRegistration.Assets.load_1ake_A_4ake_A()
X, Y = PointCloudRegistration.Assets.load_1ih7_A_1ig9_A()
# X, Y = PointCloudRegistration.Assets.load_1q9x_B_1q9y_A()
avg_nn_dist = PointCloudRegistration.avg_nn_dist(X)
minscale = avg_nn_dist / 2

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
    scale_reservoir =
        range(sqrt(maximum(X.coveigvals)), minscale; length = 5)[begin:(end - 1)]
    # range(20minscale, minscale; length = 5)[begin:(end - 1)]
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
                    Smm(100, Xoshiro(-seed))
                else
                    NoSmm()
                end
                T = register_kc(
                    Y,
                    pX;
                    restarts = RandomRestarts(nrestarts, Xoshiro(seed)),
                    iterations,
                    smm,
                    report_restart = restart_collector,
                )
                kc = PointCloudRegistration.eval_kernel_correlation(
                    ref_al,
                    Y,
                    T,
                )
                push!(
                    my_rows,
                    (;
                        T,
                        smm = do_smm,
                        scale,
                        kc,
                        restartTs = restart_collector.transformations,
                    ),
                )
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

let
    smm_color = colorant"#0074D9"
    no_smm_color = colorant"#01FF70"
    fig = Figure()
    ax_sc = Axis(fig[1, 1]; ylabel = "annealing scales")
    ax_su = Axis(fig[2, 1]; ylabel = "number of restarts\nfor 1 % failure rate")
    Legend(
        fig[2, 1],
        [
            PolyElement(; color = smm_color, strokewidth = 0),
            PolyElement(; color = no_smm_color, strokewidth = 0),
        ],
        ["with SMM", "without SMM"];
        halign = :left,
        valign = :top,
        tellwidth = false,
        tellheight = false,
    )
    linkxaxes!(ax_sc, ax_su)
    hidexdecorations!(ax_sc)
    hidexdecorations!(ax_su)
    sorted_idcs = sortperm(success_rates; rev = true)
    stripes_lo = (1:2:length(data)) .- 0.5
    stripes_hi = (1:2:length(data)) .+ 0.5
    for ax in (ax_sc, ax_su)
        vspan!(
            ax,
            stripes_lo,
            stripes_hi;
            color = (:gray, 0.1),
            inspectable = false,
        )
    end
    plot_idx = 0
    for i in sorted_idcs
        scale = data[i].scale
        plot_idx += 1
        x = fill(plot_idx, length(scale))
        points = mapreduce(vcat, scale) do s
            [Point(plot_idx - 0.2, s), Point(plot_idx + 0.2, s)]
        end
        color = ifelse(data[i].smm, smm_color, no_smm_color)
        linesegments!(ax_sc, points; color, linewidth = 4, linecap = :round)
        num_trials = log(1 - success_rates[i], 0.01)
        if isfinite(num_trials)
            barplot!(ax_su, Point(plot_idx, num_trials); color)
        else
            text!(
                ax_su,
                Point(plot_idx, 0);
                text = "?",
                align = (:center, :bottom),
            )
        end
        # barplot!(ax_su, Point(plot_idx, success_rates[i]); color)
    end
    DataInspector(fig)
    fig
end
