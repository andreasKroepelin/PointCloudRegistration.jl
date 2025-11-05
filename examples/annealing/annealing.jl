using Revise
using PointCloudRegistration
using Rotations
using GLMakie
using Format
using Combinatorics
using Random
using Statistics

X, Y = PointCloudRegistration.Assets.load_1ake_A_4ake_A()

avg_nn_dist = PointCloudRegistration.avg_nn_dist(X)

minscale = avg_nn_dist

let
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    markersize = 0.5avg_nn_dist
    meshscatter!(ax, X.points; markersize)
    meshscatter!(ax, Y.points; markersize)
    fig
end

@time data = let
    ref_al = prepare_target_kc(X; scale = minscale).annealing_levels[end]
    scale_reservoir = range(sqrt(maximum(X.coveigvals)), minscale; length = 6)
    scale_powerset = collect(powerset(scale_reservoir, 1))
    kcs = fill(0.0, length(scale_powerset))
    Threads.@threads for i in eachindex(scale_powerset, kcs)
        scale = scale_powerset[i]
        pX = prepare_target_kc(X; scale)
        local_kcs = Float32[]
        for seed in 1:10
            T = register_kc(
                Y,
                pX;
                restarts = RandomRestarts(100, Xoshiro(seed)),
                iterations = 100,
            )
            kc = PointCloudRegistration.eval_kernel_correlation(ref_al, Y, T)
            push!(local_kcs, kc)
        end
        kcs[i] = quantile(local_kcs, .1)
    end
    (; scale_powerset, kcs)
end

let
    fig = Figure()
    ax_s = Axis(
        fig[1, 1:2];
        dim1_conversion = Makie.CategoricalConversion()#=yscale = log10=#
    )
    ax_a = Axis(fig[2, 1:2]; dim1_conversion = Makie.CategoricalConversion())
    linkxaxes!(ax_s, ax_a)
    ylims!(ax_a, 0, nothing)
    sl_min = Slider(fig[3, 1]; range = eachindex(data.minscales))
    sl_max = Slider(fig[4, 1]; range = eachindex(data.maxscales))
    Label(fig[3, 2], @lift(format("{:.1f}", data.minscales[$(sl_min.value)];)))
    Label(fig[4, 2], @lift(format("{:.1f}", data.maxscales[$(sl_max.value)];)))
    # j = 1
    # k = 5
    for i in eachindex(data.numscales)
        scale_obs = map(sl_min.value, sl_max.value) do j, k
            logrange(
                data.maxscales[k],
                data.minscales[j];
                length = data.numscales[i],
            )
        end
        is_obs = map(scale -> fill(i, length(scale)), scale_obs)
        angle_obs = map(sl_min.value, sl_max.value) do j, k
            Point(i, data.angles[i, j, k])
        end
        scatterlines!(ax_s, is_obs, scale_obs)
        scatter!(ax_a, angle_obs)
    end
    fig
end

let
    fig = Figure()
    ax_s = Axis(fig[1, 1:2], ylabel = "annealing scales")
    ax_kc = Axis(fig[2, 1:2], ylabel = "final KC")
    linkxaxes!(ax_s, ax_kc)
    # ylims!(ax_kc, 0, nothing)
    hidexdecorations!(ax_s)
    hidexdecorations!(ax_kc)
    sorted_idcs = sortperm(data.kcs; rev = true)
    # sorted_idcs = sort(
    #     vec(CartesianIndices(data.kcs));
    #     by = ci -> data.kcs[ci],
    #     rev = true,
    # )
    plot_idx = 0
    for i in sorted_idcs
        scale = data.scale_powerset[i]
        plot_idx += 1
        x = fill(plot_idx, length(scale))
        scatterlines!(ax_s, x, scale;#=color = :blue=#)
        scatter!(ax_kc, Point(plot_idx, data.kcs[i]);#=color = :blue=#)
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
