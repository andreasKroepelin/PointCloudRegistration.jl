using Revise
using PointCloudRegistration
using Rotations
using GLMakie
using Format
using DelimitedFiles
using Random

ake_path = pkgdir(PointCloudRegistration, "assets", "ake")
pc_1ake = PointCloud(readdlm(joinpath(ake_path, "1ake.csv"), ','))
pc_4ake = PointCloud(readdlm(joinpath(ake_path, "4ake.csv"), ','))

avg_nn_dist = PointCloudRegistration.avg_nn_dist(pc_1ake)

minscale = avg_nn_dist

let
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    markersize = .5avg_nn_dist
    meshscatter!(ax, Tgmc(pc_4ake).points; markersize)
    meshscatter!(ax, Topt(pc_4ake).points; markersize)
    fig
end

@time data = let
    maxscales = range(minscale, sqrt(maximum(pc_1ake.coveigvals)); length = 10)
    numscales = 2:20
    kcs = fill(0.0, length(numscales), length(maxscales))
    Threads.@threads for j in eachindex(maxscales)
        maxscale = maxscales[j]
        for (i, numscale) in enumerate(numscales)
            scale = logrange(maxscale, minscale; length = numscale)
            prepd_target = prepare_target_kc(pc_1ake; scale)
            T = register_kc(pc_4ake, prepd_target; restarts = RandomRestarts(100, Xoshiro(1)), iterations = 100)
            kc = PointCloudRegistration.eval_kernel_correlation(prepd_target.annealing_levels[end], pc_4ake, T)
            kcs[i, j] = kc
        end
    end
    (; maxscales, numscales, kcs)
end

let
    fig = Figure()
    ax_s = Axis(fig[1, 1:2]; dim1_conversion = Makie.CategoricalConversion() #=yscale = log10=#)
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
            logrange(data.maxscales[k], data.minscales[j]; length = data.numscales[i])
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
    ax_s = Axis(fig[1, 1:2] #=yscale = log10=#)
    ax_kc = Axis(fig[2, 1:2])
    linkxaxes!(ax_s, ax_kc)
    # ylims!(ax_kc, 0, nothing)
    hidexdecorations!(ax_s)
    hidexdecorations!(ax_kc)
    sorted_idcs = sort(vec(CartesianIndices(data.kcs)); by = ci -> data.kcs[ci], rev = true)
    plot_idx = 0
    for ci in sorted_idcs
        i, j = Tuple(ci)
        maxscale = data.maxscales[j]
        numscale = data.numscales[i]
        scale = logrange(maxscale, minscale; length = numscale)
        plot_idx += 1
        x = fill(plot_idx, length(scale))
        scatterlines!(ax_s, x, scale; #=color = :blue=#)
        scatter!(ax_kc, Point(plot_idx, data.kcs[i, j]); #=color = :blue=#)
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
