using Revise
using PointCloudRegistration
using Rotations
using GLMakie
using Format
using DelimitedFiles

ake_path = pkgdir(PointCloudRegistration, "assets", "ake")
pc_1ake = PointCloud(readdlm(joinpath(ake_path, "1ake.csv"), ','))
pc_4ake = PointCloud(readdlm(joinpath(ake_path, "4ake.csv"), ','))

avg_nn_dist = PointCloudRegistration.avg_nn_dist(pc_1ake)

# Topt = register_kc(pc_4ake, pc_1ake; scale = DownTo(.1avg_nn_dist, 10), restarts = RandomRestarts(100))
Topt = register_kc(pc_4ake, pc_1ake; restarts = RandomRestarts(100))
invTopt = inv(Topt)

Tgmc = register_gmc(pc_4ake, pc_1ake; scale = DownTo(.1avg_nn_dist, 10), restarts = RandomRestarts(100))
invTgmc = inv(Tgmc)

let
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    markersize = .5avg_nn_dist
    meshscatter!(ax, Tgmc(pc_4ake).points; markersize)
    meshscatter!(ax, Topt(pc_4ake).points; markersize)
    fig
end

@time data = let
    maxscales = [1., 3., 5., 10., sqrt(maximum(pc_1ake.coveigvals)), 20.]
    minscales = [1., 1.5, 2., 3., avg_nn_dist, 5., 10.]
    numscales = [2, 3, 4, 5, 10 ]
    Ts = fill(PointCloudRegistration.simple_transformation(pc_4ake, pc_1ake), length(numscales), length(minscales), length(maxscales))
    Threads.@threads for k in eachindex(maxscales)
        maxscale = maxscales[k]
        for j in eachindex(minscales)
            minscale = minscales[j]
            minscale <= maxscale || continue
            for (i, numscale) in enumerate(numscales)
                scale = logrange(maxscale, minscale; length = numscale)
                prepd_target = try
                    prepare_target_kc(pc_1ake; scale)
                catch e
                    if e isa OutOfMemoryError
                        @info "OOM!" minscale maxscale numscale
                    end
                end
                T = register_kc(pc_4ake, prepd_target; restarts = RandomRestarts(100))
                Ts[i, j, k] = T
                # Tdiff = T ∘ invTopt
                # angle = rotation_angle(RotMatrix(Tdiff.linear)) |> rad2deg
                # angles[i, j, k] = angle
                # @info "result" maxscale minscale numscale angle
            end
        end
    end
    (; maxscales, minscales, numscales, Ts)
end

angles_to_gmc = map(data.Ts) do T
    (T ∘ invTgmc).linear |> RotMatrix |> rotation_angle |> rad2deg
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
    ax_a = Axis(fig[2, 1:2])
    linkxaxes!(ax_s, ax_a)
    ylims!(ax_a, 0, nothing)
    hidexdecorations!(ax_s)
    hidexdecorations!(ax_a)
    sorted_idcs = sort(vec(CartesianIndices(angles_to_gmc)); by = ci -> angles_to_gmc[ci])
    plot_idx = 0
    for ci in sorted_idcs
        i, j, k = Tuple(ci)
        maxscale = data.maxscales[k]
        minscale = data.minscales[j]
        numscale = data.numscales[i]
        minscale > maxscale && continue
        scale = logrange(maxscale, minscale; length = numscale)
        plot_idx += 1
        x = fill(plot_idx, length(scale))
        scatterlines!(ax_s, x, scale; color = :blue)
        scatter!(ax_a, Point(plot_idx, angles_to_gmc[i, j, k]); color = :blue)
    end
    fig
end
