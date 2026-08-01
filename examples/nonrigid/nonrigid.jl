using Revise
using PointCloudRegistration
# using ExactOptimalTransport
# using Tulip
using Mooncake
using LinearAlgebra
using Distances
using GLMakie

# X, Y = PointCloudRegistration.Assets.load_1ake_A_4ake_A()
# X, Y = PointCloudRegistration.Assets.load_1su4_A_1iwo_A()
Y, X = PointCloudRegistration.Assets.load_cats()
# Y, X = PointCloudRegistration.Assets.load_1ih7_A_1ig9_A()
# Y, X = PointCloudRegistration.Assets.load_1q9x_B_1q9y_A()
# X = PointCloud(collect(map(x -> Float64.(x), X.points)))
# Y = PointCloud(collect(map(x -> Float64.(x), Y.points)))
T_rigid = rigid_registration(Y, X)
Y = T_rigid(Y)

function report_iteration(; iter, new_source_points, sqsigma, kwargs...)
    pseudo_sqrt(x) = x < 0 ? NaN : sqrt(x)
    sigma = pseudo_sqrt(sqsigma)
    # iter % 100 == 0 && @info "iteration" iter sigma
    push!(new_source_points_hist, copy(new_source_points))
    push!(sigma_hist, sigma)
end

new_source_points_hist = []
sigma_hist = []
prepd_Y = prepare_source_distancepreserving(Y; max_edge_length = 10.)
dpr = nonrigid_registration(Y, X, DistancePreserving(; max_edge_length = 10, iterations = 200_000, report_iteration, sensitivity = 1.8, rel_deviation = 5e-2); source_preparation = prepd_Y)
dY = dpr(Y)

let
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    # ax = Axis(fig[1, 1]; autolimitaspect = 1)
    plot!(ax, dY; color = :lightgray)
    nodes = [(dY.points[j1], dY.points[j2]) for (j1, j2) in prepd_Y.neighbor_graph.edges]
    distances = map(splat(euclidean), nodes)
    abs_log_ratios = distances ./ prepd_Y.neighbor_graph.distances .|> log .|> abs
    colormap = range(
        Makie.to_color((:aqua, 0.01)),
        Makie.to_color((:tomato, 1.0)),
    )
    # lsplt = linesegments!(ax, nodes; color = exp.(abs_log_ratios) .- 1, colormap, linewidth = abs_log_ratios .* 20)
    # Colorbar(fig[1, 2], lsplt)
    lsplt = linesegments!(ax, nodes; color = :gray)
    fig
end

let
    fig = Figure()
    ax1 = Axis3(fig[1, 1]; aspect = :data)
    # ax1 = Axis(fig[1, 1]; autolimitaspect = 1)
    ax2 = Axis(fig[1, 2];)
    sl = Slider(fig[2, 1]; range = eachindex(new_source_points_hist))
    lines!(ax2, sigma_hist)
    iY = @lift PointCloud(new_source_points_hist[$(sl.value)], Y.weights)
    io = @lift [Point2($(sl.value), sigma_hist[$(sl.value)])]
    # plot!(ax1, Y; sizefactor = 2)
    plot!(ax1, X; sizefactor = 2, color = (:gray, .3))
    plot!(ax1, iY; sizefactor = 2, color = (:teal, .3))
    nodes = @lift [(($iY).points[j1], ($iY).points[j2]) for (j1, j2) in prepd_Y.neighbor_graph.edges]
    abs_log_ratios = @lift map(splat(euclidean), $nodes) ./ prepd_Y.neighbor_graph.distances .|> log .|> abs
    color = @lift exp.($abs_log_ratios) .- 1
    linewidth = @lift $abs_log_ratios .* 10
    colormap = range(
        Makie.to_color((:aqua, 0.01)),
        Makie.to_color((:tomato, 1.0)),
    )
    lsplt = linesegments!(ax1, nodes; color, colormap, colorrange = (0, .5), linewidth)
    Colorbar(fig[3, 1], lsplt; vertical = false)
    scatter!(ax2, io; markersize = 10)
    fig
end

trials = [
    let
        @info "trial" sensitivity log10rel_deviation
        displacement = nonrigid_registration(
            Y,
            X,
            DistancePreserving(;
                max_edge_length = 50,
                iterations = 200_000,
                sensitivity,
                rel_deviation = 10.0^log10rel_deviation,
            )
        )
        (; sensitivity, log10rel_deviation, displacement)
    end
    for sensitivity in 1.0:0.2:2.0, log10rel_deviation in -5:1:-2
]

let
    fig = Figure()
    axs = [Axis(fig[Tuple(ci)...]; yreversed = true, aspect = DataAspect()) for ci in CartesianIndices(trials)]
    hidedecorations!.(axs)
    for (ax, trial) in zip(axs, trials)
        arrows2d!(ax, trial.displacement)
    end
    for (i, row) in enumerate(eachrow(trials))
        sensitivity = [trial.sensitivity for trial in row] |> unique |> only
        Label(fig[i, 0], string(sensitivity); tellheight = false)
    end
    for (i, col) in enumerate(eachcol(trials))
        log10rel_deviation = [trial.log10rel_deviation for trial in col] |> unique |> only
        Label(fig[0, i], "10^$log10rel_deviation"; tellwidth = false)
    end
    fig
end



registrations = (
    kc_springs = nonrigid_kc_springs(Y, X; stiffness = 3e-2, scale = 10., max_spring_length = 30.),
    # divfree = nonrigid_divfree(Y, X; scale = 3.0, degree = 3),
    # sinkhorn = nonrigid_sinkhorn(Y, X),
    # cpd = nonrigid_cpd(Y, X; corr_length = 20., expected_displacement = 20.),
)

#=
function report_iteration(; iter, gradient, displaced_source_points, kwargs...)
    mod(iter, 100) == 0 || return
    @info "iteration" iter norm(gradient)
    push!(dsps, copy(displaced_source_points))
end

dsps = []
registration = nonrigid_kc_springs(Y, X; stiffness = 1//200, iterations = 10_000, report_iteration);
Y_disp = apply_displacements(Y, displacements(registration))

let
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    src_plt = plot!(ax, Y; label = "source")
    trg_plt = plot!(ax, X; label = "target")
    # plot!(ax, Y_disp; label = "displaced source")
    dsrc_plt = plot!(ax, Y; label = "displaced source")
    t = 0
    on(events(fig).tick) do tick
        @show tick
        t = mod(ceil(Int, 10tick.time), eachindex(dsps))
        ax.title[] = string(t)
        Makie.update!(dsrc_plt; arg1 = PointCloud(dsps[t]))
    end
    axislegend(ax)
    fig
end
=#


heatmap(correspondences(registrations.bcpd))

Ys_disp = map(registrations) do registration
    registration(Y)
end

let
    key = :kc_springs
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    # src_plt = plot!(ax, Y; label = "source")
    # trg_plt = plot!(ax, X; label = "target")
    plot!(ax, Ys_disp[key]; label = "displaced source")
    # arrows3d!(ax, Y.points, Ys_disp[key].points .- Y.points; label = "displacement")
    axislegend(ax)
    fig
end
