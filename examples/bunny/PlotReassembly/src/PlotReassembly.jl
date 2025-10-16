module PlotReassembly

using GLMakie
using Rotations
using HDF5
using PointCloudRegistration
using PrecompileTools: @compile_workload

function (@main)(args)
    fig = Figure()
    results = read_result.(args)
    sort!(results, by = res -> res.scales[1])
    for (i, result) in enumerate(results)
        plot_result(result, fig, i)
    end
    Legend(
        fig[2, 1:length(args)],
        [
            [
                LineElement(; color = :gray, linestyle = :dash, linewidth = 3),
                MarkerElement(; color = :black, marker = '%', markersize = 15),
            ],
            PolyElement(; color = colorant"#7fdbff", strokewidth = 0),
            PolyElement(; color = colorant"#ff851b", strokewidth = 0),
        ],
        ["common points", "success", "failure"];
        patchsize = (30, 20),
        orientation = :horizontal,
        # tellwidth = false,
        # tellheight = false,
        # valign = :top,
        # halign = :left,
        # margin = (0, 0, 0, 0),
    )
    wait(display(fig))
end

function read_result(file)
    h5result = h5open(file, "r")
    (;
        successes = read(h5result, "successes"),
        overlaps = read(h5result, "overlaps"),
        threshold_range = read(h5result, "threshold_range"),
        points = read(h5result, "points")[[1, 3, 2], :],
        weights = read(h5result, "weights"),
        projected = read(h5result, "projected"),
        scales = read(h5result, "scales"),
        resolution = read_attribute(h5result, "resolution"),
        normal = read_attribute(h5result, "normal")[[1, 3, 2]],
    )
end

function plot_result(result, fig, i)
    (;successes,overlaps,threshold_range,points,weights,projected,scales,resolution,normal)=result

    # lo, hi = extrema(projected)
    # displayed_relative_thresholds = (0.2, 0.7)
    # displayed_thresholds = lo .+ displayed_relative_thresholds .* (hi - lo)

    ax = Axis(
        fig[1, i];
        aspect = DataAspect(),
        xlabel = "begin overlap",
        ylabel = "end overlap",
        xaxisposition = :top,
        xgridvisible = false,
        ygridvisible = false,
        limits = (0, 1, 0, 1),
        title = Makie.Format.format("σ = {:.1f}r", scales[1] / resolution),
    )
    hidespines!(ax, :b, :r)
    heatmap!(
        ax,
        threshold_range,
        threshold_range,
        successes;
        colormap = [colorant"#ff851b", colorant"#7fdbff"],
    )
    #=
    lower_color = colorant"#239dad"
    upper_color = colorant"#85144b"
    lines!(
        ax,
        [(displayed_relative_thresholds[1], 1), displayed_relative_thresholds];
        color = lower_color,
        linewidth = 3,
    )
    lines!(
        ax,
        [(0, displayed_relative_thresholds[2]), displayed_relative_thresholds];
        color = upper_color,
        linewidth = 3,
    )
    scatter!(
        ax,
        [displayed_relative_thresholds];
        marker = :xcross,
        color = :black,
        markersize = 15,
    )
    =#
    contour!(
        ax,
        threshold_range,
        threshold_range,
        overlaps;
        levels = 0.0:0.2:1.0,
        labels = true,
        labelcolor = :black,
        labelformatter = x -> Makie.Format.format("{:.0%}", x),
        labelsize = 15,
        color = :gray,
        linestyle = :dash,
        linewidth = 3,
    )
    #=
    =#

    limits = ((-0.1, 0.1), (-0.1, 0.1), (0, 0.2))
    #=
    ax3 = Axis3(
        fig[1, 3];
        aspect = :data,
        protrusions = 0,
        limits,
        title = "exemplary slicing",
    )
    hidedecorations!(ax3)
    hidespines!(ax3)
    corners = [[0, -0.1, 0.0], [0, -0.1, 0.2], [0, 0.1, 0.2], [0, 0.1, 0.0]]
    rotation = rotation_between([1, 0, 0], normal)
    vertices1 = [
        rotation * corner + displayed_thresholds[1] * normal for
        corner in corners
    ]
    vertices2 = [
        rotation * corner + displayed_thresholds[2] * normal for
        corner in corners
    ]
    meshscatter!(ax3, points; color = :lightgray, markersize = resolution)
    mesh!(ax3, vertices1, [1 2 3; 3 4 1]; color = (lower_color, 0.7))
    mesh!(ax3, vertices2, [1 2 3; 3 4 1]; color = (upper_color, 0.7))
    =#

    ax_kde = Axis3(
        fig[1, i];
        aspect = :data,
        width = Relative(0.5),
        height = Relative(0.5),
        halign = .9,
        valign = .1,
        protrusions = 0,
        limits,
    )
    prepd = prepare_target_kc(PointCloud(points, weights); scale = scales[1])
    hidedecorations!(ax_kde)
    hidespines!(ax_kde)
    al = prepd.annealing_levels[1]
    domains =
        map(d -> (first(d), last(d)), PointCloudRegistration.domains(al.grid))
    cmap = range(colorant"#0074d9", colorant"#0074d9")
    kde = collect(al.convd_weights_target)
    maxdensity = maximum(kde)
    # kde[kde .< .5 .* maxdensity] .= NaN
    volume!(ax_kde, domains..., kde; algorithm = :iso, isovalue = .6 * maxdensity, isorange = .2 * maxdensity, colormap = cmap)
end

#=
@compile_workload begin
    example_data = joinpath(pkgdir(PlotReassembly),"..", "result-2025-10-15T17:18:10.180.h5")
    plot_result(example_data)
    nothing
end
=#

end # module PlotReassembly
