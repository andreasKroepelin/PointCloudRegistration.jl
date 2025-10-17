module PlotReassembly

using GLMakie
using Rotations
using HDF5
using PointCloudRegistration
using Optim
using LogExpFunctions
using RollingFunctions
using PrecompileTools: @compile_workload

function (@main)(args)
    command, files... = args
    results = read_result.(files)
    sort!(results; by = res -> res.scales[1])

    if command == "success"
        fig = plot_success(results)
    elseif command == "slices"
        fig = plot_slices(results)
    elseif command == "inits"
        fig = plot_inits(results)
    else
        @error "unknown command" command
        return 1
    end
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

struct Loss{Y}
    ys::Y
end

function (loss::Loss)(params)
    k, x0 = params
    sum(enumerate(loss.ys)) do (x, y)
        (y - logistic(k * (x - x0)))^2
    end
end

function find_step(ys)
    opt = optimize(Loss(ys), [1.0, length(ys) / 2])
    step = Optim.minimizer(opt)[2]
    clamp(round(Int, step), eachindex(ys))
end

function boundary_line(result)
    @info "computing boundary" result.scales[1]
    coords = stack(enumerate(eachcol(result.successes))) do (i, ss)
        ys = map(ss) do s
            if isnan(s)
                1.0
            else
                1.0 - s
            end
        end
        step = find_step(ys)
        [result.threshold_range[i], result.threshold_range[step]]
    end
    runmean(coords[2, :], 10), coords[1, :]
end

function scale_ratio_text(result)
    scale_ratio = result.scales[1] / result.resolution
    "σ = $(chopsuffix(Makie.Format.format("{:.1f}", scale_ratio), ".0"))r"
end

function plot_success(results)
    fig = Figure()
    ax = Axis(
        fig[1, 2];
        aspect = DataAspect(),
        limits = (0, 1, 0, 1),
        xlabel = "begin overlap",
        ylabel = "end overlap",
        xaxisposition = :top,
        xgridvisible = false,
        ygridvisible = false,
    )
    hidespines!(ax, :b, :r)
    cp = contourf!(
        ax,
        results[1].threshold_range,
        results[1].threshold_range,
        results[1].overlaps;
        levels = 0.0:0.2:1.0,
        colormap = :Blues,
    )
    for (i, result) in enumerate(results)
        lines!(
            ax,
            boundary_line(result)...;
            color = cgrad(:sun, length(results); categorical = true)[i],
            linewidth = 4,
            joinstyle = :round,
            label = scale_ratio_text(result),
        )
    end
    axislegend(
        ax,
        "success/failure boundary\nfor initial annealing scale";
        position = :rb,
        patchsize = (30, 20),
    )
    Colorbar(
        fig[1, 1],
        cp;
        ticks = 0:0.2:1,
        tickformat = xs -> [Makie.Format.format("{:.0%}", x) for x in xs],
    )
    Label(
        fig[1, 0],
        "proportion of common points";
        rotation = deg2rad(90),
        tellheight = false,
    )

    fig
end

function plot_slices(results)
    result = results[1]
    (; points, projected, resolution, normal) = result
    fig = Figure()
    lower_color = colorant"#239dad"
    upper_color = colorant"#85144b"
    ax = Axis(
        fig[1:3, 1];
        aspect = DataAspect(),
        limits = (0, 1, 0, 1),
        xlabel = "begin overlap",
        ylabel = "end overlap",
        xaxisposition = :top,
        xgridvisible = false,
        ygridvisible = false,
        xlabelcolor = lower_color,
        xtickcolor = lower_color,
        xticklabelcolor = lower_color,
        topspinecolor = lower_color,
        ylabelcolor = upper_color,
        ytickcolor = upper_color,
        yticklabelcolor = upper_color,
        leftspinecolor = upper_color,
    )
    hidespines!(ax, :b, :r)
    for interaction in interactions(ax)
        deregister_interaction!(ax, interaction[1])
    end

    lo, hi = extrema(projected)
    positions = [(0.1, 0.7), (0.2, 0.3), (0.7, 0.8)]
    textlabel!(ax, positions; text = string.('A':'C'), fontsize = 15)
    limits = ((-0.1, 0.1), (-0.1, 0.1), (0, 0.2))
    corners =
        [[0, -0.08, 0.02], [0, -0.08, 0.18], [0, 0.08, 0.18], [0, 0.08, 0.02]]
    rotation = rotation_between([1, 0, 0], result.normal)
    for (position, i, label) in zip(positions, 1:3, 'A':'C')
        lines!(
            ax,
            [(0, position[2]), position];
            color = upper_color,
            linewidth = 2,
        )
        lines!(
            ax,
            [(position[1], 1), position];
            color = lower_color,
            linewidth = 2,
        )
        lt, ut = lo .+ position .* (hi - lo)
        Label(
            fig[i, 2],
            string(label);
            tellheight = false,
            tellwidth = false,
            valign = :center,
            halign = :left,
        )
        ax3 = Axis3(
            fig[i, 2];
            aspect = :data,
            protrusions = 0,
            limits,
            viewmode = :fitzoom,
            azimuth = 1.45pi,
            elevation = 0.01pi,
        )
        hidedecorations!(ax3)
        hidespines!(ax3)
        vertices1 = [rotation * corner + lt * normal for corner in corners]
        vertices2 = [rotation * corner + ut * normal for corner in corners]
        meshscatter!(ax3, points; color = :lightgray, markersize = resolution)
        mesh!(ax3, vertices1, [1 2 3; 3 4 1]; color = (lower_color, 0.7))
        mesh!(ax3, vertices2, [1 2 3; 3 4 1]; color = (upper_color, 0.7))
    end

    rowgap!(fig.layout, 0)

    fig
end

function plot_inits(results)
    fig = Figure()
    limits = ((-0.1, 0.1), (-0.1, 0.1), (0, 0.2))
    for (i, result) in enumerate(results)
        (; points, weights, scales)=result
        ax = Axis3(
            fig[1, i];
            aspect = :data,
            protrusions = (0, 0, 0, 20),
            limits,
            title = scale_ratio_text(result),
        )
        prepd =
            prepare_target_kc(PointCloud(points, weights); scale = scales[1])
        hidedecorations!(ax)
        hidespines!(ax)
        al = prepd.annealing_levels[1]
        domains = map(
            d -> (first(d), last(d)),
            PointCloudRegistration.domains(al.grid),
        )
        cmap = range(colorant"#0074d9", colorant"#0074d9")
        kde = al.convd_weights_target
        maxdensity = maximum(kde)
        volume!(
            ax,
            domains...,
            kde;
            algorithm = :iso,
            isovalue = 0.6 * maxdensity,
            isorange = 0.2 * maxdensity,
            colormap = cmap,
        )
    end
    fig
end

function plot_result(result, fig, i)
    (;
        successes,
        overlaps,
        threshold_range,
        points,
        weights,
        projected,
        scales,
        resolution,
        normal,
    )=result

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

end

#=
@compile_workload begin
    example_data = joinpath(pkgdir(PlotReassembly),"..", "result-2025-10-15T17:18:10.180.h5")
    plot_result(example_data)
    nothing
end
=#

end # module PlotReassembly
