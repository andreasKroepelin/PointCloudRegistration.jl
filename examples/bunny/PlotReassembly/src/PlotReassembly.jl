module PlotReassembly

using GLMakie
using Rotations
using HDF5
using PrecompileTools: @compile_workload

function @main(args)
    fig = plot_result(args[1])
    wait(display(fig))
end

function plot_result(file)
    h5result = h5open(file, "r")
    successes = read(h5result, "successes")
    overlaps = read(h5result, "overlaps")
    threshold_range = read(h5result, "threshold_range")
    points = read(h5result, "points")[[1, 3, 2], :]
    projected = read(h5result, "projected")
    resolution = read_attribute(h5result, "resolution")
    normal = read_attribute(h5result, "normal")[[1, 3, 2]]

    lo, hi = extrema(projected)
    displayed_relative_thresholds = (.2, .7)
    displayed_thresholds = lo .+ displayed_relative_thresholds .* (hi - lo)

    fig = Figure()
    ax = Axis(fig[1:2, 1:2]; aspect = DataAspect(), xlabel = "begin overlap", ylabel = "end overlap", xaxisposition = :top, xgridvisible = false, ygridvisible = false, limits = (0,1,0,1))
    hidespines!(ax, :b, :r)
    heatmap!(ax, threshold_range, threshold_range, successes; colormap = [colorant"#ff851b", colorant"#7fdbff"])
    lower_color = colorant"#239dad"
    upper_color = colorant"#85144b"
    lines!(ax, [(displayed_relative_thresholds[1], 1), displayed_relative_thresholds]; color =lower_color, linewidth=3)
    lines!(ax, [(0, displayed_relative_thresholds[2]), displayed_relative_thresholds]; color =upper_color, linewidth=3)
    scatter!(ax, [displayed_relative_thresholds]; marker = :xcross, color=:black, markersize = 15)
    contour!(ax, threshold_range, threshold_range, overlaps; levels = 0.:.2:1., labels = true, labelcolor = :black, labelformatter = x -> Makie.Format.format("{:.0%}", x), labelsize = 15, color = :gray, linestyle = :dash, linewidth = 3)
    Legend(fig[2, 2], [[LineElement(color = :gray, linestyle = :dash, linewidth = 3), MarkerElement(color = :black, marker = '%', markersize = 15)], PolyElement(color =  colorant"#7fdbff", strokewidth = 0), PolyElement(color =  colorant"#ff851b", strokewidth = 0), MarkerElement(color = :black, marker=:xcross, markersize=15)], ["common points", "success", "failure", "displayed slicing"], patchsize = (30, 20), tellwidth = false, tellheight=false, valign = :top, halign=:left, margin = (0, 0, 0, 0) )

    ax3 = Axis3(fig[1:2, 3:4]; aspect = :data, yreversed=false)
    hidedecorations!(ax3)
    hidespines!(ax3)
    corners = [
        [0, -.1, .0],
        [0, -.1, .2],
        [0, .1, .2],
        [0, .1, .0],
    ]
    rotation = rotation_between([1, 0, 0], normal)
    vertices1 = [rotation * corner + displayed_thresholds[1] * normal for corner in corners]
    vertices2 = [rotation * corner + displayed_thresholds[2] * normal for corner in corners]
    meshscatter!(ax3, points; color=:lightgray, markersize = resolution)
    mesh!(ax3, vertices1, [1 2 3; 3 4 1], color = (lower_color, .7))
    mesh!(ax3, vertices2, [1 2 3; 3 4 1], color = (upper_color, .7))
    
    fig
end

#=
@compile_workload begin
    example_data = joinpath(pkgdir(PlotReassembly),"..", "result-2025-10-15T17:18:10.180.h5")
    plot_result(example_data)
    nothing
end
=#

end # module PlotReassembly
