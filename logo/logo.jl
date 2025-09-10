using GLMakie
using CairoMakie
using Rotations
using Statistics

whiten(col, deg) =
    range(Makie.to_color(col), Makie.to_color((:gray80, 1)))[clamp(
        round(Int, deg * 100),
        1,
        100,
    )]

angles = range(0, 2pi; length = 4)[1:3]
centers = map(c -> 0.5 .* c, sincos.(angles))
n = 20
r = 0.3
coords = hcat(
    [
        stack([
            center .+ r .* sqrt(rand()) .* sincos(rand() * 2pi) for _ in 1:n
        ]) for center in centers
    ]...,
)
base_colors = collect(GLMakie.Colors.JULIA_LOGO_COLORS[(:green, :purple, :red)])
tcolors = Makie.to_color.(tuple.(base_colors, 0.5))
colors = repeat(base_colors; inner = n);

let
    CairoMakie.activate!(; px_per_unit = 1)
    fig = Figure(; size = (512, 512), backgroundcolor = :transparent)
    ax = Axis(fig[1, 1]; autolimitaspect = 1, backgroundcolor = :transparent)
    hidedecorations!(ax)
    hidespines!(ax)
    rot = Angle2d(deg2rad(-25))
    tlt = [1.0, 0.3]
    common_marker_args = (
        strokecolor = (:black, 0.2),
        strokewidth = 4,
        markersize = 0.2,
        markerspace = :data,
    )
    scatter!(
        ax,
        rot * coords .+ tlt;
        color = whiten.(colors),
        marker = :rect,
        common_marker_args...,
    )
    for i in axes(coords, 2)
        i % 2 == 0 || continue
        x = coords[:, i]
        traj = stack([rot^t * x .+ t * tlt for t in range(0, 1; length = 100)])
        lines!(
            ax,
            traj;
            color = (:black, 0.7),
            linestyle = (:dash, :normal),
            linewidth = 2,
        )
    end
    scatter!(ax, coords; color = colors, common_marker_args...)
    save("logo_.png", fig)
end

let
    CairoMakie.activate!(; px_per_unit = 1)
    fig = Figure(; size = (512, 512), backgroundcolor = :transparent)
    ax = Axis(fig[1, 1]; autolimitaspect = 1, backgroundcolor = :transparent)
    hidedecorations!(ax)
    hidespines!(ax)
    scatter!(
        ax,
        coords;
        color = colors,
        strokecolor = (:black, 0.2),
        strokewidth = 4,
        markersize = 0.2,
        markerspace = :data,
    )
    save("logo2_.png", fig)
end

let
    CairoMakie.activate!(; px_per_unit = 1)
    # GLMakie.activate!()
    # fig = Figure(; size = (512, 512))
    # ax = Axis(fig[1, 1]; autolimitaspect = 1)
    fig = Figure(; size = (512, 512), backgroundcolor = :transparent)
    ax = Axis(fig[1, 1]; autolimitaspect = 1, backgroundcolor = :transparent)
    hidedecorations!(ax)
    hidespines!(ax)
    points = stack(tuple.(1:3, 0))
    rot = Angle2d(deg2rad(-45))
    tlt = [-1.0, -1.0]
    for t in range(1, 0.1; length = 10)
        t = t^1.7
        scatter!(
            ax,
            rot^t * points .+ t * tlt;
            color = whiten.(base_colors, t),
            markersize = 1,
            markerspace = :data,
            strokecolor = :gray90,
            strokewidth = 3,
        )
    end
    scatter!(
        ax,
        points;
        color = base_colors,
        markersize = 1,
        markerspace = :data,
        strokecolor = :gray90,
        strokewidth = 3,
    )
    save("logo.png", fig)
end
