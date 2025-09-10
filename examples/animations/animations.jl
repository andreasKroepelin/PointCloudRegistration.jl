using Revise
using GLMakie
using PointCloudRegistration
using PointCloudRegistration.CoordinateTransformations
using FileIO
using Colors
using StatsBase

function partial_transformation(transformation::AffineMap{R, T}, t) where {R, T}
    rotation = transformation.linear ^ t |> real |> R
    translation = t * transformation.translation
    AffineMap(rotation, translation)
end

logo = FileIO.load("../../assets/julia-logo/julia-logo-color.png");
logo_mod = FileIO.load("../../assets/julia-logo/julia-logo-color-modified.png");

heatmap(alpha.(logo) .> 0)

function img2pc(img, num)
    idcs = vec(CartesianIndices(img))
    weights = vec(alpha.(img) .> 0)
    points =
        sample(idcs, Weights(weights), num) .|>
        Tuple .|>
        reverse |>
        stack .|>
        Float64
    PointCloud(points)
end

X = img2pc(logo, 1000)
Y = img2pc(logo_mod, 1000)

T = register_kc(Y, X)

let
    fig = Figure()
    ax = Axis(fig[1, 1]; autolimitaspect = 1, yreversed = true)
    sl = Slider(fig[2, 1]; range = 0:0.01:1)
    Y_tr = @lift (partial_transformation(T, $(sl.value)))(Y)
    scatter!(ax, X)
    scatter!(ax, Y_tr)
    fig
end

TY = T(Y)

cpd = register_cpd(
    TY,
    X;
    scale = 5.0,
    outlier_proportion = 0.01,
    regularizer_strength = 0.05,
    regularizer_lengthscale = 50.0,
);

let
    fig = Figure()
    ax = Axis(fig[1, 1]; autolimitaspect = 1, yreversed = true)
    sl = Slider(fig[2, 1]; range = 0:0.01:1)
    Y_tr = @lift let t = $(sl.value)
        TY.points .+ t .* cpd.displacement
    end
    scatter!(ax, X)
    scatter!(ax, Y_tr)
    # arrows2d!(ax, TY.points, cpd.displacement; color = (:gray, .5))
    fig
end

let
    fig = Figure(; size = (300, 200))
    ax = Axis(fig[1, 1]; autolimitaspect = 1, yreversed = true)
    hidespines!(ax)
    hidedecorations!(ax)
    t = Observable(0.0)
    Y_tr = map(t) do t
        t = min(t, 1.5)
        if t <= 1
            (partial_transformation(T, t))(Y).points
        else
            TY.points .+ 2(t - 1) .* cpd.displacement
        end
    end
    scatter!(ax, X; color = Colors.JULIA_LOGO_COLORS.blue, markersize = 4)
    scatter!(ax, Y_tr; color = Colors.JULIA_LOGO_COLORS.green, markersize = 4)
    record(fig, "register-julia-logo.mp4", 0:0.01:2; framerate = 60) do t_
        t[] = t_
    end
end
