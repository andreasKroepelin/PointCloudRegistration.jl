using Revise
using GLMakie
using PointCloudRegistration
using CoordinateTransformations
using FileIO
using Colors
using StatsBase
using LinearAlgebra

function partial_transformation(transformation::AffineMap{R, T}, t) where {R, T}
    rotation = transformation.linear ^ t |> real |> R
    translation = t * transformation.translation
    AffineMap(rotation, translation)
end

logo = FileIO.load("../../assets/julia-logo/julia-logo-color.png");
logo_mod = FileIO.load("../../assets/julia-logo/julia-logo-color-modified.png");

heatmap(alpha.(logo) .> 0)

function img2pc(img, dist)
    full_pc = density2pointcloud(alpha.(img'))
    drop_threshold(full_pc, .01)
    thin_to_distance(full_pc, dist)
end

X = img2pc(logo, 10.)
Y = img2pc(logo_mod, 10.)

prepd_X = prepare_target_kc(X);
T = rigid_kc(Y, prepd_X)

let
    fig = Figure()
    ax = Axis(fig[1, 1]; autolimitaspect = 1, yreversed = true)
    sl = Slider(fig[2, 1]; range = 0:0.01:1)
    Y_tr = @lift (partial_transformation(T, $(sl.value)))(Y)
    plot!(ax, X; sizefactor = .15)
    plot!(ax, Y_tr; sizefactor = .15)
    fig
end

TY = T(Y)
dTY = nonrigid_distancepreserving(TY, X; max_edge_length = 10, regularizer = GeneralizedLogNormalRegularizer(2, 1.0001))
# dTY = PointCloud(kcs.new_source_points, TY.weights)

let
    fig = Figure()
    ax = Axis(fig[1, 1]; autolimitaspect = 1, yreversed = true)
    plot!(ax, X; sizefactor = .15)
    # plot!(ax, TY; sizefactor = .15)
    plot!(ax, dTY; sizefactor = .15)
    arrows2d!(ax, TY.points, dTY.points .- TY.points)
    fig
end

let
    # fig = Figure(; size = (300, 200))
    fig = Figure()
    ax = Axis(fig[1, 1]; autolimitaspect = 1, yreversed = true)
    hidespines!(ax)
    hidedecorations!(ax)
    plot!(ax, X; color = Colors.JULIA_LOGO_COLORS.blue, sizefactor = .15)
    pY = plot!(ax, Y; color = Colors.JULIA_LOGO_COLORS.green, sizefactor = .15)
    on(events(fig).tick) do tick
        t = mod(tick.time, 2.5)
        # t = tick.time
        intermediate_Y = if t <= 1
            (partial_transformation(T, t))(Y)
        elseif t <= 2
            l = 1 * (t - 1)
            intermediate_points = l .* dTY.points .+ (1 - l) .* TY.points
            PointCloud(intermediate_points, Y.weights)
        else
            dTY
        end
        Makie.update!(pY; arg1 = intermediate_Y)
    end
    fig
    # record(_ -> (), fig, "register-julia-logo.mp4", 1:3*60; framerate = 60)
end
