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

heatmap(alpha.(logo_mod) .> 0)

function img2pc(img, dist)
    full_pc = density2pointcloud(alpha.(img'))
    drop_threshold(full_pc, .01)
    thin_to_distance(full_pc, dist)
end

X = img2pc(logo, 10.)
Y = img2pc(logo_mod, 10.)

function report_iteration_r(; transformation, kwargs...)
    push!(hist_transformation, transformation)
end
function report_restart(; cost, kwargs...)
    global min_cost
    global best_hist_transformation
    if cost < min_cost
        best_hist_transformation = copy(hist_transformation)
        min_cost = cost
    end
    empty!(hist_transformation)
end

prepd_X = prepare_target_kc(X);
hist_transformation = []
best_hist_transformation = []
min_cost = Inf
T = rigid_kc(Y, prepd_X; report_restart, report_iteration = report_iteration_r)
pushfirst!(
    best_hist_transformation,
    PointCloudRegistration.identity_transformation(first(best_hist_transformation)),
)
rigid_trajectory = [T(Y) for T in best_hist_transformation]

let
    fig = Figure()
    ax = Axis(fig[1, 1]; autolimitaspect = 1, yreversed = true)
    sl = Slider(fig[2, 1]; range = eachindex(rigid_trajectory))
    Y_tr = @lift rigid_trajectory[$(sl.value)]
    plot!(ax, X; sizefactor = .15)
    plot!(ax, Y_tr; sizefactor = .15)
    fig
end

function report_iteration_nr(; iter, new_source_points, sqsigma)
    iter % 1000 == 0 && @info "iteration" iter sqrt(sqsigma)
    push!(hist_new_source_points, copy(new_source_points))
end

TY = T(Y)
prepd_source = prepare_source_distancepreserving(TY; max_edge_length = 13)

hist_new_source_points = []
dTY = nonrigid_distancepreserving(prepd_source, X; init_noise = 0, regularizer = GeneralizedLogNormalRegularizer(2, 1.001), report_iteration = report_iteration_nr, iterations = 20_000)
dTY = PointCloud(collect(dTY.points), collect(dTY.weights))
nonrigid_trajectory = let
    L = length(hist_new_source_points)
    l = length(best_hist_transformation)
    [PointCloud(hist_new_source_points[i], Y.weights) for i in 1:cld(L, l):L]
end

let
    fig = Figure()
    ax = Axis(fig[1, 1]; autolimitaspect = 1, yreversed = true)
    plot!(ax, TY; sizefactor = .15)
    pts = eltype(TY.points)[]
    for (j1, j2) in prepd_source.neighbor_graph.edges
        push!(pts, TY.points[j1])
        push!(pts, TY.points[j2])
    end
    linesegments!(ax, stack(collect.(pts)); color = :gray)
    fig
end

let
    fig = Figure()
    ax = Axis(fig[1, 1]; autolimitaspect = 1, yreversed = true)
    sl = Slider(fig[2, 1]; range = eachindex(nonrigid_trajectory))
    Y_tr = @lift nonrigid_trajectory[$(sl.value)]
    plot!(ax, X; sizefactor = .15)
    plot!(ax, Y_tr; sizefactor = .15)
    fig
end

let
    fig = Figure()
    ax = Axis(fig[1, 1]; autolimitaspect = 1, yreversed = true)
    # plot!(ax, X; sizefactor = .15)
    # plot!(ax, TY; sizefactor = .15)
    # plot!(ax, dTY; sizefactor = .15)
    arrows2d!(ax, TY.points, dTY.points .- TY.points)
    fig
end

GLMakie.activate!(; px_per_unit = 2.0)
let
    fig = Figure(; size = (300, 200))
    # fig = Figure()
    ax = Axis(fig[1, 1]; autolimitaspect = 1, yreversed = true)
    hidespines!(ax)
    hidedecorations!(ax)
    plot!(ax, X; color = Colors.JULIA_LOGO_COLORS.blue, sizefactor = .15)
    pY = plot!(ax, Y; color = Colors.JULIA_LOGO_COLORS.green, sizefactor = .15)
    trajectory = vcat(rigid_trajectory, nonrigid_trajectory)
    on(events(fig).tick) do tick
        i = mod(round(Int, tick.time / 0.01), eachindex(trajectory))
        # t = tick.time
        Makie.update!(pY; arg1 = trajectory[i])
    end
    fig
    record(_ -> (), fig, "register-julia-logo.mp4", 1:170; framerate = 30, compression = 20)
end
