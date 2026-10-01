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
    full_pc = density2pointcloud(float.(alpha.(img')))
    dropped = drop_threshold(full_pc, 0.01)
    thin_to_distance(dropped, dist)
end

X = img2pc(logo, 10.0)
Y = img2pc(logo_mod, 10.0)

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

prepd_X = prepare_target_kernelcorrelation(X);
hist_transformation = []
best_hist_transformation = []
min_cost = Inf
T = rigid_registration(Y, X, KernelCorrelationMM(; report_restart, report_iteration = report_iteration_r); target_preparation = prepd_X)
pushfirst!(
    best_hist_transformation,
    PointCloudRegistration.identity_transformation(
        first(best_hist_transformation),
    ),
);
rigid_trajectory = [T(Y) for T in best_hist_transformation];

let
    fig = Figure()
    ax = Axis(fig[1, 1]; autolimitaspect = 1, yreversed = true)
    sl = Slider(fig[2, 1]; range = eachindex(rigid_trajectory))
    Y_tr = @lift rigid_trajectory[$(sl.value)]
    plot!(ax, X; sizefactor = 0.15)
    plot!(ax, Y_tr; sizefactor = 0.15)
    fig
end

function report_iteration_nr(; iteration, new_source_points, sqsigma)
    if iteration % 10_000 == 0
        @info "iteration" iteration sqrt(sqsigma)
        push!(hist_new_source_points, copy(new_source_points))
    end
end

TY = T(Y)
prepd_source = prepare_source_distancepreserving(TY; max_edge_length = 13);

hist_new_source_points = []
nr_transformation = nonrigid_registration(
    TY,
    X,
    DistancePreserving(;
        init_noise = 0,
        sensitivity = 2,
        rel_deviation = 0.01,
        report_iteration = report_iteration_nr,
        iterations = 100_000_000,
        batching = StochasticBatch(20),
        # batching = FullBatch(),
    );
    source_preparation = prepd_source,
)
dTY = nr_transformation(TY)
nonrigid_trajectory = let
    L = length(hist_new_source_points)
    l = length(best_hist_transformation)
    [PointCloud(hist_new_source_points[i], PointCloudRegistration.weights(Y)) for i in 1:cld(L, l):L]
end;

let
    fig = Figure()
    ax = Axis(fig[1, 1]; autolimitaspect = 1, yreversed = true)
    plot!(ax, TY; sizefactor = 0.15)
    pts = typeof(TY[1].coords)[]
    for (j1, j2) in prepd_source.neighbor_edges.from_to
        push!(pts, TY[j1].coords)
        push!(pts, TY[j2].coords)
    end
    linesegments!(ax, stack(collect.(pts)); color = :gray)
    fig
end

let
    fig = Figure()
    ax = Axis(fig[1, 1]; autolimitaspect = 1, yreversed = true)
    sl = Slider(fig[2, 1]; range = eachindex(nonrigid_trajectory))
    Y_tr = @lift nonrigid_trajectory[$(sl.value)]
    plot!(ax, X; sizefactor = 0.15)
    plot!(ax, Y_tr; sizefactor = 0.15)
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
    plot!(ax, X; color = Colors.JULIA_LOGO_COLORS.blue, sizefactor = 0.15)
    pY = plot!(ax, Y; color = Colors.JULIA_LOGO_COLORS.green, sizefactor = 0.15)
    trajectory = vcat(rigid_trajectory, nonrigid_trajectory)
    on(events(fig).tick) do tick
        i = mod(round(Int, tick.time / 0.01), eachindex(trajectory))
        # t = tick.time
        Makie.update!(pY; arg1 = trajectory[i])
    end
    fig
    record(
        _ -> (),
        fig,
        "register-julia-logo.mp4",
        1:120;
        framerate = 30,
        compression = 20,
    )
end
