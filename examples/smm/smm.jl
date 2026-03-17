using Revise
using PointCloudRegistration
using GLMakie
using Random

X, Y = PointCloudRegistration.Assets.load_1ake_A_4ake_A()
scale = 0.2 * PointCloudRegistration.avg_nn_dist(X)
pX = prepare_target_kc(X; scale = DownTo(scale));

iterations = 200
transformations = []
function warn_divergence(; iter, transformation, kwargs...)
    push!(transformations, transformation)
    iter == iterations && @warn "did not converge!"
end

restart_transformations = []
restart_costs = []
function report_restart(; restart, cost, transformation, kwargs...)
    push!(restart_transformations, transformation)
    push!(restart_costs, cost)
end

empty!(transformations)
empty!(restart_transformations)
empty!(restart_costs)
@time rigid_kc(
    Y,
    pX;
    restarts = RandomRestarts(1000, Xoshiro(1)),
    iterations,
    stochastic_majorization_minimization = SomePoints(100, Xoshiro(1)),
    # report_iteration = warn_divergence,
    report_restart,
)

mask = restart_costs .< -50
restart_transformations = restart_transformations[mask]
restart_costs = restart_costs[mask]
let
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    sl = Slider(fig[2, 1]; range = eachindex(restart_transformations))
    markersize = 0.5PointCloudRegistration.avg_nn_dist(X)
    meshscatter!(ax, X.points; markersize)
    meshscatter!(
        ax,
        (@lift ((restart_transformations[$(sl.value)])(Y)).points);
        markersize,
    )
    Label(
        fig[0, 1],
        (@lift string(restart_costs[$(sl.value)]));
        tellwidth = false,
    )
    fig
end
