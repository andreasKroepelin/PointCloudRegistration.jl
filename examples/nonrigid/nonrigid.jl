using Revise
using PointCloudRegistration
# using OptimalTransport
using LinearAlgebra
using GLMakie

# X, Y = PointCloudRegistration.Assets.load_1ake_A_4ake_A()
# X, Y = PointCloudRegistration.Assets.load_1su4_A_1iwo_A()
Y, X = PointCloudRegistration.Assets.load_1ih7_A_1ig9_A()
# X = PointCloud(collect(map(x -> Float64.(x), X.points)))
# Y = PointCloud(collect(map(x -> Float64.(x), Y.points)))
T_rigid = rigid_kc(Y, X)
Y = T_rigid(Y)

function report_iteration(; iter, new_source_points, sqsigma, sqlambda, kwargs...)
    pseudo_sqrt(x) = x < 0 ? NaN : sqrt(x)
    sigma = pseudo_sqrt(sqsigma)
    lambda = pseudo_sqrt(sqlambda)
    iter % 100 == 0 && @info "iteration" iter sigma lambda
    push!(new_source_points_hist, copy(new_source_points))
    push!(sigma_hist, sigma)
    push!(lambda_hist, lambda)
end

function report_iteration(; iter, new_source_points, kc, spring_energy, spring_pairs, kwargs...)
    push!(new_source_points_hist, copy(new_source_points))
    push!(kc_hist, kc)
    push!(spring_energy_hist, spring_energy)
    if iter == 1
        empty!(all_spring_pairs)
        append!(all_spring_pairs, spring_pairs)
    end
end

new_source_points_hist = []
sigma_hist = []
lambda_hist = []
nonrigid_gmml(Y, X; max_spring_length = 10., report_iteration)

new_source_points_hist = []
kc_hist = []
spring_energy_hist = []
all_spring_pairs = []
kcsr = nonrigid_kc_springs(Y, X; stiffness = 8e6, scale = 10., max_spring_length = 7., report_iteration);

let
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    plot!(ax, Y; sizefactor = 2)
    ls = mapreduce(((j1, j2),) -> [Y.points[j1], Y.points[j2]], vcat, all_spring_pairs)
    linesegments!(ax, ls; color = :gray)
    fig
end

let
    fig = Figure()
    ax1 = Axis3(fig[1, 1]; aspect = :data)
    ax2 = Axis(fig[1, 2];)
    sl = Slider(fig[2, 1]; range = eachindex(new_source_points_hist))
    lines!(ax2, sigma_hist)
    lines!(ax2, lambda_hist)
    iY = @lift PointCloud(new_source_points_hist[$(sl.value)], Y.weights)
    io = @lift [Point2($(sl.value), sigma_hist[$(sl.value)]), Point2($(sl.value), lambda_hist[$(sl.value)])]
    # ls = @lift mapreduce((y, iy) -> [y, iy], vcat, Y.points, new_source_points_hist[$(sl.value)])
    # plot!(ax1, Y; sizefactor = 2)
    plot!(ax1, X; sizefactor = 2)
    plot!(ax1, iY; sizefactor = 2 #=, color = eachindex(Y.points) =#)
    # linesegments!(ax1, ls; color = :gray)
    scatter!(ax2, io; markersize = 10)
    fig
end

let
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    # src_plt = plot!(ax, Y; label = "source")
    # trg_plt = plot!(ax, X; label = "target")
    plot!(ax, kcsr(Y); label = "displaced source")
    # arrows3d!(ax, Y.points, kcsr(Y).points .- Y.points; label = "displacement")
    axislegend(ax)
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
