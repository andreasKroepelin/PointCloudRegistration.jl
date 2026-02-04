# using Revise
using PointCloudRegistration
using Mooncake
using OptimalTransport
using LinearAlgebra
using GLMakie

# X, Y = PointCloudRegistration.Assets.load_1ake_A_4ake_A()
X, Y = PointCloudRegistration.Assets.load_1ih7_A_1ig9_A()
X = PointCloud(collect(X.points))
Y = PointCloud(collect(Y.points))
T_rigid = register_kc(Y, X)
Y = T_rigid(Y)

function report_iteration(; iter, gradient, displaced_source_points, kwargs...)
    mod(iter, 100) == 0 || return
    @info "iteration" iter norm(gradient)
    push!(dsps, copy(displaced_source_points))
end

dsps = []
registration = register_kc_springs(Y, X; stiffness = 1//200, iterations = 10_000, report_iteration);
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

registration = register_sinkhorn(Y, X)
Y_disp = apply_displacements(Y, displacements(registration))
