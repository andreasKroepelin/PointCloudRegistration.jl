using Revise
using DelimitedFiles
using PointCloudRegistration
using GLMakie

pc_1ake = readdlm("../../assets/ake/1ake.csv", ',')
pc_4ake = readdlm("../../assets/ake/4ake.csv", ',')

function report_pair(; source_idx, target_idx, dist)
    push!(correspondences, (source_idx, target_idx))
    push!(distances, dist)
end
function report_iteration(; iter, cost, kwargs...)
    global correspondences
    global distances
    push!(
        correspondence_collection,
        (; restart = current_restart, iter, cost, correspondences, distances),
    )
    correspondences = Tuple{Int, Int}[]
    distances = Float64[]
end
function report_restart(; restart, kwargs...)
    global current_restart
    current_restart = restart
end

good_T = rigid_gmc(pc_1ake, pc_4ake)

correspondence_collection = []
correspondences = Tuple{Int, Int}[]
distances = Float64[]
current_restart = 0
rigid_icp(
    pc_1ake,
    pc_4ake;
    dist_cutoff = 3.0,
    restarts = FixedRestarts([good_T]),
    report_pair,
    report_iteration,
    report_restart,
)

correspondence_collection

let
    fig = Figure()
    sl = Slider(fig[2, 1]; range = eachindex(correspondence_collection))
    title = map(sl.value) do idx
        cc = correspondence_collection[idx]
        string(
            "restart: ",
            cc.restart,
            " iteration: ",
            cc.iter,
            " cost: ",
            cc.cost,
        )
    end
    ax = Axis(fig[1, 1]; autolimitaspect = 1, title)
    to_plot = @lift correspondence_collection[$(sl.value)].correspondences
    color = @lift correspondence_collection[$(sl.value)].distances
    plt = scatter!(ax, to_plot; color)
    Colorbar(fig[1, 2], plt)
    fig
end
