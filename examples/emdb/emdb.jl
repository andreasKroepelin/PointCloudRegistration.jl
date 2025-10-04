using Revise
using MRCFile
using GLMakie
using PointCloudRegistration
using PointCloudRegistration.StaticArrays
using MappedArrays
using Base.Iterators

mrcs = (IA = read("/home/andreas/Downloads/emd_21619.map.gz", MRCData),
IB = read("/home/andreas/Downloads/emd_21620.map.gz", MRCData),
);

function mrc2pc(mrc)
    f = let ranges = voxelaxes(header(mrc))
        function(ci)
            is = Tuple(ci)
            SVector(getindex.(ranges, is))
        end
    end
    points = vec(mappedarray(f, CartesianIndices(mrc)))
    weights = vec(mappedarray(w -> max(w, zero(w)), mrc.data))
    PointCloud(points, weights)
end

pcs_full = map(mrc2pc, mrcs)

pcs_dense = map(pc -> thin_droplowweight(pc, 0.01), pcs_full)

map((pc_dense, pc_full) -> length(pc_dense.points) / length(pc_full.points), pcs_dense, pcs_full)

function showthem(pcs...)
    fig = Figure()
    for (i, pc) in enumerate(pcs)
        ax = Axis3(fig[1, i]; title = string(i), aspect = :data)
        maxw = maximum(pc.weights)
        nw = pc.weights ./ maxw
        scatter!(ax, pc.points; color = tuple.(:blue, nw))
    end
    fig
end

function report_iteration(; iteration, relchange, numclusters, additions)
    println("Iteration $iteration -- relative change $relchange -- $numclusters clusters ($additions newly added).")
end

resolution = 2f1

pcs = map(pc -> thin_dpmeans(pc, resolution; report_iteration), pcs_dense)
showthem(pcs...)

prepd_IA = prepare_target_kc(pcs.IA; scale = DownTo(resolution));

T = register_kc(pcs.IB, prepd_IA, restarts = RandomRestarts(100))

function flipbook(pcs...; markersize = 2)
    fig = Figure()
    ax = Axis3(fig[1, 1], aspect = :data)
    i = Observable(1)
    meshscatter!(ax, @lift(pcs[$i]); markersize)
    last_t = -Inf
    on(events(fig).tick) do tick
        if tick.time - last_t > .1
            last_t = tick.time
            i[] = mod(i[] + 1, eachindex(pcs))
        end
    end
    fig
end

TIB = T(pcs.IB)
IA = pcs.IA

flipbook(TIB, IA)

cpd = register_cpd(TIB, IA; scale = resolution, regularizer_lengthscale = 50f0, outlier_proportion = .1, regularizer_strength = .01)

let
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    scatter!(ax, TIB.points)
    scatter!(ax, IA.points)
    arrows3d!(ax, TIB.points, cpd.displacement)
    fig
end

