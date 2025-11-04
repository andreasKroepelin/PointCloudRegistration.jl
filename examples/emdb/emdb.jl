using Revise
using MRCFile
using GLMakie
using PointCloudRegistration
using PointCloudRegistration.StaticArrays
using PointCloudRegistration.Distances
using MappedArrays
using Base.Iterators
using Downloads
using LinearAlgebra
using SoftBandwidthMinimization

function emdb_url(id)
    "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-$id/map/emd_$id.map.gz"
end
emdb_file(id) = "emd_$id.map.gz"

#=
from doi.org/10.1038/s41586-020-2447-x
EMD-21619 (structure I-A),
EMD-21620 (structure I-B),
EMD-21621 (structure II-A),
EMD-21622 (structure II-B1),
EMD-21623 (structure II-B2),
EMD-21624 (structure II-C1),
EMD-21625 (structure II-C2),
EMD-21626 (structure II-D),
EMD-21627 (structure III-A),
EMD-21628 (structure III-B),
EMD-21629 (structure III-C),
EMD-21630 (structure IV-A),
EMD-21631 (structure IV-B),
EMD-21632 (structure V-A),
EMD-21633 (structure V-B),
EMD-21634 (structure VI-A),
EMD-21635 (structure VI-B),
EMD-21636 (structure IV-B1-nc),
EMD-21637 (structure IV-B2-nc),
EMD-21638 (structure V-A1-nc),
EMD-21639 (structure V-A2-nc),
EMD-21640 (structure V-B1-nc),
EMD-21641 (structure V-B2-nc)
=#
emdb_ids = (;
    IA = "21619",
    IB = "21620",
    IIA = "21621",
    IIB1 = "21622",
    IIB2 = "21623",
    IIC1 = "21624",
    IIC2 = "21625",
    IID = "21626",
    IIIA = "21627",
    IIIB = "21628",
    IIIC = "21629",
    IVA = "21630",
    IVB = "21631",
    VA = "21632",
    VB = "21633",
    VIA = "21634",
    VIB = "21635",
    IVB1nc = "21636",
    IVB2nc = "21637",
    VA1nc = "21638",
    VA2nc = "21639",
    VB1nc = "21640",
    VB2nc = "21641",
)

function load_map(id)
    filename = emdb_file(id)
    if !isfile(filename)
        Downloads.download(emdb_url(id), filename)
    end
    read(filename, MRCData)
end

mrcs = map(load_map, emdb_ids[(:IVA, :VIB)]);

function mrc2pc(mrc)
    f = let ranges = voxelaxes(header(mrc))
        function (ci)
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

map(
    (pc_dense, pc_full) -> length(pc_dense.points) / length(pc_full.points),
    pcs_dense,
    pcs_full,
)

function showthem(pcs)
    fig = Figure()
    for (i, k) in enumerate(keys(pcs))
        pc = pcs[k]
        ax = Axis3(fig[1, i]; title = string(k), aspect = :data)
        maxw = maximum(pc.weights)
        nw = pc.weights ./ maxw
        scatter!(ax, pc.points; color = tuple.(:blue, nw))
    end
    fig
end

function report_iteration(; iteration, relchange, numclusters, additions)
    println(
        "Iteration $iteration -- relative change $relchange -- $numclusters clusters ($additions newly added).",
    )
end

resolution = 15.0f0

pcs = map(pc -> thin_dpmeans(pc, resolution; report_iteration), pcs_dense)
# # pcs = map(pc -> thin_kmeans(pc, 1000), pcs_dense)
# pcs = map(pc -> thin_kmeans(pc, rand(1000:1500)), pcs_dense)
showthem(pcs)

source = pcs.VIB
target = pcs.IVA
# prepd_target = prepare_target_kc(target; scale = DownTo(resolution));
prepd_target = prepare_target_kc(target);

T = register_kc(source, prepd_target; restarts = RandomRestarts(100))

function flipbook(pcs...; markersize = 2, interval = 0.1)
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    i = Observable(1)
    meshscatter!(
        ax,
        @lift(pcs[$i]);
        markersize,
        color = @lift(eachindex(pcs[$i].points))
    )
    last_t = -Inf
    on(events(fig).tick) do tick
        if tick.time - last_t > interval
            last_t = tick.time
            i[] = mod(i[] + 1, eachindex(pcs))
        end
    end
    fig
end

T_source = T(source)

flipbook(source, target; markersize = 5)

prepd_source = PointCloudRegistration.prepare_source_cpd(
    source;
    regularizer_lengthscale = 10.0f0,
)
cpd = register_cpd(
    prepd_source,
    target;
    scale = resolution / 1,
    outlier_proportion = 0.1,
    regularizer_strength = 0.001,
)

source_disp = PointCloud(source.points .+ cpd.displacement, source.weights)

let
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    # scatter!(ax, source.points; label = "source")
    # scatter!(ax, source_disp.points; label = "displaced source")
    scatter!(ax, cpd.target_representatives; label = "target representatives")
    scatter!(ax, target.points; label = "target")
    arrows3d!(ax, source.points, cpd.displacement; markerscale = 0.1)
    axislegend(ax)
    fig
end

cmap = range(colorant"#0074d900", colorant"#0074d9ff");

heatmap(cpd.correspondences; axis = (; autolimitaspect = 1), colormap = cmap)

trg_sorted_idcs, src_sorted_idcs =
    minimize_soft_bandwidth(cpd.correspondences; iterations = 10)

heatmap(
    cpd.correspondences[trg_sorted_idcs, src_sorted_idcs];
    axis = (; autolimitaspect = 1),
    colormap = cmap,
)

flipbook(
    source[src_sorted_idcs],
    target[trg_sorted_idcs];
    markersize = 10,
    interval = 0.5,
)
