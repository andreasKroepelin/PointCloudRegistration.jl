using Revise
using MRCFile
using GLMakie
using PointCloudRegistration
using PointCloudRegistration.StaticArrays
using PointCloudRegistration.Distances
using MappedArrays
using Base.Iterators
using Downloads
using TravelingSalesmanHeuristics

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

resolution = 2.0f1

# pcs = map(pc -> thin_dpmeans(pc, resolution; report_iteration), pcs_dense)
pcs = map(pc -> thin_kmeans(pc, 1000), pcs_dense)
showthem(pcs)

source = pcs.VIB
target = pcs.IVA
# prepd_target = prepare_target_kc(target; scale = DownTo(resolution));
prepd_target = prepare_target_kc(target);

T = register_kc(source, prepd_target; restarts = RandomRestarts(100))

function flipbook(pcs...; markersize = 2)
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    i = Observable(1)
    meshscatter!(ax, @lift(pcs[$i]); markersize)
    last_t = -Inf
    on(events(fig).tick) do tick
        if tick.time - last_t > 0.1
            last_t = tick.time
            i[] = mod(i[] + 1, eachindex(pcs))
        end
    end
    fig
end

T_source = T(source)

flipbook(source, target)

cpd = register_cpd(
    T_source,
    target;
    scale = resolution,
    regularizer_lengthscale = 50.0f0,
    outlier_proportion = 0.1,
    regularizer_strength = 0.0001,
)

let
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    # scatter!(ax, source.points)
    # scatter!(ax, target.points)
    arrows3d!(ax, source.points, cpd.displacement; markerscale = 1)
    fig
end

function tsp_idcs(pc)
    idcs, _ =
        nearest_neighbor(pairwise(euclidean, pc.points); closepath = false)
    # pc[idcs]
    idcs
end

cpd.correspondences[tsp_idcs(target), tsp_idcs(source)]

function diagonalize_permutation!(A)
    rowidcs, colidcs = axes(A)
    cumrowperm = collect(rowidcs)
    cumcolperm = collect(colidcs)
    rowidxdists = similar(rowidcs)
    colidxdists = similar(colidcs)
    roworder = similar(rowidcs)
    rowperm = similar(rowidcs)
    colorder = similar(colidcs)
    colperm = similar(colidcs)
    for iter in 1:1
        @info "iteration" iter
        # for j in colidcs
        for j in 1:100:lastindex(A, 2)
            mid = j * size(A, 1) ÷ size(A, 2)
            @info "j" mid
            rowidxdists .= .- abs.(rowidcs .- mid)
            sortperm!(roworder, rowidxdists)
            sortperm!(rowperm, view(A, cumrowperm, j))
            cumrowperm[roworder] = cumrowperm[rowperm]
        end
        # for i in rowidcs
        # for i in 1:1
        #     mid = i * size(A, 2) ÷ size(A, 1)
        #     @info "i" mid
        #     colidxdists .= .- abs.(colidcs .- mid)
        #     sortperm!(colorder, colidxdists)
        #     sortperm!(colperm, view(A, i, cumcolperm))
        #     cumcolperm[colorder] = cumcolperm[colperm]
        # end
        # for j in reverse(colidcs)
        #     mid = j * size(A, 1) ÷ size(A, 2)
        #     rowidxdists .= .- abs.(rowidcs .- mid)
        #     sortperm!(roworder, rowidxdists)
        #     sortperm!(rowperm, view(A, cumrowperm, j))
        #     cumrowperm[roworder] = cumrowperm[rowperm]
        # end
        # for i in reverse(rowidcs)
        #     mid = i * size(A, 2) ÷ size(A, 1)
        #     colidxdists .= .- abs.(colidcs .- mid)
        #     sortperm!(colorder, colidxdists)
        #     sortperm!(colperm, view(A, i, cumcolperm))
        #     cumcolperm[colorder] = cumcolperm[colperm]
        # end
    end
    A[cumrowperm, cumcolperm]
end

diagonalize_permutation(A) = diagonalize_permutation!(copy(A))

function minimize_soft_bandwidth_score!(A)
    rowidcs, colidcs = axes(A)
    weights = map(CartesianIndices(A)) do ci
        i, j = Tuple(ci)
        (i * size(A, 2) - j * size(A, 1))^2
    end
    rowcosts = map(rowidcs) do i
        row = view(A, i, :)
        wrow = view(weights, i, :)
        mapreduce((w, a) -> w * a^2, +, wrow, row)
    end
    colcosts = map(colidcs) do j
        col = view(A, :, j)
        wcol = view(weights, :, j)
        mapreduce((w, a) -> w * a^2, +, wcol, col)
    end
    cumrowperm = collect(rowidcs)
    cumcolperm = collect(colidcs)
    oldscore = Inf
    for i1 in rowidcs, i2 in rowidcs
        # i1, i2 = rand(rowidcs), rand(rowidcs)
        newrowcost2 = let
            row = view(A, cumrowperm[i1], cumcolperm)
            wrow = view(weights, i2, :)
            mapreduce((w, a) -> w * a^2, +, wrow, row)
        end
        newrowcost1 = let
            row = view(A, cumrowperm[i2], cumcolperm)
            wrow = view(weights, i1, :)
            mapreduce((w, a) -> w * a^2, +, wrow, row)
        end
        if newrowcost1 + newrowcost2 < rowcosts[i1] + rowcosts[i2]
            # @info "improvement" newrowcost1 newrowcost2 rowcosts[i1] rowcosts[i2]
            rowcosts[i1] = newrowcost1
            rowcosts[i2] = newrowcost2
            cumrowperm[i1], cumrowperm[i2] = cumrowperm[i2], cumrowperm[i1]
        end
    end
    for j1 in colidcs, j2 in colidcs
        # j1, j2 = rand(colidcs), rand(colidcs)
        newcolcost2 = let
            col = view(A, cumcolperm, cumcolperm[j1])
            wcol = view(weights, :, j2)
            mapreduce((w, a) -> w * a^2, +, wcol, col)
        end
        newcolcost1 = let
            col = view(A, cumcolperm, cumcolperm[j2])
            wcol = view(weights, :, j1)
            mapreduce((w, a) -> w * a^2, +, wcol, col)
        end
        if newcolcost1 + newcolcost2 < colcosts[j1] + colcosts[j2]
            colcosts[j1] = newcolcost1
            colcosts[j2] = newcolcost2
            cumcolperm[j1], cumcolperm[j2] = cumcolperm[j2], cumcolperm[j1]
        end
    end
    A[cumrowperm, cumcolperm]
end

minimize_soft_bandwidth_score(A) = minimize_soft_bandwidth_score!(copy(A))

let
    fig = Figure()
    ax = Axis(fig[1, 1]; autolimitaspect = 1)
    Aobs = Observable(copy(cpd.correspondences))
    heatmap!(ax, Aobs)
    last_t = -Inf
    rowbuffer = similar(cpd.correspondences, size(cpd.correspondences, 2))
    colbuffer = similar(cpd.correspondences, size(cpd.correspondences, 1))
    old_score = soft_bandwidth_score(cpd.correspondences)
    B = similar(cpd.correspondences)
    on(events(fig).tick) do tick
        tick.time - last_t < 0.001 && return
        last_t = tick.time
        A = Aobs[]
        for _ in 1:100
            i1 = rand(axes(A, 1))
            i2 = rand(axes(A, 1))
            j1 = rand(axes(A, 2))
            j2 = rand(axes(A, 2))
            B .= A
            rowbuffer .= @view A[i1, :]
            A[i1, :] .= @view A[i2, :]
            A[i2, :] .= rowbuffer
            colbuffer .= @view A[:, j1]
            A[:, j1] .= @view A[:, j2]
            A[:, j2] .= colbuffer
            new_score = soft_bandwidth_score(A)
            if new_score > old_score
                old_score = new_score
            else
                A .= B
            end
        end
        notify(Aobs)
    end
    fig
end
