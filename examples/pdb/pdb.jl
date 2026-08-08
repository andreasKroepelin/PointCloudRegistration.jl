using Revise
using BioStructures
using BioSequences
using BioAlignments
using ProteinSecondaryStructures
using StatsBase
using LinearAlgebra
using Random
using PointCloudRegistration
using PointCloudRegistration.Rotations
import PointCloudRegistration as PCReg
using DataFrames
using Makie
import CairoMakie
import GLMakie
using Makie.Unitful
using ProgressMeter
using Clustering
using InvertedIndices

function compute_rmsd(Y, X)
    T = rigid_rmsd(Y, X)
    PCReg.evaluate_rmsd(Y, X, T)
end

pointclouds, pdb_ids = let
    all_ids = split(
        """
        1IWO,1KJU,1SU4,1T5S,1T5T,1VFP,1WPG,1XP5,2AGV,2BY4,2C88,2C8K,2C8L,2C9M,\
        2DQS,2EAR,2EAT,2EAU,2O9J,2OA0,2YFY,2ZBD,2ZBE,2ZBF,2ZBG,3AR2,3AR3,3AR4,\
        3AR5,3AR6,3AR7,3AR8,3AR9,3B9B,3B9R,3BA6,3FGO,3FPB,3FPS,3J7T,3N5K,3N8G,\
        3NAL,3NAM,3NAN,3W5A,3W5B,3W5C,3W5D,4H1W,4J2T,4KYT,4UU0,4UU1,4XOU,4Y3U,\
        4YCL,4YCM,4YCN,5A3Q,5A3R,5A3S,5NCQ,5XA7,5XA8,5XA9,5XAA,5XAB,6HEF,6YAA,\
        6YSO,4BEW,8OWA,8OWL\
        """,
        ",",
    )
    all_pdbs = [retrievepdb(id; dir = "data")["A"] for id in all_ids]
    exact_seq_mask =
        [countatoms(pdb, calphaselector) == 994 for pdb in all_pdbs]
    exact_seq_ids = all_ids[exact_seq_mask]
    exact_seq_pdbs = all_pdbs[exact_seq_mask]
    exact_seq_pcs = [
        PointCloud(coordarray(pdb, calphaselector)) for pdb in exact_seq_pdbs
    ]
    rmsds = [compute_rmsd(Y, X) for Y in exact_seq_pcs, X in exact_seq_pcs]
    clustering = hclust(Symmetric(rmsds))
    assignments = cutree(clustering; h = 4.0)
    pcs = PointCloud{3, Float64}[]
    ids = String[]
    for a in unique(assignments)
        idx = findfirst(==(a), assignments)
        push!(pcs, exact_seq_pcs[idx])
        push!(ids, exact_seq_ids[idx])
    end
    pcs, ids
end

secondary_structure = dssp_run("data/1SU4.cif") .|> ss_code

function greedy_shortest_path(dists)
    path = [1]
    dists[1, :] .= typemax(eltype(dists))
    dists = copy(dists)
    while length(path) < size(dists, 1)
        next = argmin(view(dists, :, last(path)))
        push!(path, next)
        dists[next, :] .= typemax(eltype(dists))
    end
    path
end

pairs = [
    (pointclouds[i], pointclouds[j]) for i in eachindex(pointclouds) for
    j in eachindex(pointclouds) if i > j
]
id_pairs = [
    (pdb_ids[i], pdb_ids[j]) for i in eachindex(pdb_ids) for
    j in eachindex(pdb_ids) if i > j
]

Ts_per_method = (
    rmsd = [rigid_rmsd(source, target) for (source, target) in pairs],
    gmc = [rigid_gmc(source, target) for (source, target) in pairs],
);
registered_pairs_per_method = map(Ts_per_method) do Ts
    [(T(source), target) for (T, (source, target)) in zip(Ts, pairs)]
end;

norms = map(registered_pairs_per_method) do reg_pairs
    ns = stack(reg_pairs) do (source, target)
        norm.(source.points .- target.points)
    end
end
orders = map(norms) do ns
    cm = cor(ns; dims = 1)
    greedy_shortest_path(-cm)
end

let
    fig = Figure(; size = (600, length(pairs) * 20 + 10))
    Label(fig[1, 0], "RMSD"; tellheight = false, rotation = pi/2, font = :bold)
    Label(fig[2, 0], "GMC"; tellheight = false, rotation = pi/2, font = :bold)
    yticks = (
        eachindex(id_pairs),
        [rich("$src \u2013 $trg"; fontsize = 8) for (src, trg) in id_pairs],
    )
    axs = (rmsd = Axis(fig[1, 1]; yticks), gmc = Axis(fig[2, 1]; yticks))
    hidexdecorations!(axs.rmsd)
    linkaxes!(axs.rmsd, axs.gmc)
    nmax = maximum(maximum, norms)
    colormap = :lisbon
    Colorbar(
        fig[1:2, 2];
        colormap,
        colorrange = (0, nmax),
        label = "distance of corresponding atoms [Å]",
    )
    for method in [:rmsd, :gmc]
        heatmap!(
            axs[method],
            norms[method][:, orders.rmsd];
            colorrange = (0, nmax),
            colormap,
            rasterize = 10,
        )
    end
    # GLMakie.activate!(); display(fig)
    CairoMakie.activate!(; pdf_version = "1.5")
    save("../../paper/bioinformatics/src/img/pdb-heatmaps.pdf", fig)
end

function show_all(pcs)
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    sl = Slider(fig[2, 1]; range = eachindex(pcs))
    meshscatter!(
        ax,
        @lift(pcs[$(sl.value)].points);
        markersize = 3,
        color = eachindex(pcs[1].points),
    )
    GLMakie.activate!()
    display(fig)
end

show_all(pointclouds)
show_all(registered_pointclouds_per_method.rmsd)
show_all(registered_pointclouds_per_method.gmc)

norms = map(registered_pointclouds_per_method) do reg_pcs
    stack(reg_pcs) do reg_pc
        norm.(target.points .- reg_pc.points)
    end
end
# dissimilar_mask = vec(sqrt.(mean(norms.rmsd .^ 2; dims = 1)) .> 10)
# norms_filtered = map(norms) do ns
#     ns[:, dissimilar_mask]
# end

qs = [0.25, 0.5, 0.75] |> reverse
normqs = map(norms_filtered) do ns
    stack(eachrow(ns)) do row
        [quantile(row, q) for q in qs]
    end |> permutedims
end
normqs_avrgd = map(normqs) do nqs
    w = 30
    mapreduce(vcat, 1:(lastindex(nqs, 1) - w)) do offset
        mean(@view(nqs[offset:min(offset + w, end), :]); dims = 1)
    end
end

let
    methods = [:rmsd, :gmc]
    fig = Figure()
    axs = [Axis(fig[1, i]) for i in eachindex(methods)]
    for ax in axs
        linkyaxes!(ax, axs[1])
    end
    colors = cgrad(:blues, length(pointclouds))
    for (ax, method) in zip(axs, methods)
        for (color, nqs) in zip(colors, eachcol(normqs[method]))
            # lines!(ax, nqs)
            band!(ax, 1:length(nqs), zeros(length(nqs)), nqs)
        end
    end
    GLMakie.activate!()
    display(fig)
end

let
    methods = [:rmsd, :gmc] |> reverse
    colors = (rmsd = :tomato, gmc = :cornflowerblue)
    fig = Figure()
    ax = Axis(fig[1, 1])
    for method in methods
        series!(
            ax,
            norms[method]';
            color = fill(colors[method], 59) #=, linewidth = 0, markersize = 3=#
        )
        # for nqs in eachcol(norms[method])
        #     lines!(ax, nqs; color = colors[method], alpha = .2)
        #     # band!(ax, 1:length(nqs), zeros(length(nqs)), nqs)
        # end
    end
    # axislegend(ax)
    Legend(
        fig[1, 1],
        [
            LineElement(; color = colors[method], linewidth = 3) for
            method in methods
        ],
        uppercase.(string.(methods));
        tellwidth = false,
        tellheight = false,
        valign = :top,
        halign = :right,
        margin = ntuple(Returns(10), 4),
    )
    GLMakie.activate!()
    display(fig)
end

let
    methods = [:rmsd, :gmc]
    colorrange_hi = maximum(maximum, norms[methods])
    cmap = :managua
    fig = Figure(; size = (400, 400))
    axs = [
        Axis3(
            fig[1, i];
            aspect = :data,
            title = uppercase(string(methods[i])),
            [Symbol(d, :ticklabelsvisible) => false for d in [:x, :y, :z]]...,
        ) for i in (1, 2)
    ]
    # Colorbar(fig[1, 3]; limits = (0, colorrange_hi), colormap = cmap, label = "distance of corresponding atoms [Å]")
    # meshscatter!(ax, Ts.gmc(Y).points; markersize = 3)
    # meshscatter!(ax, X.points; markersize = 3)
    # meshscatter!(ax, Ts.kc(Y).points; markersize = 3, color = eachindex(X.points))
    # meshscatter!(ax, X.points; markersize = 3, color = eachindex(X.points), marker = Rect3f(Point3f(-1), Vec3f(2)))
    segments = [NTuple{2, Point3f}[] for _ in axs]
    for (y, x) in zip(Y.points, X.points)
        for (ss, method) in zip(segments, methods)
            T = Ts[method]
            push!(ss, (T(y), x))
        end
    end
    for (ax, ss, method) in zip(axs, segments, methods)
        hidespines!(ax)
        # hidedecorations!(ax)
        T = Ts[method]
        # linesegments!(ax, ss; linewidth = 1)
        # meshscatter!(ax, X.points; markersize = 3, color = norms[method], colorrange = (0, colorrange_hi), colormap = cmap)
        meshscatter!(
            ax,
            X.points;
            markersize = max.(0.2, 3 .* norms[method] ./ colorrange_hi),
        )
    end
    # arrows3d!(ax, Ts.gmc(Y).points, X.points .- Ts.gmc(Y).points; markerscale = .1, lengthscale = 1)
    # GLMakie.activate!(); display(fig)
    GLMakie.activate!()
    save(
        "../../paper/bioinformatics/src/img/pdb-connections.png",
        fig;
        px_per_unit = 3,
    )
end

norms = map(T -> norm.(X.points .- T(Y).points), Ts)
maxnorm = maximum(maximum, norms)
edges = range(0, maxnorm; length = 60)
norm_hists = map(ns -> fit(Histogram, ns, edges), norms)

let
    fig = Figure(; size = (400, 350))
    methods = [:gmc, :mad, :kc, :rmsd]
    offset = -0.05
    ax = Axis(
        fig[1, 1];
        title = "$id_Y \u2194 $id_X",
        yticks = offset .* eachindex(methods),
        yticklabelsvisible = false,
        ylabel = "frequency",
        xlabel = "distance of corresponding atoms [Å]",
    )
    for (i, method) in enumerate(methods)
        density!(
            ax,
            norms[method];
            label = uppercase(string(method)),
            alpha = 0.8,
            strokearound = true,
            strokewidth = 2,
            offset = offset * i,
            bandwidth = 0.5,
        )
    end
    axislegend(ax; valign = :bottom)
    # GLMakie.activate!(); display(fig)
    CairoMakie.activate!(; pdf_version = "1.5")
    save("../../paper/bioinformatics/src/img/pdb-distances.pdf", fig)
end

data = let # X = PCReg.rand_transformation(Xoshiro(-2), Y, X)(X)
    scale = DownTo(PCReg.avg_nn_dist(X))
    prepd_X = prepare_target_kc(X)
    rr() = RandomRestarts(100_000, Xoshiro(1))
    Ts_global = (
        gmc = rigid_gmc(Y, X; scale, restarts = rr()),
        kc = rigid_kc(Y, prepd_X; restarts = rr()),
        icp = rigid_icp(Y, X; restarts = rr()),
    )
    inv_Ts_global = map(inv, Ts_global)
    rows = []
    @showprogress for nr in round.(Int, logrange(1, 10_000, length = 20))
        seeds = min(10_000 ÷ nr, 100)
        for seed in 1:seeds
            rng = Xoshiro(seed)
            inits = [PCReg.rand_transformation(rng, Y, X) for _ in 1:nr]
            restarts =
                FixedRestarts(Vector{typeof(first(Ts_global))}(inits))
            Ts = (;
                gmc = rigid_gmc(Y, X; scale, restarts),
                kc = rigid_kc(Y, prepd_X; restarts),
                icp = rigid_icp(Y, X; restarts),
            )
            diff_Ts = map(∘, Ts, inv_Ts_global)
            angles = map(
                rad2deg ∘ rotation_angle ∘ RotMatrix ∘ (T -> T.linear),
                diff_Ts,
            )
            norms = map(norm ∘ (T -> T.translation), diff_Ts)
            frobs = map(Ts, Ts_global) do T, T_global
                norm(vec(T.linear .- T_global.linear)) +
                norm(T.translation - T_global.translation)
            end
            for method in keys(Ts)
                push!(
                    rows,
                    (;
                        angle = angles[method],
                        norm = norms[method],
                        frob = frobs[method],
                        method,
                        restarts = nr,
                    ),
                )
            end
        end
    end
    DataFrame(rows)
end

quartile1(x) = quantile(x, 1//4)
quartile3(x) = quantile(x, 3//4)

grouped = groupby(data, [:restarts, :method])
stats = combine(
    grouped,
    :angle => mean,
    :angle => std,
    :angle => maximum,
    :norm => mean,
    :norm => std,
    :norm => maximum,
    :frob => mean,
    :frob => std,
    :frob => maximum,
    :angle => quartile1,
    :angle => quartile3,
    :angle => median,
    :angle => minimum,
)

let
    fig = Figure()
    ax = Axis(fig[1, 1]; xscale = log10, title = "$id_Y \u2194 $id_X")
    methods = (:gmc, :kc, :icp)
    stats_method = map(methods) do method
        subset(stats, :method => (m -> m .== method))
    end
    for (i, method) in enumerate(methods)
        s = stats_method[i]
        band!(ax, s.restarts, s.angle_quartile1, s.angle_quartile3; alpha = 0.5)
        scatterlines!(ax, s.restarts, s.angle_median; label = string(method))
    end
    axislegend(ax)
    fig
end
