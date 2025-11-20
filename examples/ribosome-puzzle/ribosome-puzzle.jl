using Revise
using PointCloudRegistration
using BioStructures
using MRCFile
using EmdbHelper
using GLMakie

function report_iteration(; iteration, relchange, numclusters, additions)
    println(
        "Iteration $iteration -- relative change $relchange -- $numclusters clusters ($additions newly added).",
    )
end

target_mrc = EmdbHelper.load_map("60060");
target = PointCloud(target_mrc; threshold = 0.0244f0)
resolution = 10f0
target_thinned = thin_dpmeans(target, resolution; report_iteration)

let
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    lims = map(r -> (first(r), last(r)), voxelaxes(header(target_mrc)))
    volume!(
        ax,
        lims...,
        target_mrc.data;
        algorithm = :iso,
        isovalue = 0.0244f0,
        isorange = 0.001f0
    )
    meshscatter!(
        ax,
        target_thinned.points;
        markersize = 15 .* target_thinned.weights
            ./ maximum(target_thinned.weights),
        color = :orange,
    )
    fig
end

source_pdb = retrievepdb("8zfg")
source_chains = source_pdb |> chains |> values

function isprotein(chain)
    any(r -> r.name in ("ALA", "GLY"), values(residues(chain)))
end
function isrna(chain)
    any(r -> r.name in ("A", "C"), values(residues(chain)))
end
function sensible_selector(chain)
    if isprotein(chain)
        calphaselector
    elseif isrna(chain)
        Base.Fix2(atomnameselector, tuple("C3'"))
    else
        error("Could not recognize type of chain. ", chain)
    end
end

sources = [
    PointCloud(coordarray(chain, sensible_selector(chain)))
    for chain in source_chains
];
sort!(sources, by = pc -> length(pc.points), rev = true);

Ts = let Ts = []
    artificial_target = target_thinned
    for source in sources
        target_prepd = prepare_target_kc(artificial_target; scale = [15f0, 10f0])
        T = register_kc(source, target_prepd; restarts = RandomRestarts(500), smm = Smm(50))
        push!(Ts, T)
        @info "next source" T
        anti_source = PointCloud(T(source).points, fill(-1f1, length(source.points)))
        artificial_target = vcat(artificial_target, anti_source)
    end
    Ts
end

let
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    meshscatter!(
        ax,
        target_thinned.points;
        markersize = 15 .* target_thinned.weights
            ./ maximum(target_thinned.weights),
        color = :lightgray,
        label = "target"
    )
    for (i, (source, T)) in enumerate(zip(sources, Ts))
        meshscatter!(ax, source.points, markersize = 5, label = string(i), visible = false)
        meshscatter!(ax, T(source).points, markersize = 5, label = string("T", i), visible = false)
    end
    axislegend(ax, nbanks = 5)
    fig
end
