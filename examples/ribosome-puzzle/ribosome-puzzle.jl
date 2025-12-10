using Revise
using PointCloudRegistration
using BioStructures
using MRCFile
using EmdbHelper
using NearestNeighbors
using GLMakie
using Random

function report_iteration(; iteration, relchange, numclusters, additions)
    println(
        "Iteration $iteration -- relative change $relchange -- $numclusters clusters ($additions newly added).",
    )
end

target_mrc = EmdbHelper.load_map("60060");
target = thin_droplowweight(PointCloud(target_mrc; threshold = 0.0244f0), .001)
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
        markersize = resolution .* target_thinned.weights
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
    # PointCloud(coordarray(chain))
    for chain in source_chains
];
sort!(sources, by = pc -> length(pc.points), rev = true);
sources = sources[1:2]
sources = [PointCloudRegistration.rand_transformation(Random.default_rng(), src, src)(src) for src in sources]
# fake_target = mapreduce(identity, vcat, sources)

let
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    sl = Slider(fig[2, 1]; range = eachindex(sources))
    # meshscatter!(
    #     ax,
    #     fake_target.points;
    #     markersize = 1,
    #     color = :lightgray,
    #     label = "all sources",
    # )
    meshscatter!(
        ax,
        @lift(sources[$(sl.value)].points);
        markersize = 3,
        # label = @lift(string("source ", $(sl.value))),
    )
    # axislegend(ax)
    fig
end

prepd_target = prepare_target_kc(target_thinned; scale = [2resolution, resolution]);

Ts = map(sources) do source
    Threads.@spawn begin
        T = register_kc(source, prepd_target; restarts = RandomRestarts(500), smm = Smm(50))
        @info "next source" T
        T
    end
end .|> fetch

Ts = let Ts = []
    artificial_target = target_thinned
    for source in sources
        target_prepd = prepare_target_kc(artificial_target; scale = [2resolution, resolution])
        T = register_kc(source, target_prepd; restarts = RandomRestarts(500), smm = Smm(50))
        push!(Ts, T)
        @info "next source" T
        anti_source = PointCloud(T(source).points, fill(-1f1, length(source.points)))
        artificial_target = vcat(artificial_target, anti_source)
    end
    Ts
end

Ts, artificial_targets = let Ts = []
    atargets = []
    artificial_target = target_thinned
    for source in sources
        target_prepd = prepare_target_kc(artificial_target; scale = [2resolution, resolution])
        T = register_kc(source, target_prepd; restarts = RandomRestarts(500), smm = Smm(50))
        push!(Ts, T)
        @info "next source" T
        new_target_points = similar(artificial_target.points) |> empty!
        new_target_weights = similar(artificial_target.weights) |> empty!
        tree = KDTree(T(source).points)
        for (point, weight) in zip(artificial_target.points, artificial_target.weights)
            if isempty(inrange(tree, point, 1.0f0 * resolution))
                push!(new_target_weights, weight)
                push!(new_target_points, point)
            end
        end
        artificial_target = PointCloud(new_target_points, new_target_weights)
        push!(atargets, artificial_target)
        @info "shrinked target" size(artificial_target)
    end
    Ts, atargets
end

let
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    meshscatter!(
        ax,
        target_thinned.points;
        markersize = resolution .* target_thinned.weights
            ./ maximum(target_thinned.weights),
        color = :lightgray,
        label = "target"
    )
    for (i, (source, T)) in enumerate(zip(sources, Ts))
        meshscatter!(ax, source.points, markersize = 3, label = string(i), visible = false)
        meshscatter!(ax, T(source).points, markersize = 3, label = string("T", i), visible = false)
    end
    axislegend(ax, nbanks = 5)
    fig
end

let
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    sl = Slider(fig[2, 1]; range = eachindex(artificial_targets))
    factor = resolution / maximum(target_thinned.weights)
    target_plt = meshscatter!(
        ax,
        target_thinned.points;
        markersize = factor .* target_thinned.weights,
        color = :lightgray,
        label = "target"
    )
    source_plt = meshscatter!(ax, PointCloud{3, Float32}().points, markersize = 3)
    on(sl.value) do i
        Makie.update!(target_plt; arg1 = artificial_targets[i].points, markersize = factor .* artificial_targets[i].weights)
        Makie.update!(source_plt; arg1 = Ts[i](sources[i]).points)
    end
    # for (i, (source, T)) in enumerate(zip(sources, Ts))
    #     meshscatter!(ax, source.points, markersize = 3, label = string(i), visible = false)
    #     meshscatter!(ax, T(source).points, markersize = 3, label = string("T", i), visible = false)
    # end
    # axislegend(ax, nbanks = 5)
    fig
end
