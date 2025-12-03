using Revise
using BioStructures
using BioSequences
using BioAlignments
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

includet("alignment.jl")

# id_Y = "1ih7_A"
# id_X = "1ig9_A"

# id_Y = "1q9y_A"
# id_X = "1q9x_B"

id_Y = "1su4_A"
id_X = "1iwo_A"

# id_Y = "1ake_A"
# id_X = "4ake_A"

# id_Y = "1ysy_A"
# id_X = "2ahm_D"

pdb_Y = retrievepdb(split(id_Y, "_")[1])[split(id_Y, "_")[2]]
pdb_X = retrievepdb(split(id_X, "_")[1])[split(id_X, "_")[2]]
Y, X = aligned_atoms(pdb_Y, pdb_X, notwaterselector)

pdb_ids = "1IWO,1KJU,1SU4,1T5S,1T5T,1VFP,1WPG,1XP5,2AGV,2BY4,2C88,2C8K,2C8L,2C9M,2DQS,2EAR,2EAT,2EAU,2O9J,2OA0,2YFY,2ZBD,2ZBE,2ZBF,2ZBG,3AR2,3AR3,3AR4,3AR5,3AR6,3AR7,3AR8,3AR9,3B9B,3B9R,3BA6,3FGO,3FPB,3FPS,3J7T,3N5K,3N8G,3NAL,3NAM,3NAN,3W5A,3W5B,3W5C,3W5D,4H1W,4J2T,4KYT,4UU0,4UU1,4XOU,4Y3U,4YCL,4YCM,4YCN,5A3Q,5A3R,5A3S,5NCQ,5XA7,5XA8,5XA9,5XAA,5XAB,6HEF,6YAA,6YSO,4BEW,8OWA,8OWL" |> Base.Fix2(split, ",")
pdbs = [retrievepdb(id; dir = "data")["A"] for id in pdb_ids]
pointclouds = [PointCloud(coordarray(pdb, calphaselector)) for pdb in pdbs if countatoms(pdb, calphaselector) == 994]

target = pointclouds[1];
Ts_per_method = (
    rmsd = [register_rmsd(source, target) for source in pointclouds],
    gmc = [register_gmc(source, target) for source in pointclouds],
);
registered_pointclouds_per_method = map(Ts_per_method) do Ts
    [T(pc) for (T, pc) in zip(Ts, pointclouds)]
end;

function show_all(pcs)
    fig = Figure()
    states = collect.([X, Y])
    ax = Axis3(fig[1, 1]; aspect = :data)
    sl = Slider(fig[2, 1]; range = eachindex(pcs))
    meshscatter!(ax, @lift(pcs[$(sl.value)].points); markersize = 3, color = eachindex(pcs[1].points))
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

qs = [.75, .5, .25]
normqs = map(norms) do ns
    stack(eachrow(ns)) do row
        [quantile(row, q) for q in qs]
    end |> permutedims
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
    GLMakie.activate!(); display(fig)
end

let
    methods = [:rmsd, :gmc]
    colorrange_hi = maximum(maximum, norms[methods])
    cmap = :managua
    fig = Figure(size = (400, 400))
    axs = [Axis3(fig[1, i]; aspect = :data, title = uppercase(string(methods[i])), [Symbol(d, :ticklabelsvisible) => false for d in [:x, :y, :z]]...) for i in (1, 2)]
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
        meshscatter!(ax, X.points; markersize = max.(.2, 3 .* norms[method] ./ colorrange_hi))
    end
    # arrows3d!(ax, Ts.gmc(Y).points, X.points .- Ts.gmc(Y).points; markerscale = .1, lengthscale = 1)
    # GLMakie.activate!(); display(fig)
    GLMakie.activate!()
    save("../../paper/bioinformatics/src/img/pdb-connections.png", fig; px_per_unit = 3)
end

norms = map(T -> norm.(X.points .- T(Y).points), Ts)
maxnorm = maximum(maximum, norms)
edges = range(0, maxnorm; length = 60)
norm_hists = map(ns -> fit(Histogram, ns, edges), norms)

let
    fig = Figure(size = (400, 350))
    methods = [:gmc, :mad, :kc, :rmsd]
    offset = -.05
    ax = Axis(
        fig[1, 1];
        title = "$id_Y \u2194 $id_X",
        yticks = offset .* eachindex(methods),
        yticklabelsvisible = false,
        ylabel = "frequency",
        xlabel = "distance of corresponding atoms [Å]",
    )
    for (i, method) in enumerate(methods)
        density!(ax, norms[method]; label = uppercase(string(method)), alpha = 0.8, strokearound = true, strokewidth=2, offset = offset * i, bandwidth = .5)
    end
    axislegend(ax,valign=:bottom)
    # GLMakie.activate!(); display(fig)
    CairoMakie.activate!(pdf_version = "1.5")
    save("../../paper/bioinformatics/src/img/pdb-distances.pdf", fig)
end

data = let # X = PCReg.rand_transformation(Xoshiro(-2), Y, X)(X)
    scale = DownTo(PCReg.avg_nn_dist(X))
    prepd_X = prepare_target_kc(X)
    rr() = RandomRestarts(100_000, Xoshiro(1))
    Ts_global = (
        gmc = register_gmc(Y, X; scale, restarts = rr()),
        kc = register_kc(Y, prepd_X; restarts = rr()),
        icp = register_icp(Y, X; restarts = rr()),
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
                gmc = register_gmc(Y, X; scale, restarts),
                kc = register_kc(Y, prepd_X; restarts),
                icp = register_icp(Y, X; restarts),
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
