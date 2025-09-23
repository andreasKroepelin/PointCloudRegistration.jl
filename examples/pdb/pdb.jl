using Revise
using BioStructures
using BioSequences
using BioAlignments
using StatsBase
using PointCloudRegistration
using GLMakie
using GLMakie.Makie.Unitful

includet("alignment.jl")

id_Y = "1ih7_A"
id_X = "1ig9_A"

# id_Y = "1q9y_A"
# id_X = "1q9x_B"

# id_Y = "1su4_A"
# id_X = "1iwo_A"

# id_Y = "1ake_A"
# id_X = "4ake_A"

# id_Y = "1ysy_A"
# id_X = "2ahm_D"

pdb_Y = retrievepdb(split(id_Y, "_")[1]; dir = tempdir())[split(id_Y, "_")[2]]
pdb_X = retrievepdb(split(id_X, "_")[1]; dir = tempdir())[split(id_X, "_")[2]]
Y, X = aligned_atoms(pdb_Y, pdb_X, notwaterselector)

Ts = (
    rmsd = register_rmsd(Y, X),
    gmc = register_gmc(Y, X),
    kc = register_kc(Y, X; restarts = RandomRestarts(100)),
)

function show_both(X, Y)
    fig = Figure()
    states = collect.([X, Y])
    ax = Axis3(fig[1, 1]; aspect = :data)
    sld = Slider(fig[2, 1]; range = 0.01:0.01:2, startvalue = 1.0)
    i_obs = Observable(false)
    points_obs = @lift states[$i_obs + 1]
    meshscatter!(ax, points_obs; markersize = 3, color = eachindex(X.points))
    t = 0
    on(events(fig).tick) do tick
        if tick.time - t > sld.value[]
            i_obs[] = !(i_obs[])
            t = tick.time
        end
    end
    fig
end

show_both(X, Ts.rmsd(Y))
show_both(X, Ts.gmc(Y))
show_both(X, Ts.kc(Y))

arrows3d(Y.points, X.points .- Ts.kc(Y).points; markerscale = 5)

norms = map(T -> norm.(X.points .- T(Y).points), Ts)
maxnorm = maximum(maximum, norms)
edges = range(0, maxnorm; length = 50)
norm_hists = map(ns -> fit(Histogram, ns, edges), norms)

mids(es) = es[begin:(end - 1)] .+ diff(es) ./ 2

let
    fig = Figure()
    methods = [:gmc, :rmsd, :kc]
    offsets = cumsum([-0.5 * maximum(norm_hists[m].weights) for m in methods])
    ax = Axis(
        fig[1, 1];
        title = "$id_Y \u2194 $id_X",
        yticks = offsets,
        yticklabelsvisible = false,
        ylabel = "frequency",
        xlabel = "distance of corresponding atoms [Å]",
    )
    for (i, method) in enumerate(methods)
        offset = offsets[i]
        xs = mids(norm_hists[method].edges...) # .* 0.1u"nm"
        band!(
            ax,
            xs,
            offset,
            offset .+ norm_hists[method].weights;
            alpha = 0.8,
            label = string(method),
        )
        lines!(ax, xs, offset .+ norm_hists[method].weights; linewidth = 3)
    end
    axislegend(ax)
    fig
end
