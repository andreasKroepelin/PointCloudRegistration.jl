using Revise
using BioStructures
using BioSequences
using BioAlignments
using PointCloudRegistration
using Makie, GLMakie

includet("alignment.jl")

# pdb_Y = retrievepdb("1su4"; dir = tempdir())["A"]
# pdb_X = retrievepdb("1iwo"; dir = tempdir())["A"]
pdb_Y = retrievepdb("1ake"; dir = tempdir())["A"]
pdb_X = retrievepdb("4ake"; dir = tempdir())["A"]
Y, X = aligned_atoms(pdb_Y, pdb_X, notwaterselector)
iY, iX = guess_correspondences(Y, X)
twc = register_gmc(Y, X; restarts = 100)
twc.transformation.linear
Y_, X_ = Y[iY], X[iX]
twc_ = register_gmc(Y_, X_; restarts = 100)
twc_.transformation.linear

# transformation(Y)

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

show_both(X, twc.transformation(Y))
show_both(X, twc_.transformation(Y))
