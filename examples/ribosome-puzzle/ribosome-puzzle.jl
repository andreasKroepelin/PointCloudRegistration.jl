# # Fitting a point cloud into a density map
# In this example, we will demonstrate how to deal with structures that are not
# directly available as point clouds but rather as density maps.
#
# This prominently occurs in cryo electron microscopy.
# We will use a cryo density describing the *Methanosarcina acetivorans* 70S
# ribosome, available on the EMDB with id 49998, as the target.
# For the source, we will use the 23S and 16S ribosomal RNAs, available on the
# PDB with id 9o17.
#
# The general strategy will be to convert the target density into a point cloud,
# then extract the source point clouds and randomly transform then, such that
# we can then do the actual registration and assess if we were successful.
#
# ## Packages
# Let us set up the environment first.

using Revise #hide
using PointCloudRegistration
using BioStructures
using MRCFile
using EmdbHelper # a local helper package to download data from the EMDB
using NearestNeighbors
using DimensionalData
using GLMakie
using Random #hide
using Statistics
import LinearAlgebra: norm_sqr

Random.seed!(3) #hide

# ## The target

# First, we obtain the target density from the EMDB.
# The helper package `EmdbHelper` makes this very convenient and provides us
# with an `MRCData` object from MRCFiles.jl.
# We convert it into a `DimArray` from DimesionalData.jl such that it is easier
# to handle and also attach units of Angstrom (Å) to the axes.

target_mrc = EmdbHelper.load_map("49998");
target_dimarr = EmdbHelper.mrc2dimarr(target_mrc);
show(IOContext(stdout, :limit => true), MIME"text/plain"(), target_dimarr) #hide

# A small helper function allows us to create rotating plots:

center = mean.(bounds(target_dimarr))
function record_rotating(fig, ax)
    Makie.origin!.(ax.scene.plots, center...)
    Record(fig, range(0, 2pi; length = 50); framerate = 24) do angle
        Makie.rotate!.(ax.scene.plots, angle)
    end
end

# And so this is how the target volume looks:

let
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    hidedecorations!(ax)
    hidespines!(ax)
    volume!(ax, target_dimarr)
    record_rotating(fig, ax)
end

# Next, we convert the density map into a point cloud.
# This happens in two steps:
# First, we use the function `density2pointcloud` from PointCloudRegistration.jl
# to do a very naive conversion.
# We simply place a point at every voxel and use the voxel's value as the
# weight.
# We additionally drop all points that have a weight below the *author
# threshold* that is provided by the EMDB
# ([here](https://www.ebi.ac.uk/emdb/EMD-49998?tab=experiment) under Map/Contour
# list), telling us what threshold the authors of the EMDB entry used to discern
# background and structure.

target_full = density2pointcloud(target_dimarr)
author_threshold = 0.046f0
target = drop_threshold(target_full, author_threshold)

# The second step is to perform thinning on the target to reduce the
# computational burden later on.
# We define a grid cell width of 5 Å and thin our pointcloud to a grid of that
# size:

resolution = 5.0f0 # Å
target_thinned = thin_to_distance(target, resolution)

# We can check that it worked by measuring the average nearest neighbor distance
# in the result.

PointCloudRegistration.avg_nn_dist(target_thinned)

# We now have a much smaller point cloud:

length(target_thinned.points) / length(target_full.points)

# And this is how the target looks:

let
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    hidedecorations!(ax)
    hidespines!(ax)
    plot!(ax, target_thinned)
    record_rotating(fig, ax)
end

# ## The sources
# Next, we need the 23S and 16S ribosomal RNAs as our sources.
# They are part of the PDB structure 9o17, which we can get using
# BioStructures.jl.

source_pdb = retrievepdb("9o17")

# On [this page](https://www.ebi.ac.uk/pdbe/entry/pdb/9o17?activeTab=macromolecules),
# we can see that the 23S rRNA has chain ID `BA` and the 16S rRNA has chain ID
# `AA`.

chain_ids = (rRNA23S = "BA", rRNA16S = "AA")
source_chains = map(key -> chains(source_pdb)[key], chain_ids)

# Let us extract only the `C3'` atoms from both chains and collect them into
# two point clouds, again adding Angstrom units.

threeprimeselector(atom) = atomnameselector(atom, tuple("C3'"))
sources = map(source_chains) do chain
    PointCloud(coordarray(chain, threeprimeselector) #= .* Å =#)
end;
#-
sources.rRNA23S
#-
sources.rRNA16S

# In principle, we are good to go now and we could perform the rigid
# registration.
# However, it would be a bit pointless, since the PDB data and the EMDB density
# are already perfectly aligned since the former is computed from the latter:

let
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    hidedecorations!(ax)
    hidespines!(ax)
    plot!(ax, target_thinned; color = :lightgray, label = "full ribosome")
    plot!(ax, sources.rRNA23S; label = "23S")
    plot!(ax, sources.rRNA16S; label = "16S")
    axislegend(ax)
    record_rotating(fig, ax)
end

# Instead, we draw a random rigid transformation for every source:

trueinvTs = map(PointCloudRegistration.rand_transformation, sources)

# ... and use it to transform the two point clouds.

randomized_sources = map((src, invT) -> invT(src), sources, trueinvTs);

# ## The registration
# We can now finally demonstrate how to fit the rRNA sources into the full
# ribosome target.
# Since we obviously have no correspondence information, we use the kernel
# correlation method.

target_preparation = prepare_target_kernelcorrelation(target_thinned);

Ts = map(randomized_sources) do source
    rigid_registration(
        source,
        target_thinned,
        KernelCorrelationMM(restarts = RandomRestarts(1000));
        target_preparation
    )
end

# We should now see that the `Ts` are the inverses of `trueinvTs` and their
# composition is the identity transformation:

residualTs = map(∘, Ts, trueinvTs);
#-
residualTs.rRNA23S.linear
#-
residualTs.rRNA23S.translation
#-
residualTs.rRNA16S.linear
#-
residualTs.rRNA16S.translation

# So this worked great for the larger rRNA.
# For the smaller one... not so much.
# The issue is that the target ribosome is just too large and it is hard to find
# this comparatively small substructure.
#
# ### The trick
# However, we can employ additional domain knowledge.
# Namely, the 23S and 16S rRNAs do not overlap.
# We can thus remove parts of the target that likely belong to the 23S rRNA
# (which we can successfully register) and superimpose the 16S rRNA with the
# remaining target.

target_without_23S = let
    tree = KDTree(Ts.rRNA23S(randomized_sources.rRNA23S).points)
    ## Find all points in `target_thinned` that are at least `resolution` away
    ## from the closest point in the 23S rRNA:
    mask = map(target_thinned.points) do point
        _nnidx, dist = nn(tree, point)
        dist > resolution
    end
    target_thinned[mask]
end

# Perform the registration:

T_16S_better = rigid_registration(
    randomized_sources.rRNA16S,
    target_without_23S,
    KernelCorrelationMM(restarts = RandomRestarts(500)),
)
Ts = (; Ts.rRNA23S, rRNA16S = T_16S_better)

# And let's do the check again:

residualTs = map(∘, Ts, trueinvTs);
#-
residualTs.rRNA16S.linear
#-
residualTs.rRNA16S.translation

# This looks great now!
#
# We can also assess the registration visually.

function truth_and_fitted(key)
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    hidedecorations!(ax)
    hidespines!(ax)
    plot!(ax, sources[key]; label = "truth")
    plot!(ax, Ts[key](randomized_sources[key]); label = "fitted")
    axislegend(ax)
    record_rotating(fig, ax)
end

# For the 23S rRNA:
truth_and_fitted(:rRNA23S)

# For the 16S rRNA:
truth_and_fitted(:rRNA16S)

# To finish off, we can also compute how close we got to the original PDB based
# point clouds in terms of the RMSD:

#-

rmsds = map(Ts, randomized_sources, sources) do T, randomized_source, source
    a = source.points
    b = T(randomized_source).points
    sqrt(mean(norm_sqr, a .- b))
end

# So we achieved a very close result, far below our chosen resolution.
