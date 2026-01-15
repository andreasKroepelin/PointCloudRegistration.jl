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
# Let us load some packages first.

using Revise # hide
using PointCloudRegistration
using BioStructures
using MRCFile
using EmdbHelper # a local helper package to download data from the EMDB
using NearestNeighbors
using DimensionalData
using GLMakie
using Random
using Statistics
import LinearAlgebra: norm_sqr

# ## The target

# First, we obtain the target density from the EMDB.
# The helper package `EmdbHelper` makes this very convenient and provides us
# with an `MRCData` object from MRCFiles.jl.
# We convert it into a `DimArray` from DimesionalData.jl such that it is easier
# to handle.

target_mrc = EmdbHelper.load_map("49998");
target_dimarr = EmdbHelper.mrc2dimarr(target_mrc);
show(IOContext(stdout, :limit => true), MIME"text/plain"(), target_dimarr) # hide

# We can get a first visual impression using a volume plot.
# For this, we make use of the *author thresold* that is provided by the EMDB
# ([here](https://www.ebi.ac.uk/emdb/EMD-49998?tab=experiment) under Map/Contour
# list), telling us what threshold the authors of the EMDB entry used to discern
# background and structure.

author_threshold = 0.046f0
volume(
    target_dimarr;
    algorithm = :iso,
    isovalue = author_threshold + .01f0,
    isorange = .01f0,
)

# Next, we convert the density map into a point cloud.
# This happens in two steps:
# First, we use the function `density2pointcloud` from PointCloudRegistration.jl
# to do a very naive conversion.
# We simply place a point at every voxel and use the voxel's value as the
# weight.
# We additionally drop all points that have a weight below `author_threshold`
# to again follow the practise of the authors observing the structure.

target_full = density2pointcloud(target_dimarr)
target = drop_low_weight(target_full; threshold = author_threshold)

# The second step is to perform thinning on the target to reduce the
# computational burden later on.
# We define a target resolution of 5 Å and thin our pointcloud to that nearest
# neighbor distance:

resolution = 5.0f0
target_thinned = thin_to_distance(target, resolution)

# We can check that it worked by measuring the average nearest neighbor distance
# in the result.

PointCloudRegistration.avg_nn_dist(target_thinned)

# We now have a much smaller point cloud

length(target_thinned.points) / length(target_full.points)

# that still describes the full ribosome well, as we can see when plotting
# `target_dimarr` and `target_thinned` together:

let
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    volume!(
        ax,
        target_dimarr;
        algorithm = :iso,
        isovalue = author_threshold + .01f0,
        isorange = .01f0,
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
# two point clouds.

threeprimeselector(atom) = atomnameselector(atom, tuple("C3'"))
sources = map(source_chains) do chain
    PointCloud(coordarray(chain, threeprimeselector))
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
    meshscatter!(
        ax,
        target_thinned.points;
        markersize = resolution .* target_thinned.weights
            ./ maximum(target_thinned.weights),
        color = :lightgray,
        label = "full ribosome",
    )
    meshscatter!(ax, sources.rRNA23S.points; markersize = 3, label = "23S")
    meshscatter!(ax, sources.rRNA16S.points; markersize = 3, label = "16S")
    axislegend(ax)
    fig
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

prepd_target = prepare_target_kc(target_thinned);

Ts = map(randomized_sources) do source
    register_kc(source, prepd_target; restarts = RandomRestarts(500))
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

T_16S_better = register_kc(
    randomized_sources.rRNA16S,
    target_without_23S;
    restarts = RandomRestarts(500)
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
# To finish off, we can also compute how close we got to the original PDB based
# point clouds in terms of the RMSD:

#-

rmsds = map(Ts, randomized_sources, sources) do T, randomized_source, source
    a = source.points
    b = T(randomized_source).points
    sqrt(mean(norm_sqr, a .- b))
end

# So we achieved a very close result, far below our chosen resolution.
