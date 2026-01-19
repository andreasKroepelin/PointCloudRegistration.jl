# # Image registration
# In this example, we demonstrate how one can perform rigid registration of
# images with this package.
# We will use the BioTISR data set containing time lapse micrographs of
# cellular structures ([Zenodo](https://zenodo.org/records/13843670)).
# In the *Mitochondria* data set, we find 20 images per cell that show a
# specific scene evolving over time.
# The camera is fixed so that we can consider the images perfectly registered,
# providing us with a ground truth.
#
# We will apply random rigid transformation to the images and afterwards try
# to undo them via rigid registration of appropriate point clouds.

# ## Packages

using Revise #src
using PointCloudRegistration
using Makie
import GLMakie
using Statistics
using LinearAlgebra
using StaticArrays
using Rotations
using CoordinateTransformations
using DimensionalData
import Images: warp, mapwindow, binarize, Otsu, RGB, paddedviews, colorview, zeroarray
import Images
using PythonCall
using Chain


# ## Loading the images
# Assuming that the file `BioTISR_Mitochondria.zip` is downloaded and extracted
# in the working directory, we can access the images as `.mrc` files.
# However, the files seem to have issues, which is why MRCFile.jl cannot read
# them.
# Instead, we use the Python package mrcfile via PythonCall.jl:

const mrcfile = pyimport("mrcfile")
mrc = pyconvert(Array, mrcfile.read("BioTISR_Mitochondria/Cell_006/SIM_gt.mrc"))

# After permuting the dimensions (Python and Julia have reverse array dimension
# orders) and slicing along the last dimension, we obtain our images.
# We also wrap them with `DimArray`s from DimensionalData.jl to track
# transformations later.

orig_imgs = map(eachslice(permutedims(mrc, (3, 2, 1)); dims = 3)) do slice
    DimArray(slice, (X, Y))
end

# We can use Makie to display them:

image(orig_imgs[1])

# ## Transforming the images
# To make the registration interesting, we first randomly transform our images.
# We need a rigid transformation for every image:

trueinvTs = [
    Translation(1000 * randn(SVector{2})) ∘ LinearMap(rand(RotMatrix2))
    for _ in orig_imgs[2:end]
]
pushfirst!(trueinvTs, Translation(SA[0., 0.]) ∘ LinearMap(one(RotMatrix2{Float64})))

# We can apply a transformtion to an image using the `warp` function from
# Images.jl.
# Note that it interprets transformation inversely to how we do here,
# so we have to apply its inverse.
# The necessarily arising parts of the rectangular image not covered by the
# actual image we fill with zeros (third argument to `warp`).
# Additionally, we shift the intensities of each pixel such that they are
# non-negative.

function transform_img(img::DimMatrix, T)
    (xlo, xhi), (ylo, yhi) = extrema.(dims(img))
    corners = [SA[xlo, ylo], SA[xhi, ylo], SA[xlo, yhi], SA[xhi, yhi]]
    newlo, newhi = PointCloudRegistration.bbox(T.(corners))
    xs = newlo[1]:newhi[1]
    ys = newlo[2]:newhi[2]
    invT = inv(T)
    raw = map(Iterators.product(xs, ys)) do (Tx, Ty)
        x, y = invT(SA[Tx, Ty])
        if xlo <= x <= xhi && ylo <= y <= yhi
            img[X = Near(x), Y = Near(y)]
        else
            zero(eltype(img))
        end
    end
    DimArray(raw, (X(xs), Y(ys)))
end

imgs = map(orig_imgs, trueinvTs) do orig_img, trueinvT
    img = orig_img .- minimum(orig_img)
    transform_img(img, trueinvT)
end;

# Translation is represented by the images being `OffsetArray`s:

typeof(imgs[1])

# It looks like this:

image(imgs[1])

# ## Conversion to point clouds
# To apply this package, we need to work with point clouds.
# The naive first step is to place a point at every pixel and use the pixel's
# intensity as the weight.
# This is done by the function `density2pointcloud` provided by this package.
# It also handles the offset axes.

pointclouds_full = map(density2pointcloud, imgs);

# We can show the pointcloud on top of the image.
# To be able to see anything at all, we need to zoom into a fairly small region.

let
    plt = image(imgs[1]; axis = (;aspect = DataAspect()))
    plot!(plt.axis, pointclouds_full[1]; color = :lime)
    limits!(plt.axis, (630, 680), (790, 850))
    plt
end

# As we can see, there are very many points that don't contribute to a valuable
# description of the image.
# Hence, let us drop all points with a weight below the 95 % quantile per point
# cloud.

pointclouds_foreground = map(pointclouds_full) do pc
    PointCloudRegistration.drop_quantile(pc, .995)
    # PointCloudRegistration.drop_proportion(pc, .4)
end;

# The same region from before now looks like this:

let
    plt = image(imgs[1]; axis = (;aspect = DataAspect()))
    plot!(plt.axis, pointclouds_foreground[1]; color = :lime)
    limits!(plt.axis, (630, 680), (790, 850))
    plt
end

# Lastly, we can observe that the "resolution" of our point clouds is
# unnecessarily high.
# We can thin them to a nearest neighbor distance of roughly 20 (pixels).

pointclouds_thinned = map(pointclouds_foreground) do pc
    thin_to_distance(pc, 20.)
end;

# This looks much less cluttered now:

let
    plt = image(imgs[1]; axis = (;aspect = DataAspect()))
    plot!(plt.axis, pointclouds_thinned[1]; color = :lime)
    limits!(plt.axis, (630, 680), (790, 850))
    plt
end

# We can take a look at the full point cloud as well:

let
    plt = image(imgs[1]; axis = (;aspect = DataAspect()))
    plot!(plt.axis, pointclouds_thinned[1]; color = :lime)
    plt
end

# # Registering the point clouds.
# Since there is no inherently true reference coordinate system, we might as
# well pick one of the point clouds as the targets.

target = pointclouds_thinned[1]
prepd_target = prepare_target_kc(target);

Ts = map(pointclouds_thinned) do src
    Threads.@spawn begin
        register_kc(src, prepd_target; smm = Smm(50), restarts = RandomRestarts(50))
    end
end .|> fetch

residualTs = Ts .∘ trueinvTs .∘ (inv(trueinvTs[1]), )

reg_imgs = transform_img.(imgs, Ts);

reg_pointclouds = map((T, pc) -> T(pc), Ts, pointclouds_thinned);

let
    fig = Figure()
    ax = Axis(fig[1, 1]; aspect = DataAspect())
    sl = Slider(fig[1, 2]; range = eachindex(imgs), horizontal = false)
    image!(ax, @lift(reg_imgs[$(sl.value)]))
    plot!(ax, @lift(reg_pointclouds[$(sl.value)]); color = :lime)
    fig
end

let
    fig = Figure()
    ax = Axis(fig[1, 1]; aspect = DataAspect())
    idx = 19
    image!(ax, orig_imgs[idx]; colormap = :greens)
    image!(ax, transform_img(imgs[idx], Ts[idx]); colormap = :blues, alpha = .3)
    fig
end

let
    limits = ((500, 1000), (800, 980))
    sz = map(splat(-) ∘ reverse, limits)
    fig = Figure()
    for img_idx in eachindex(orig_imgs)
        ax = Axis(
            fig[Tuple(CartesianIndices((5, 4))[img_idx])...];
            aspect = DataAspect(),
            limits,
            width = sz[1] ÷ 2,
            height = sz[2] ÷ 2,
        )
        hidedecorations!(ax)
        image!(ax, orig_imgs[img_idx]; colormap = :greens)
        image!(ax, transform_img(imgs[img_idx], Ts[img_idx]); colormap = :blues, alpha = .3)
    end
    rowgap!(fig.layout, 5)
    colgap!(fig.layout, 5)
    resize_to_layout!(fig)
    # save("../../paper/bioinformatics/src/img/biotisr.png", fig)
    fig
end
