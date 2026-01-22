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
using Extents
using JLD2


# ## Loading the images
# Assuming that the file `BioTISR_Mitochondria.zip` is downloaded and extracted
# in the working directory, we can access the images as `.mrc` files.
# However, the files seem to have issues, which is why MRCFile.jl cannot read
# them.
# Instead, we use the Python package mrcfile via PythonCall.jl:

# After permuting the dimensions (Python and Julia have reverse array dimension
# orders) and slicing along the last dimension, we obtain our images.
# We also wrap them with `DimArray`s from DimensionalData.jl to track
# transformations later.

@load "biotisr-mitochondria-sim-gt-006.jld2" img_tensor

orig_imgs = map(eachslice(img_tensor; dims = 3)) do slice
    DimArray(slice, (X, Y))
end;

# We can use Makie to display them:

let
    fig = Figure()
    ax = Axis(fig[1, 1]; aspect = DataAspect())
    hidedecorations!(ax)
    idx = Observable(1)
    image!(ax, @lift(orig_imgs[$idx]))
    Record(fig, eachindex(orig_imgs); framerate = 10) do i
        idx[] = i
    end
end

# ## Transforming the images
# To make the registration interesting, we first randomly transform our images.
# We need a rigid transformation for every image but leave the first image
# untransformed to have it as a reference orientation.

trueinvTs = map(eachindex(orig_imgs)) do i
    if i == 1
        R = one(RotMatrix2{Float64})
        t = zero(SVector{2})
    else
        R = rand(RotMatrix2)
        t = 1000 * randn(SVector{2})
    end
    AffineMap(R, t)
end

# The following function transforms an image represented as a `DimArray` by some
# transformation.
# We first calculate the necessary rectangular bounding box and then fill each
# pixel by looking up the corresponding pixel in the original image (hence the
# inverted transformation).
# Parts of the image without a correspondence in the original are filled with
# zeros.

function transform_img(img::DimMatrix, T)
    xlo, xhi  = DimensionalData.bounds(img, X)
    ylo, yhi  = DimensionalData.bounds(img, Y)
    corners = [SA[xlo, ylo], SA[xhi, ylo], SA[xlo, yhi], SA[xhi, yhi]]
    newlo, newhi = PointCloudRegistration.bbox(T.(corners))
    invT = inv(T)
    [
        let
            x, y = invT(SA[Tx, Ty])
            if xlo <= x <= xhi && ylo <= y <= yhi
                img[X = Near(x), Y = Near(y)]
            else
                zero(eltype(img))
            end
        end
        for Tx in X(newlo[1]:newhi[1]), Ty in Y(newlo[2]:newhi[2])
    ]
end

# We apply the function to all our images and additionally shift the intensities
# of each pixel such that they are non-negative.

imgs = map(orig_imgs, trueinvTs) do orig_img, trueinvT
    img = orig_img .- minimum(orig_img)
    transform_img(img, trueinvT)
end;

# It looks like this:

image(imgs[2]; axis = (;aspect = DataAspect()))

# ## Conversion to point clouds
# To apply this package, we need to work with point clouds.
# The naive first step is to place a point at every pixel and use the pixel's
# intensity as the weight.
# This is done by the function `density2pointcloud` provided by this package.
# It also handles the special axes.

pointclouds_full = map(density2pointcloud, imgs);

# We can show the pointcloud on top of the image.
# To be able to see anything at all, we need to zoom into a fairly small region.

let
    fig = Figure()
    ax = Axis(fig[1, 1]; aspect = DataAspect(), limits = ((630, 680), (790, 850)))
    image!(ax, imgs[1])
    plot!(ax, pointclouds_full[1]; color = :lime)
    fig
end

# As we can see, there are very many points that don't contribute to a valuable
# description of the image.
# Hence, let us drop all points with a weight below the 99.5 % quantile per
# point cloud.

pointclouds_foreground = map(pointclouds_full) do pc
    drop_quantile(pc, .995)
end;

# The same region from before now looks like this:

let
    fig = Figure()
    ax = Axis(fig[1, 1]; aspect = DataAspect(), limits = ((630, 680), (790, 850)))
    image!(ax, imgs[1])
    plot!(ax, pointclouds_foreground[1]; color = :lime)
    fig
end

# Lastly, we can observe that the "resolution" of our point clouds is
# unnecessarily high.
# We can thin them to a nearest neighbor distance of roughly 10 (pixels).

pointclouds_thinned = map(pointclouds_foreground) do pc
    thin_to_distance(pc, 10.)
end;

# This looks much less cluttered now:

let
    fig = Figure()
    ax = Axis(fig[1, 1]; aspect = DataAspect(), limits = ((630, 680), (790, 850)))
    image!(ax, imgs[1])
    plot!(ax, pointclouds_thinned[1]; color = :lime)
    fig
end

# We can take a look at the full point cloud as well:

let
    fig = Figure()
    ax = Axis(fig[1, 1]; aspect = DataAspect())
    image!(ax, imgs[1])
    plot!(ax, pointclouds_thinned[1]; color = :lime)
    fig
end

# # Registering the point clouds.
# As noted above, we use the first point cloud as our target reference.
# Since we don't know correspondences between the point clouds, we use the
# kernel correlation method.
# We can speed up the registration by preparing the karget for computing the
# kernel correlation.

target = pointclouds_thinned[1]
prepd_target = prepare_target_kc(target);
Ts = map(pointclouds_thinned) do src
    Threads.@spawn register_kc(src, prepd_target; restarts = RandomRestarts(100))
end .|> fetch

# We can check if the registration worked by comparing `Ts` with `trueinvTs`
# from above.
# In case of success, they should cancel each other, i.e. their composition
# should be the identity transformation with rotation angle and translation norm
# zero.

residualTs = Ts .∘ trueinvTs
rotation_angles = [rad2deg(rotation_angle(resT.linear)) for resT in residualTs]
#-
translation_norms = [norm(resT.translation) for resT in residualTs]

# Let us use the `Ts` to transform all our images as well as point clouds.

reg_imgs = transform_img.(imgs, Ts);
reg_pointclouds = map((T, pc) -> T(pc), Ts, pointclouds_thinned);

# We can check the registration visually by animating the time lapse with the
# registered images and point clouds:

let
    fig = Figure()
    ax = Axis(fig[1, 1]; aspect = DataAspect())
    hidedecorations!(ax)
    # sl = Slider(fig[1, 2]; range = eachindex(imgs), horizontal = false) #src
    idx = Observable(1)
    image!(ax, @lift(reg_imgs[$idx]))
    plot!(ax, @lift(reg_pointclouds[$idx]); color = :lime)
    resize_to_layout!(fig)
    Record(fig, eachindex(imgs); framerate = 10) do i
        idx[] = i
    end
    # fig #src
end

# To analyse the results in detail, let us write a short helper function
# that takes to gray-scale images and produces a color image with the first
# input as the green channel and the second one as the blue channel.

function diffview(img1, img2)
    newbounds = Extents.bounds(Extents.union(extent(img1), extent(img2)))
    greens, blues = map([img1, img2]) do img
        xlo, xhi  = DimensionalData.bounds(img, X)
        ylo, yhi  = DimensionalData.bounds(img, Y)
        maxintensity = maximum(img)
        [
            if xlo <= x <= xhi && ylo <= y <= yhi
                img[X = Near(x), Y = Near(y)] / maxintensity
            else
                zero(eltype(img))
            end
            for x in X(range(newbounds.X...)), y in Y(range(newbounds.Y...))
        ]
    end
    Makie.RGB.(0, greens, blues)
end

# We can use this function to view a small section of all twenty time lapse
# images where we show the original image (green channel) and the randomly
# transformed and then registered image (blue channel) together.

let
    limits = ((500, 1000), (800, 980))
    sz = map(splat(-) ∘ reverse, limits)
    fig = Figure()
    for idx in eachindex(orig_imgs)
        ax = Axis(
            fig[Tuple(CartesianIndices((5, 4))[idx])...];
            aspect = DataAspect(),
            limits,
            width = sz[1] ÷ 2,
            height = sz[2] ÷ 2,
        )
        hidedecorations!(ax)
        image!(ax, diffview(orig_imgs[idx], transform_img(imgs[idx], Ts[idx])))
    end
    rowgap!(fig.layout, 5)
    colgap!(fig.layout, 5)
    resize_to_layout!(fig)
    save("../../paper/bioinformatics/src/img/biotisr.png", fig) #src
    fig
end
