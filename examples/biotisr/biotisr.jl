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
    for _ in orig_imgs
]

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

# Let us create a small helper function for plotting point clouds:

function plot_pointcloud(pc; kwargs...)
    sizefactor = PointCloudRegistration.avg_nn_dist(pc) / maximum(pc.weights)
    scatter(pc.points; markersize = sizefactor .* pc.weights, markerspace = :data, kwargs...)
end

function plot_pointcloud!(axis, pc; kwargs...)
    sizefactor = PointCloudRegistration.avg_nn_dist(pc) / maximum(pc.weights)
    scatter!(axis, pc.points; markersize = sizefactor .* pc.weights, markerspace = :data, kwargs...)
end

# We can use it to show the pointcloud on top of the image.
# To be able to see anything at all, we need to zoom into a fairly small region.

let
    plt = image(imgs[1]; axis = (;aspect = DataAspect()))
    plot_pointcloud!(plt.axis, pointclouds_full[1]; color = :lime)
    limits!(plt.axis, (-400, -300), (1320, 1390))
    plt
end

# As we can see, there are very many points that don't contribute to a valuable
# description of the image.
# Hence, let us drop all points with a weight below the 95 % quantile per point
# cloud.

pointclouds_foreground = map(pointclouds_full) do pc
    PointCloudRegistration.drop_quantile(pc, .95)
end;

# The same region from before now looks like this:

let
    plt = image(imgs[1]; axis = (;aspect = DataAspect()))
    plot_pointcloud!(plt.axis, pointclouds_foreground[1]; color = :lime)
    limits!(plt.axis, (-400, -300), (1320, 1390))
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
    plot_pointcloud!(plt.axis, pointclouds_thinned[1]; color = :lime)
    limits!(plt.axis, (-400, -300), (1320, 1390))
    plt
end

# We can take a look at the full point cloud as well:

let
    plt = image(imgs[1]; axis = (;aspect = DataAspect()))
    plot_pointcloud!(plt.axis, pointclouds_thinned[1]; color = :lime)
    plt
end

# # Registering the point clouds.
# Since there is no inherently true reference coordinate system, we might as
# well pick one of the point clouds as the targets.

target = pointclouds_thinned[1]
prepd_target = prepare_target_kc(target)

Ts = map(pointclouds_thinned) do src
    Threads.@spawn begin
        register_kc(src, prepd_target; smm = Smm(50), restarts = RandomRestarts(50))
    end
end .|> fetch

residualTs = Ts .∘ trueinvTs .∘ (inv(trueinvTs[1]), )

reg_imgs = transform_img.(imgs, Ts)


function run_experiment(distances)
    results = []
    for distance in distances
        @info "Running next experiment" distance
        time = @elapsed begin
            pointclouds = map(imgs) do img
                Threads.@spawn @chain img begin
                    density2pointcloud
                    drop_low_weight(_; proportion = .4)
                    thin_to_distance(_, distance)
                end
            end .|> fetch
            resolution = minimum(pc -> PointCloudRegistration.avg_nn_dist(pc), pointclouds)
            target = pointclouds[1]
            # prepd_target = prepare_target_kc(target; scale = [2, 1] .* resolution);
            prepd_target = prepare_target_kc(target)
            Ts = map(pointclouds) do src
                Threads.@spawn begin
                    register_kc(src, prepd_target; smm = Smm(50), restarts = RandomRestarts(50))
                end
            end .|> fetch
        end
        residualTs = Ts .∘ invTs .∘ (inv(invTs[1]), )
        push!(results, (; distance, time, resolution, pointclouds, Ts, residualTs))
    end
    results
end

let
    fig = Figure()
    ax = Axis(fig[1, 1]; autolimitaspect = 1)
    sl = Slider(fig[1, 2]; horizontal = false, range = eachindex(pointclouds))
    im = image!(ax, extrema.(axes(imgs[1]))..., parent(imgs[1]))
    sp = scatter!(ax, pointclouds[1].points; markersize = resolution ./ maxweight .* pointclouds[1].weights, markerspace = :data, color = :lime)
    on(sl.value) do i
        pc = pointclouds[i]
        Makie.update!(sp; arg1 = pc.points, markersize = resolution ./ maxweight .* pc.weights)
        lims = extrema.(axes(imgs[i]))
        Makie.update!(im; arg1 = lims[1], arg2 = lims[2], arg3 = parent(imgs[i]))
    end
    fig
end

# reg_pointclouds = map(pointclouds, Ts) do pc, T
#     T(pc)
# end

# results = run_experiment(100:500:4000);
results = run_experiment([50., 20., 10.]);

reg_imgs = map(results) do result
    map(imgs, result.Ts) do img, T
        Timg = warp(img, inv(T), 0)
        # Timg[iszero.(Timg)] .= NaN
        Timg
    end
end;


let
    fig = Figure()
    # ax = PolarAxis(fig[1, 1]; theta_0 = -pi/2, direction = -1)
    # thetalims!(ax, -pi/2, pi/2)
    ax = Axis(fig[1, 1])
    colors = cgrad(:viridis, length(results); categorical = true)
    sl = Slider(fig[1, 2]; range = eachindex(results), horizontal = false)
    # pts = map(sl.value) do i
    #     result = results[i]
    #     angles = [rad2deg(rotation_angle(T.linear)) for T in result.residualTs]
    #     norms = [norm(T.translation) / result.resolution for T in result.residualTs]
    #     Point.(angles, norms)
    # end
    # scatter!(ax, pts)
    for (result, color) in zip(results, colors)
        angles = [rad2deg(rotation_angle(T.linear)) for T in result.residualTs]
        norms = [norm(T.translation) / result.resolution for T in result.residualTs]
        med_angles = median(angles)
        med_norms = median(norms)
        lo_angles, hi_angles = extrema(angles)
        lo_norms, hi_norms = extrema(norms)
        rangebars!(ax, [med_angles], [lo_norms], [hi_norms]; color)
        rangebars!(ax, [med_norms], [lo_angles], [hi_angles]; direction = :x, color)
        # scatter!(ax, med_angles, med_norms; color)
        scatter!(ax, angles, norms; color)
    end
    fig
end

let
    fig = Figure()
    ax = Axis(fig[1:2, 1]; autolimitaspect = 1)
    sl = Slider(fig[2, 2]; horizontal = false, range = eachindex(pointclouds))
    Label(fig[1, 2], @lift(string($(sl.value))))
    scatter!(ax, target.points; markersize = resolution ./ maxweight .* target.weights, markerspace = :data)
    sp = scatter!(ax, target.points; markersize = resolution ./ maxweight .* target.weights, markerspace = :data)
    on(sl.value) do i
        pc = Ts[i](pointclouds[i])
        Makie.update!(sp; arg1 = pc.points, markersize = resolution ./ maxweight .* pc.weights)
    end
    fig
end

let
    fig = Figure()
    ax = Axis(fig[1:3, 1]; autolimitaspect = 1)
    sli = Slider(fig[1, 2]; horizontal = false, range = eachindex(imgs), tellwidth = true)
    # slr = Slider(fig[1, 3]; horizontal = false, range = eachindex(results))
    Label(fig[2, 2], "discretization:")
    mnr = Menu(fig[3, 2]; options = [(results[i].distance, i) for i in eachindex(results)], tellwidth = false)
    img_to_plot = @lift reg_imgs[$(mnr.selection)][$(sli.value)]
    lim1 = @lift extrema(axes($img_to_plot, 1))
    lim2 = @lift extrema(axes($img_to_plot, 2))
    arr = @lift parent($img_to_plot)
    pc = @lift results[$()].pointclouds[$()]
    res = map(mnr.selection, sli.value) do ri, i
        result = results[ri]
        pc = result.pointclouds[i]
        PointCloudRegistration.avg_nn_dist(pc)
    end
    pc = map(mnr.selection, sli.value) do ri, i
        result = results[ri]
        pc = result.pointclouds[i]
        T = result.Ts[i]
        factor = PointCloudRegistration.avg_nn_dist(pc) / maximum(pc.weights)
        PointCloud(T(pc).points, pc.weights .* factor)
    end
    pc_points = @lift ($pc).points
    pc_weights = @lift ($pc).weights
    im = image!(ax, lim1, lim2, arr; label = "image")
    sp = scatter!(ax, pc_points; markersize = pc_weights, markerspace = :data, color = :lime, label = "pointcloud")
    axislegend(ax)
    fig
end

let
    fig = Figure()
    ax = Axis(
        fig[1, 1];
        # xscale = log10,
        # yscale = log10,
        xlabel = "Maximum residual translation (in pixels)",
        ylabel = "Processing time (in seconds)"
    )
    max_deviations = map(results) do result
        norms = maximum(T -> norm(T.translation), result.residualTs)
    end
    scatter!(ax, [r.distance for r in results], max_deviations)
    fig
end

let
    fig = Figure()
    ax = Axis(fig[1, 1])
    # ax = Axis(fig[1, 1]; yscale = log10)
    for result in results
        norms = map(T -> norm(T.translation), result.residualTs)
        scatterlines!(ax, norms ./ result.resolution; label = string(result.number))
    end
    axislegend(ax; position = :rb)
    fig
end

function diff_view(img1, img2)
    # p1, p2 = paddedviews(0, float.(img1 .> 0), float.(img2 .> 0))
    p1, p2 = paddedviews(0, img1, img2)
    # 1 .- abs.(p1 .- p2)
    g = similar(p1)
    g .= 0
    d = collect(colorview(RGB, p1, g, p2))
    d[iszero.(d)] .= RGB(1, 1, 1)
    d
end

let
    result_idx = 2
    result = results[result_idx]
    limits = ((500, 1000), (800, 980))
    sz = map(splat(-) ∘ reverse, limits)
    # fig = Figure(size = (4, 5) .* sz .÷ 2)
    fig = Figure()
    @info "showing images for result" result.distance result.resolution
    for img_idx in eachindex(orig_imgs)
        ax = Axis(
            fig[Tuple(CartesianIndices((5, 4))[img_idx])...];
            aspect = DataAspect(),
            limits,
            width = sz[1] ÷ 2,
            height = sz[2] ÷ 2,
        )
        hidedecorations!(ax)
        img1 = orig_imgs[img_idx]
        img2 = warp(orig_imgs[img_idx], result.residualTs[img_idx], 0)
        for (img, cmap, alpha) in ((img1, :greens, 1.), (img2, :blues, .5))
            image!(ax, extrema(axes(img, 1)), extrema(axes(img, 2)), img; colormap = cmap, alpha)
        end
    end
    rowgap!(fig.layout, 5)
    colgap!(fig.layout, 5)
    resize_to_layout!(fig)
    # save("../../paper/bioinformatics/src/img/biotisr.png", fig)
    fig
end
