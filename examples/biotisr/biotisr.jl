using Revise
using PointCloudRegistration
using Makie
import GLMakie
using Statistics
using LinearAlgebra
using StaticArrays
using Rotations
using CoordinateTransformations
import Images: warp, mapwindow, binarize, Otsu, RGB, paddedviews, colorview, zeroarray
import Images
using PythonCall
using Chain

# data source: https://zenodo.org/records/13843670/files/BioTISR_Mitochondria.zip?download=1

# use the Python package because MRCFile.jl seems to have trouble properly
# reading the file
const mrcfile = pyimport("mrcfile")
mrc = pyconvert(Array, mrcfile.read("BioTISR_Mitochondria/Cell_006/SIM_gt.mrc"));
orig_imgs = eachslice(permutedims(mrc, (3, 2, 1)); dims = 3)

# orig_imgs_edges = [canny(img, (Percentile(80), Percentile(40)), 10) for img in orig_imgs]
# orig_imgs_edges = [mapwindow(median!, img, (11, 11)) for img in orig_imgs]
orig_imgs_edges = [Images.binarize(mapwindow(median!, img, (5, 5)), Images.Otsu()) for img in orig_imgs]

let
    fig = Figure()
    ax = Axis(fig[1, 1]; autolimitaspect = 1)
    sl = Slider(fig[1, 2]; horizontal = false, range = eachindex(orig_imgs))
    im = image!(ax, @lift(orig_imgs_edges[$(sl.value)]))
    fig
end

invTs = [AffineMap(rand(RotMatrix2), 1000 * (@SVector randn(2))) for _ in 1:20]

imgs = map(orig_imgs, invTs) do orig_img, invT
    img = copy(orig_img)
    lo, hi = extrema(img)
    img .-= lo
    img ./= hi - lo
    img = mapwindow(median!, img, (5, 5))
    mask = binarize(img, Otsu())
    img[iszero.(mask)] .= 0
    warp(img, inv(invT), 0)
end

let
    fig = Figure()
    ax = Axis(fig[1, 1]; autolimitaspect = 1)
    sl = Slider(fig[1, 2]; horizontal = false, range = eachindex(imgs))
    im = image!(ax, (0, 1), (0, 1), zeros(1, 1))
    on(sl.value) do i
        img = imgs[i]
        arg1, arg2 = extrema.(axes(img))
        Makie.update!(im; arg1, arg2, arg3 = parent(img))
    end
    fig
end

function run_experiment(numbers)
    results = []
    for number in numbers
        @info "Running next experiment" number
        time = @elapsed begin
            pointclouds = map(imgs) do img
                Threads.@spawn @chain img begin
                    density2pointcloud
                    drop_low_weight(_; proportion = .1)
                    thin_to_number(_, number)
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
        push!(results, (; number, time, resolution, pointclouds, Ts, residualTs))
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

results = run_experiment(100:500:4000);

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
    mnr = Menu(fig[3, 2]; options = [(results[i].number, i) for i in eachindex(results)], tellwidth = false)
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
        factor = 5 * PointCloudRegistration.min_nn_dist(pc) / maximum(pc.weights)
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
    scatter!(ax, [r.number for r in results], max_deviations)
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
    result_idx = 6
    result = results[result_idx]
    limits = ((500, 1000), (800, 980))
    sz = map(splat(-) ∘ reverse, limits)
    # fig = Figure(size = (4, 5) .* sz .÷ 2)
    fig = Figure()
    @info "showing images for result" result.number result.resolution
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
    save("../../paper/bioinformatics/src/img/biotisr.png", fig)
end
