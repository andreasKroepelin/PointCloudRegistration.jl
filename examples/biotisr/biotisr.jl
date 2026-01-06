using Revise
using PointCloudRegistration
using Makie
import GLMakie
using Statistics
using StaticArrays
using Rotations
using CoordinateTransformations
using ImageTransformations
using ImageCore
using PythonCall
using Chain

# data source: https://zenodo.org/records/13843670/files/BioTISR_Mitochondria.zip?download=1

# use the Python package because MRCFile.jl seems to have trouble properly
# reading the file
const mrcfile = pyimport("mrcfile")
mrc = pyconvert(Array, mrcfile.read("BioTISR_Mitochondria/Cell_006/SIM_gt.mrc"));
orig_imgs = eachslice(permutedims(mrc, (3, 2, 1)); dims = 3)

let
    fig = Figure()
    ax = Axis(fig[1, 1]; autolimitaspect = 1)
    sl = Slider(fig[1, 2]; horizontal = false, range = eachindex(orig_imgs))
    im = image!(ax, @lift(orig_imgs[$(sl.value)]))
    fig
end

invTs = [AffineMap(rand(RotMatrix2), 1000 * (@SVector randn(2))) for _ in 1:20]

imgs = map(orig_imgs, invTs) do orig_img, invT
    img = copy(orig_img)
    lo, hi = extrema(img)
    img .-= lo
    img ./= hi - lo
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
        time = @elapsed begin
            pointclouds = map(imgs) do img
                Threads.@spawn @chain img begin
                    density2pointcloud
                    drop_low_weight(_; proportion = .4)
                    thin_to_number(_, number)
                end
            end .|> fetch
            resolution = minimum(pc -> PointCloudRegistration.avg_nn_dist(pc), pointclouds)
            target = pointclouds[1]
            prepd_target = prepare_target_kc(target; scale = [2, 1] .* resolution);
            Ts = map(pointclouds) do src
                Threads.@spawn begin
                    register_kc(src, prepd_target; smm = Smm(50), restarts = RandomRestarts(number))
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

results = run_experiment(100:200:2000);

reg_imgs = map(results) do result
    map(imgs, result.Ts) do img, T
        Timg = warp(img, inv(T), 0)
        Timg[iszero.(Timg)] .= NaN
        Timg
    end
end


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
        factor = 1 * PointCloudRegistration.min_nn_dist(pc) / maximum(pc.weights)
        PointCloud(T(pc).points, pc.weights .* factor)
    end
    pc_points = @lift ($pc).points
    pc_weights = @lift ($pc).weights
    im = image!(ax, lim1, lim2, arr; label = "image")
    sp = scatter!(ax, pc_points; markersize = res, markerspace = :data, color = :lime, label = "pointcloud")
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
    scatter!(ax, max_deviations, [r.time for r in results])
    fig
end

let
    fig = Figure()
    ax = Axis(fig[1, 1]; yscale = log10)
    for result in results
        norms = map(T -> norm(T.translation), result.residualTs)
        scatterlines!(ax, norms ./ result.resolution)
    end
    fig
end
