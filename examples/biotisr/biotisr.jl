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

@time pointclouds = map(imgs) do img
    Threads.@spawn @chain img begin
        density2pointcloud
        drop_low_weight(_; proportion = .4)
        thin_to_number(_, 500)
    end
end .|> fetch

maxweight = maximum(pc -> median(pc.weights), pointclouds)
resolution = minimum(pc -> PointCloudRegistration.avg_nn_dist(pc), pointclouds)

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

target = pointclouds[1]
# sources = [invT(pc) for (invT, pc) in zip(invTs, pointclouds)]
@time prepd_target = prepare_target_kc(target; scale = [2, 1] .* resolution);

@time Ts = map(pointclouds) do src
    Threads.@spawn begin
        register_kc(src, prepd_target; smm = Smm(50), restarts = RandomRestarts(200))
    end
end .|> fetch

reg_imgs = map(imgs, Ts) do img, T
    Timg = warp(img, inv(T), 0)
    Timg[iszero.(Timg)] .= NaN
    Timg
end
reg_pointclouds = map(pointclouds, Ts) do pc, T
    T(pc)
end

TinvTs = Ts .∘ invTs .∘ (inv(invTs[1]), )
# TinvTs = Ts .∘ invTs

let
    fig = Figure()
    col_rot = :red
    col_trl = :blue
    # ax = PolarAxis(fig[1, 1]; theta_0 = -pi/2, direction = -1)
    # thetalims!(ax, -pi/2, pi/2)
    ax = Axis(fig[1, 1])
    angles = [rad2deg(rotation_angle(T.linear)) for T in TinvTs]
    norms = [norm(T.translation) for T in TinvTs]
    scatter!(ax, angles, norms)
    # scatter!(ax_rot, angles; color = col_rot)
    # scatter!(ax_trl, ; color = col_trl)
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
    ax = Axis(fig[1, 1]; autolimitaspect = 1)
    sl = Slider(fig[1, 2]; horizontal = false, range = eachindex(imgs))
    im = image!(ax, (0, 1), (0, 1), zeros(1, 1); label = "image")
    sp = scatter!(ax, [SA[0., 0.]]; markersize = [0.], markerspace = :data, color = :lime, label = "pointcloud")
    on(sl.value) do i
        img = reg_imgs[i]
        arg1, arg2 = extrema.(axes(img))
        Makie.update!(im; arg1, arg2, arg3 = parent(img))
        pc = reg_pointclouds[i]
        Makie.update!(sp; arg1 = pc.points, markersize = resolution ./ maxweight .* pc.weights)
    end
    axislegend(ax)
    fig
end
