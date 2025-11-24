# data source:
# https://github.com/StefanBaar/cell_align/tree/b1fb54c4dc82a8b951d5b97fb0ed20de6bb384a1/samples/F9-1(MIT14v2-4ng)

using Revise
using PointCloudRegistration
using GLMakie
using Colors
using StaticArrays
using InvertedIndices
using ImageIO
using FileIO

function show_images(images)
    fig = Figure()
    ax = Axis(fig[1, 1])
    sl = Slider(fig[2, 1]; range = eachindex(images))
    image!(ax, @lift(images[$(sl.value)]))
    DataInspector(fig)
    fig
end

original_images = [load(lpad(i, 4, "0") * ".jpeg") for i in 1:10]
show_images(original_images)

gray_images = [Gray.(img) for img in original_images]
show_images(gray_images)

@assert allequal(size.(gray_images))

pointclouds_raw = let
    points = gray_images |> first |> CartesianIndices |> vec .|> Tuple .|> SVector .|> float
    bg_val = 0.5f0
    map(gray_images) do gray_img
        weights = [abs(float(px.val) - bg_val) for px in vec(gray_img)]
        PointCloud(points, weights)
    end
end

let
    fig = Figure()
    ax = Axis(fig[1, 1]; autolimitaspect = 1)
    sl = Slider(fig[2, 1]; range = eachindex(pointclouds_raw))
    factor = 5 / maximum(pc -> maximum(pc.weights), pointclouds_raw)
    scatter!(
        ax,
        @lift(pointclouds_raw[$(sl.value)].points);
        markersize = @lift(factor .* pointclouds_raw[$(sl.value)].weights),
    )
    fig
end

resolution = 7.0f0
pointclouds = map(pointclouds_raw) do pc
    Threads.@spawn begin
        step1 = thin_droplowweight(pc, 0.002)
        step2 = thin_dpmeans(step1, resolution)
        step3 = thin_droplowweight(step2, 0.1)
        step3
    end
end .|> fetch

let
    fig = Figure()
    ax = Axis(fig[1, 1]; autolimitaspect = 1)
    sl = Slider(fig[2, 1]; range = eachindex(pointclouds))
    factor = 5 / maximum(pc -> maximum(pc.weights), pointclouds)
    scatter!(
        ax,
        @lift(pointclouds[$(sl.value)].points);
        markersize = @lift(factor .* pointclouds[$(sl.value)].weights),
    )
    fig
end

target_idx = 1
target = pointclouds[target_idx]
# sources = pointclouds[Not(target_idx)]
sources = pointclouds[begin:end]
prepd_target = prepare_target_kc(target; scale = DownTo(resolution));

Ts = map(sources) do source
    register_kc(source, prepd_target; restarts = RandomRestarts(30), smm = Smm(50))
end

let
    fig = Figure()
    ax = Axis(fig[1, 1]; autolimitaspect = 1)
    sl = Slider(fig[2, 1]; range = eachindex(sources))
    factor = 5 / maximum(pc -> maximum(pc.weights), pointclouds)
    # scatter!(ax, target.points; markersize = factor .* target.weights)
    src_plt = scatter!(ax, Ts[1](sources[1]).points, markersize = factor .* sources[1].weights)
    on(sl.value) do i
        Makie.update!(src_plt; arg1 = Ts[i](sources[i]).points, markersize = factor .* sources[i].weights)
    end
    fig
end

cpds = map(sources, Ts) do source, T
    Threads.@spawn begin
        register_cpd(target, T(source); scale = resolution, outlier_proportion = 0.01f0, regularizer_strength = .01f0, regularizer_lengthscale = 20f0)
    end
end .|> fetch

let
    fig = Figure()
    ax = Axis(fig[1, 1]; autolimitaspect = 1)
    ax_h = Axis(fig[1, 2]; autolimitaspect = 1)
    sl = Slider(fig[2, 1]; range = eachindex(sources))
    factor = 5 / maximum(pc -> maximum(pc.weights), pointclouds)
    # scatter!(ax, target.points; markersize = factor .* target.weights)
    src_plt = scatter!(ax, Ts[1](sources[1]).points, markersize = factor .* sources[1].weights, label = "original")
    dtrg_plt = scatter!(ax, target.points .+ cpds[1].displacement, markersize = factor .* target.weights, label = "reconstructed")
    # dis_plt = arrows2d!(ax, Ts[1](sources[1]).points, cpds[1].displacement)
    hm_plt = heatmap!(ax_h, cpds[1].correspondences)
    on(sl.value) do i
        Makie.update!(src_plt; arg1 = Ts[i](sources[i]).points, markersize = factor .* sources[i].weights)
        # Makie.update!(dis_plt; arg1 = Ts[i](sources[i]).points, arg2 = cpds[i].displacement)
        Makie.update!(dtrg_plt; arg1 = target.points .+ cpds[i].displacement)
        Makie.update!(hm_plt; arg1 = cpds[i].correspondences)
    end
    axislegend(ax)
    fig
end
