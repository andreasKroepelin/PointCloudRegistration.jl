# data source:
# https://github.com/StefanBaar/cell_align/tree/b1fb54c4dc82a8b951d5b97fb0ed20de6bb384a1/samples/F9-1(MIT14v2-4ng)

using Revise
using PointCloudRegistration
using Makie
import GLMakie
import CairoMakie
using Colors
using StaticArrays
using InvertedIndices
using ImageIO
using FileIO

function show_images(images; invert = false)
    transform = if invert
        img -> Gray(1) .- img
    else
        identity
    end
    fig = Figure()
    ax = Axis(fig[1, 1])
    sl = Slider(fig[2, 1]; range = eachindex(images))
    image!(ax, @lift(transform(images[$(sl.value)])))
    DataInspector(fig)
    GLMakie.activate!()
    display(fig)
end

original_images = [load(lpad(i, 4, "0") * ".jpeg") for i in 1:10]
show_images(original_images)

img_size = only(unique(size.(original_images)))

gray_images = [Gray.(img) for img in original_images]
show_images(gray_images)

bg_colors = map(gray_images) do gray_img
    colors = unique(gray_img)
    counts = [count(==(color), gray_img) for color in colors]
    colors[argmax(counts)]
end
bg_color = float(only(unique(bg_colors)))

bg_contrast_images = [abs.(img .- bg_color) for img in gray_images]
show_images(bg_contrast_images; invert = true)

pointclouds_raw = let
    points = CartesianIndices(img_size) |> vec .|> Tuple .|> SVector .|> float
    map(bg_contrast_images) do img
        weights = vec(float.(gray.(img)))
        PointCloud(points, weights)
    end
end

let
    fig = Figure()
    ax = Axis(fig[1, 1]; autolimitaspect = 1)
    ax_h = Axis(fig[1, 2])
    sl = Slider(fig[2, 1]; range = eachindex(pointclouds_raw))
    factor = 2 / maximum(pc -> maximum(pc.weights), pointclouds_raw)
    scatter!(
        ax,
        @lift(pointclouds_raw[$(sl.value)].points);
        markersize = @lift(factor .* pointclouds_raw[$(sl.value)].weights),
        markerspace = :data,
    )
    hist!(ax_h, @lift(pointclouds_raw[$(sl.value)].weights))
    fig
end

resolution = 6.0f0
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
    ax_h = Axis(fig[1, 2])
    sl = Slider(fig[2, 1]; range = eachindex(pointclouds))
    factor = 2resolution / maximum(pc -> maximum(pc.weights), pointclouds)
    scatter!(
        ax,
        @lift(pointclouds[$(sl.value)].points);
        markersize = @lift(factor .* pointclouds[$(sl.value)].weights),
        # markersize = 1,
        markerspace = :data,
    )
    hist!(ax_h, @lift(pointclouds[$(sl.value)].weights))
    # on(_ -> autolimits!(ax_h), sl.value)
    fig
end

target_idx = 1
target = pointclouds[target_idx]
# sources = pointclouds[Not(target_idx)]
sources = pointclouds[begin:end]
prepd_target = prepare_target_kc(target; scale = DownTo(resolution));

Ts = map(sources) do source
    Threads.@spawn begin
        rigid_kc(source, prepd_target; restarts = RandomRestarts(30), smm = Smm(50))
    end
end .|> fetch

let
    fig = Figure()
    ax = Axis(fig[1, 1]; autolimitaspect = 1)
    sl = Slider(fig[2, 1]; range = eachindex(sources))
    factor = 2resolution / maximum(pc -> maximum(pc.weights), pointclouds)
    # scatter!(ax, target.points; markersize = factor .* target.weights)
    src_plt = scatter!(ax, Ts[1](sources[1]).points, markersize = factor .* sources[1].weights, markerspace = :data)
    on(sl.value) do i
        Makie.update!(src_plt; arg1 = Ts[i](sources[i]).points, markersize = factor .* sources[i].weights)
    end
    fig
end

cpds = map(sources, Ts) do source, T
    Threads.@spawn begin
        nonrigid_cpd(target, T(source); scale = resolution, outlier_proportion = 0.01f0, regularizer_strength = .01f0, regularizer_lengthscale = 20f0)
    end
end .|> fetch

let
    fig = Figure()
    ax = Axis(fig[1, 1]; autolimitaspect = 1)
    ax_h = Axis(fig[1, 2]; autolimitaspect = 1)
    sl = Slider(fig[2, 1]; range = eachindex(sources))
    factor = 2resolution / maximum(pc -> maximum(pc.weights), pointclouds)
    # scatter!(ax, target.points; markersize = factor .* target.weights)
    for j in eachindex(target.points)
        pts = [Point(cpd.target_representatives[j]) for cpd in cpds]
        lines!(ax, pts; linewidth = 1, color = 1:length(cpds), colormap = :blues)
    end
    src_plt = scatter!(ax, Ts[1](sources[1]).points, markersize = factor .* sources[1].weights, markerspace = :data, label = "original")
    # dtrg_plt = scatter!(ax, target.points .+ cpds[1].displacement, markersize = factor .* target.weights, markerspace = :data, label = "reconstructed")
    dtrg_plt = scatter!(ax, cpds[1].target_representatives, markersize = factor .* target.weights, markerspace = :data, label = "reconstructed")
    # dis_plt = arrows2d!(ax, Ts[1](sources[1]).points, cpds[1].displacement)
    hm_plt = heatmap!(ax_h, cpds[1].correspondences)
    on(sl.value) do i
        Makie.update!(src_plt; arg1 = Ts[i](sources[i]).points, markersize = factor .* sources[i].weights)
        # Makie.update!(dis_plt; arg1 = Ts[i](sources[i]).points, arg2 = cpds[i].displacement)
        # Makie.update!(dtrg_plt; arg1 = target.points .+ cpds[i].displacement)
        Makie.update!(dtrg_plt; arg1 = cpds[i].target_representatives)
        Makie.update!(hm_plt; arg1 = cpds[i].correspondences)
    end
    axislegend(ax)
    fig
end

# for publication:

let
    fig = Figure(size = (5, 2) .* img_size .÷ 2)
    idcs = CartesianIndices((2, 5))
    for (img, idx) in zip(bg_contrast_images, idcs)
        ax = Axis(fig[Tuple(idx)...], aspect = DataAspect())
        hidedecorations!(ax)
        image!(ax, Gray(1f0) .- img)
    end
    # GLMakie.activate!()
    # display(fig)
    CairoMakie.activate!(pdf_version = "1.5")
    save("../../paper/bioinformatics/src/img/frame-shift-imgs.pdf", fig)
end

let
    fig = Figure(size = (400, 300))
    ax = Axis(fig[1, 1]; #= autolimitaspect = 1, =# aspect = DataAspect(), title = "not registered")
    ax_t = Axis(fig[1, 2]; #= autolimitaspect = 1, =# aspect = DataAspect(), title = "registered")
    hidedecorations!(ax)
    hidedecorations!(ax_t)
    linkxaxes!(ax, ax_t)
    linkyaxes!(ax, ax_t)
    factor = 5 / maximum(pc -> maximum(pc.weights), pointclouds)
    for (pc, T) in zip(pointclouds, Ts)
        scatter!(ax, pc.points; markersize = factor .* pc.weights, markerspace = :data)
        scatter!(ax_t, T(pc).points; markersize = factor .* pc.weights, markerspace = :data)
    end
    GLMakie.activate!(); display(fig)
    # CairoMakie.activate!(pdf_version = "1.5")
    # save("../../paper/bioinformatics/src/img/frame-shift-pointclouds.pdf", fig)
end

let
    fig = Figure(size = (400, 300))
    ax = Axis(fig[1, 1]; #= autolimitaspect = 1, =# aspect = DataAspect(), title = "not registered")
    ax_t = Axis(fig[1, 2]; #= autolimitaspect = 1, =# aspect = DataAspect(), title = "registered")
    hidedecorations!(ax)
    hidedecorations!(ax_t)
    linkxaxes!(ax, ax_t)
    linkyaxes!(ax, ax_t)
    for (img, T) in zip(bg_contrast_images, Ts)
        # scatter!(ax, pc.points; markersize = factor .* pc.weights, markerspace = :data)
        # scatter!(ax_t, T(pc).points; markersize = factor .* pc.weights, markerspace = :data)
        heatmap!(ax, gray.(img), alpha = 0.3)
    end
    GLMakie.activate!(); display(fig)
    # CairoMakie.activate!(pdf_version = "1.5")
    # save("../../paper/bioinformatics/src/img/frame-shift-pointclouds.pdf", fig)
end
