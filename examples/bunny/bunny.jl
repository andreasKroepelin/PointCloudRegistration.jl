using Revise
using Downloads
using CodecZlib
using Tar
using PlyIO
using PointCloudRegistration
using PointCloudRegistration.Distances
using PointCloudRegistration.Rotations
using PointCloudRegistration.CoordinateTransformations
using GLMakie

url = "http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz"
iobuffer = Downloads.download(url, IOBuffer())
seekstart(iobuffer)
bunny_dir = Tar.extract(GzipDecompressorStream(iobuffer))
data_dir = joinpath(bunny_dir, "bunny", "data")

files = readdir(data_dir; join = true)
plys = filter(endswith(".ply"), files)
function pc_from_ply(filename)
    vertices = load_ply(filename)["vertex"]
    coords = mapreduce(d -> vertices[d]', vcat, ["x", "y", "z"])
    PointCloud(coords)
end
pointclouds = Dict([
    let
        key = basename(chopsuffix(filename, ".ply"))
        key => pc_from_ply(filename)
    end
    for filename in plys
])

# successes:
# bun000: bun315, bun045
# bun045: 

target_key = "bun045"
target = pointclouds[target_key]
source_keys = filter(!=(target_key), keys(pointclouds))
prepd_target = prepare_target_kc(target; scale = DownTo(.001));

Ts = Dict([
    let
        @info "registering $key"
        source = pointclouds[key]
        T = register_kc(source, prepd_target; restarts = PCReg.RandomRestarts(20))
        key => T
    end
    for key in source_keys
])
aligned_sources = Dict([key => Ts[key](pointclouds[key]) for key in source_keys])

let
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    menu = Menu(fig[2, 1]; options = source_keys)
    source = @lift aligned_sources[$(menu.selection)]
    scatter!(ax, target)
    scatter!(ax, source)
    fig
end

let source = pointclouds["top3"]
    fig = Figure()
    ax = Axis3(fig[1, 1:2]; aspect = :data)
    sg = SliderGrid(
        fig[2, 1],
        (label = "α", range = -180:180, startvalue = 0),
        (label = "β", range = -180:180, startvalue = 0),
        (label = "γ", range = -180:180, startvalue = 0),
    )
    btn = Button(fig[2, 2]; label = "go")
    rotation = @lift RotXYZ{Float32}(
        deg2rad($(sg.sliders[1].value)),
        deg2rad($(sg.sliders[2].value)),
        deg2rad($(sg.sliders[3].value)),
    )
    initT = map(rot -> AffineMap(rot, target.mean - rot * source.mean), rotation)
    T_source = Observable{PointCloud{3, Float32}}(source)
    on(btn.clicks) do _
        T = register_kc(source, prepd_target; restarts = PCReg.FixedRestarts([initT[]]))
        T_source[] = T(source)
    end
    on(initT) do T
        T_source[] = T(source)
    end
    # kc = @lift string(PCReg.eval_kernel_correlation(last(prepd_target.annealing_levels), source, $T))
    # Label(fig[2, 2], kc; width = 100)
    scatter!(ax, target)
    scatter!(ax, T_source)
    fig
end

full_bunny = pc_from_ply(joinpath(bunny_dir, "bunny", "reconstruction", "bun_zipper.ply"))

max_radius = sqrt(maximum(full_bunny.coveigvals))
sections = map(1:100) do _
    center = rand(full_bunny.points)
    sqradius = .1 * max_radius # (max_radius * rand(.2:.001:1.))^2
    @info "radius" sqrt(sqradius)
    selected_points = [p for p in full_bunny.points if sqeuclidean(p, center) <= sqradius]
    PointCloud(selected_points)
end

let
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    sl = Slider(fig[2, 1]; range = eachindex(sections))
    scatter!(ax, @lift(sections[$(sl.value)]))
    fig
end

let source = pointclouds["bun090"]
    features_source = PCReg.nn_features(source; nfeatures = 900)
    features_target = PCReg.nn_features(target; nfeatures = 900)
    idcs_source, idcs_target = guess_correspondences(source, target, features_source, features_target; compatibility_deviation = .01, compatibility_coverage = 500)
    source_sel = source[idcs_source]
    target_sel = target[idcs_target]
    @info "selected" source_sel target_sel
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    scatter!(ax, source_sel)
    scatter!(ax, target_sel)
    arrows2d!(ax, source_sel.points, target_sel.points .- source_sel.points)
    wait(display(fig))
    T = register_gmc(source_sel, target_sel)
    T_source_sel = T(source_sel)
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    scatter!(ax, T_source_sel)
    scatter!(ax, target_sel)
    # arrows2d!(ax, T_source_sel.points, target_sel.points .- T_source_sel.points)
    wait(display(fig))
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    scatter!(ax, T(source))
    scatter!(ax, target)
    wait(display(fig))
end
