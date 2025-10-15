using Revise
using Downloads
using CodecZlib
using Tar
using PlyIO
using PointCloudRegistration
using PointCloudRegistration.StaticArrays
using PointCloudRegistration.Distances
using PointCloudRegistration.Rotations
using PointCloudRegistration.CoordinateTransformations
using GLMakie
using LinearAlgebra

if !isfile("bunny.tar.gz")
    url = "http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz"
    Downloads.download(url, "bunny.tar.gz")
end
iobuffer = IOBuffer(read("bunny.tar.gz"))
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
    end for filename in plys
])

# successes:
# bun000: bun315, bun045
# bun045: 

target_key = "bun045"
target = pointclouds[target_key]
source_keys = filter(!=(target_key), keys(pointclouds))
prepd_target = prepare_target_kc(target; scale = DownTo(0.001));

Ts = Dict([
    let
        @info "registering $key"
        source = pointclouds[key]
        T = register_kc(
            source,
            prepd_target;
            restarts = PCReg.RandomRestarts(20),
        )
        key => T
    end for key in source_keys
])
aligned_sources =
    Dict([key => Ts[key](pointclouds[key]) for key in source_keys])

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
    initT =
        map(rot -> AffineMap(rot, target.mean - rot * source.mean), rotation)
    T_source = Observable{PointCloud{3, Float32}}(source)
    on(btn.clicks) do _
        T = register_kc(
            source,
            prepd_target;
            restarts = PCReg.FixedRestarts([initT[]]),
        )
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

full_bunny = pc_from_ply(
    joinpath(bunny_dir, "bunny", "reconstruction", "bun_zipper.ply"),
)

full_bunny_thinned = thin_dpmeans(full_bunny, 0.005f0)

max_radius = sqrt(maximum(full_bunny.coveigvals))
sections = map(1:100) do _
    center = rand(full_bunny.points)
    sqradius = 0.1 * max_radius # (max_radius * rand(.2:.001:1.))^2
    @info "radius" sqrt(sqradius)
    selected_points =
        [p for p in full_bunny.points if sqeuclidean(p, center) <= sqradius]
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
    idcs_source, idcs_target = guess_correspondences(
        source,
        target,
        features_source,
        features_target;
        compatibility_deviation = 0.01,
        compatibility_coverage = 500,
    )
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

let
    fig = Figure()
    ax = Axis3(fig[1, 1:2]; aspect = :data)
    sl = Slider(fig[2, 1]; range = range(0, 1; length = 100))
    btn = Button(fig[2, 2]; label = "rnd dir!")
    nobs = Observable(SA[1., 0., 0.])
    projected_obs = map(nobs) do n
        [dot(n, p) for p in full_bunny.points]
    end
    mask = map(projected_obs, sl.value) do projected, rthr
        lo, hi = extrema(projected)
        thr = lo + rthr * (hi - lo)
        projected .<= thr
    end
    on(btn.clicks) do _
        nobs[] = normalize(randn(SVector{3}))
    end
    meshscatter!(ax, full_bunny.points; markersize = 1e-3, color = mask)
    fig
end

let
    fig = Figure()
    ax = Axis3(fig[1, 1:2]; aspect = :data)
    isl = IntervalSlider(fig[2, 1]; range = range(0, 1; length = 100))
    btn = Button(fig[2, 2]; label = "go")
    plt1 = meshscatter!(ax, full_bunny_thinned.points; markersize = 5e-3)
    plt2 = meshscatter!(ax, full_bunny_thinned.points; markersize = 5e-3)
    n = randn(SVector{3, Float32})
    projected = [dot(n, p) for p in full_bunny_thinned.points]
    lo, hi = extrema(projected)
    on(btn.clicks) do _
        rel_thresholds = isl.interval[]
        thresholds = lo .+ rel_thresholds .* (hi - lo)
        masks = (
            projected .<= thresholds[2],
            projected .>= thresholds[1],
        )
        first_slice, second_slice = map(mask -> full_bunny_thinned[mask], masks)
        prepd_second_slice = prepare_target_kc(second_slice; scale = logrange(0.02f0, 0.005f0, length = 5))
        T = register_kc(first_slice, prepd_second_slice; restarts = RandomRestarts(100))
        @info "transformation" rad2deg(rotation_angle(RotMatrix(T.linear))) norm(T.translation)
        Makie.update!(plt1, arg1 = T(first_slice).points)
        Makie.update!(plt2, arg1 = second_slice.points)
    end
    fig
end

let
    n = SA[1.0f0, 0.0f0, 0.0f0]
    projected = [dot(n, p) for p in full_bunny_thinned.points]
    lo, hi = extrema(projected)
    threshold_range = range(0, 1; length = 200)
    successes = fill(NaN, length(threshold_range), length(threshold_range))
    overlaps = copy(successes)
    progress_counter = Threads.Atomic{Int64}(0)
    max_progress = length(threshold_range)^2
    ts = Task[]
    for i in eachindex(threshold_range)
        lrt = threshold_range[i]
        lrt < 1 || continue
        t = Threads.@spawn begin
            lt = lo + lrt * (hi - lo)
            second_mask = projected .>= lt
            second_slice = full_bunny_thinned[second_mask]
            prepd_second_slice = prepare_target_kc(second_slice; scale = logrange(0.02f0, 0.005f0, length = 5))
            for j in eachindex(threshold_range)
                urt = threshold_range[j]
                Threads.atomic_add!(progress_counter, 1)
                urt <= lrt && continue
                @info "progress" progress_counter[]/max_progress
                # @info "iteration" i j lrt urt
                ut = lo + urt * (hi - lo)
                first_mask = projected .<= ut
                first_slice = full_bunny_thinned[first_mask]
                T = register_kc(first_slice, prepd_second_slice; restarts = RandomRestarts(100))
                angle = rad2deg(rotation_angle(RotMatrix(T.linear)))
                nrm = norm(T.translation)
                successes[i, j] = angle <= 2 && nrm <= .005f0
                overlaps[i, j] = count(splat(==), zip(first_mask, second_mask))
            end
        end
        push!(ts, t)
    end
    @time fetch.(ts)
    overlaps ./= length(full_bunny_thinned.points)
    fig = Figure()
    ax = Axis(fig[1:2, 1:2]; aspect = DataAspect(), xlabel = "lower threshold", ylabel = "upper threshold", xaxisposition = :top, xgridvisible = false, ygridvisible = false)
    hidespines!(ax, :b, :r)
    heatmap!(ax, threshold_range, threshold_range, successes; colormap = [colorant"#ff851b", colorant"#7fdbff"])
    contour!(ax, threshold_range, threshold_range, overlaps; levels = 0.:.2:1., labels = true, labelcolor = :black, labelformatter = x -> Makie.Format.format("{:.0%}", x), labelsize = 15, color = :gray, linestyle = :dash, linewidth = 3)
    Legend(fig[2, 2], [[LineElement(color = :gray, linestyle = :dash, linewidth = 3), MarkerElement(color = :black, marker = '%', markersize = 15)], PolyElement(color =  colorant"#7fdbff", strokewidth = 0), PolyElement(color =  colorant"#ff851b", strokewidth = 0)], ["overlap", "success", "failure"], patchsize = (30, 20), tellwidth = false, tellheight=false, valign = :top, halign=:left, margin = (0, 0, 0, 0) )
    fig
end
