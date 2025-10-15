module BunnyReassembly
using Downloads
using CodecZlib
using Tar
using PlyIO
using PointCloudRegistration
using PointCloudRegistration.StaticArrays
using PointCloudRegistration.Distances
using PointCloudRegistration.Rotations
using PointCloudRegistration.CoordinateTransformations
using LinearAlgebra
using Dates
using HDF5

function @main(args)
    if Threads.nthreads() <= 1
        @error "Not running with multiple threads."
        return
    end

    config = Dict(split.(args, '='))
    numthresholds = get(config, "numthresholds", "10") |> Base.Fix1(parse, Int)
    numrestarts = get(config, "numrestarts", "10") |> Base.Fix1(parse, Int)
    @info "config" numthresholds numrestarts Threads.nthreads()

    @info "preparing bunny (loading, thinning)..."
    bunny_dir = load_bunny_data()
    full_bunny = pc_from_ply(joinpath(bunny_dir, "bunny", "reconstruction", "bun_zipper.ply"))
    resolution = 5f-3
    pc = thin_dpmeans(full_bunny, resolution)

    @info "starting reassembly"
    n = SA[1.0f0, 0.0f0, 0.0f0]
    projected = [dot(n, p) for p in pc.points]
    lo, hi = extrema(projected)
    threshold_range = range(0, 1; length = numthresholds)
    scales = logrange(4resolution, resolution, length = 5)
    analyzer = Analyzer(; pc, scales, projected, lo, hi, threshold_range, numrestarts)
    analyzer_spawner = AnalyzerSpawner(analyzer)
    # precompile:
    analyzer(.5)
    ts = map(analyzer_spawner, threshold_range)
    @time results = fetch.(ts)
    successes = stack(res -> res.successes, results)'
    overlaps = stack(res -> res.overlaps, results)'

    h5open("result-$(Dates.now()).h5", "w") do h5
        h5["successes"] = collect(successes)
        h5["overlaps"] = collect(overlaps)
        h5["threshold_range"] = collect(threshold_range)
        attrs(h5)["normal"] = n
        attrs(h5)["restarts"] = numrestarts
    end
end

# Extracting inner loop into these awkward functor structs to avoid
# recompilation when running on multiple threads

@kwdef struct Analyzer{T, PC, S, TR}
    pc::PC
    scales::S
    projected::Vector{T}
    lo::T
    hi::T
    threshold_range::TR
    numrestarts::Int
end

function (analyzer::Analyzer)(lrt)
    (; pc, scales, projected, lo, hi, threshold_range, numrestarts) = analyzer
    successes = fill(NaN, length(threshold_range))
    overlaps = copy(successes)
    if lrt >= 1
        return (; successes, overlaps)
    end
    lt = lo + lrt * (hi - lo)
    second_mask = projected .>= lt
    second_slice = pc[second_mask]
    prepd_second_slice = prepare_target_kc(second_slice; scale = scales)
    for j in eachindex(threshold_range)
        urt = threshold_range[j]
        urt <= lrt && continue
        ut = lo + urt * (hi - lo)
        first_mask = projected .<= ut
        first_slice = pc[first_mask]
        T = register_kc(first_slice, prepd_second_slice; restarts = RandomRestarts(numrestarts))
        angle = rad2deg(rotation_angle(RotMatrix(T.linear)))
        nrm = norm(T.translation)
        successes[j] = angle <= 2 && nrm <= .005f0
        overlaps[j] = count(splat(==), zip(first_mask, second_mask)) / length(pc.points)
    end
    return (; successes, overlaps)
end

struct AnalyzerSpawner{A}
    analyzer::A
end

function (as::AnalyzerSpawner)(lrt)
    Threads.@spawn as.analyzer(lrt)
end

function load_bunny_data()
    if !isfile("bunny.tar.gz")
        @info "downloading data..."
        url = "http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz"
        Downloads.download(url, "bunny.tar.gz")
    end
    iobuffer = IOBuffer(read("bunny.tar.gz"))
    bunny_dir = Tar.extract(GzipDecompressorStream(iobuffer))
    return bunny_dir
end

function pc_from_ply(filename)
    vertices = load_ply(filename)["vertex"]
    coords = mapreduce(d -> vertices[d]', vcat, ["x", "y", "z"])
    PointCloud(coords)
end

end # module BunnyReassembly
