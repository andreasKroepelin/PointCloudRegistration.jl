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

    bunny_dir = load_bunny_data()
    full_bunny = pc_from_ply(joinpath(bunny_dir, "bunny", "reconstruction", "bun_zipper.ply"))
    resolution = 5f-3
    pc = thin_dpmeans(full_bunny, resolution)

    n = SA[1.0f0, 0.0f0, 0.0f0]
    projected = [dot(n, p) for p in pc.points]
    lo, hi = extrema(projected)
    threshold_range = range(0, 1; length = numthresholds)
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
            second_slice = pc[second_mask]
            prepd_second_slice = prepare_target_kc(second_slice; scale = logrange(4resolution, resolution, length = 5))
            for j in eachindex(threshold_range)
                urt = threshold_range[j]
                Threads.atomic_add!(progress_counter, 1)
                urt <= lrt && continue
                @info "progress" progress_counter[]/max_progress
                # @info "iteration" i j lrt urt
                ut = lo + urt * (hi - lo)
                first_mask = projected .<= ut
                first_slice = pc[first_mask]
                T = register_kc(first_slice, prepd_second_slice; restarts = RandomRestarts(numrestarts))
                angle = rad2deg(rotation_angle(RotMatrix(T.linear)))
                nrm = norm(T.translation)
                successes[i, j] = angle <= 2 && nrm <= .005f0
                overlaps[i, j] = count(splat(==), zip(first_mask, second_mask))
            end
        end
        push!(ts, t)
    end
    @time fetch.(ts)
    overlaps ./= length(pc.points)

    h5open("result-$(Dates.now()).h5", "w") do h5
        h5["successes"] = successes
        h5["overlaps"] = overlaps
        h5["threshold_range"] = collect(threshold_range)
        attrs(h5)["normal"] = n
        attrs(h5)["restarts"] = numrestarts
    end
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
