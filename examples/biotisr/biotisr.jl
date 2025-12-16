using Revise
using PointCloudRegistration
using Makie
import GLMakie
using StaticArrays
using PythonCall

# data source: https://zenodo.org/records/13843670/files/BioTISR_Mitochondria.zip?download=1

# use the Python package because MRCFile.jl seems to have trouble properly
# reading the file
const mrcfile = pyimport("mrcfile")
mrc = pyconvert(Array, mrcfile.read("BioTISR_Mitochondria/Cell_001/SIM_gt.mrc"));
data = permutedims(mrc, (3, 2, 1))
data .-= minimum(data)

let
    fig = Figure()
    ax = Axis(fig[1, 1])
    sl = Slider(fig[1, 2]; horizontal = false, range = axes(data, 3))
    cr = extrema(data)
    hm = heatmap!(ax, @lift(@view data[:, :, $(sl.value)]); colorrange = cr)
    Colorbar(fig[1, 0], hm; colorrange = cr)
    fig
end

pointclouds_dense = map(eachslice(data; dims = 3)) do img
    points = CartesianIndices(img) |> vec .|> Tuple .|> SVector .|> float
    weights = vec(img)
    PointCloud(points, weights)
end

pointclouds = map(pointclouds_dense) do pc
    Threads.@spawn begin
        @info "next"
        thin_dpmeans(pc, 10.)
    end
end

target = pointclouds[1]
sources = [PointCloudRegistration.rand_transformation(pc)(pc) for pc in pointclouds]
prepd_target = prepare_target_kc(target; scale = [20., 10.])

Ts = map(sources) do src
    Threads.@spawn begin
        register_kc(src, prepd_target; smm = Smm(50), restarts = RandomRestarts(20))
    end
end .|> fetch
