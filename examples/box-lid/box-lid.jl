using Revise
using PointCloudRegistration
using GLMakie
using NPZ

function npy2pc(file)
    npy = npzread(file)
    PointCloud(permutedims(npy, (2, 1)))
end

boxes = let
    keys = (:chaos, :closed, :half, :open, :single, :standing)
    pcs = [npy2pc("/home/andreas/Downloads/box_andreas/$key.npy") for key in keys]
    (; (keys .=> pcs)...)
end

meshscatter(boxes.standing.points; markersize = 2)
