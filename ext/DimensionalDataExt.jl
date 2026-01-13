module DimensionalDataExt
using PointCloudRegistration
import PointCloudRegistration: density2pointcloud
using StaticArrays
using DimensionalData

function density2pointcloud(density::DimArray)
    points = vec(map(SVector, Iterators.product(dims(density)...)))
    weights = vec(density)
    PointCloud(points, weights)
end

end

