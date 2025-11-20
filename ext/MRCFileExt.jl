module MRCFileExt
using PointCloudRegistration
import PointCloudRegistration: PointCloud
using StaticArrays
using MRCFile

struct GridPoints{N, T, SRL <: StepRangeLen{T}} <: AbstractVector{SVector{N, T}}
    ranges::NTuple{N, SRL}
end

Base.IndexStyle(::Type{<:GridPoints}) = IndexLinear()

Base.size(gp::GridPoints) = tuple(prod(length.(gp.ranges)))

function Base.getindex(gp::GridPoints, i::Int)
    ci = CartesianIndices(length.(gp.ranges))[i]
    SVector(getindex.(gp.ranges, Tuple(ci)))
end

struct ClampedVec{T, P <: AbstractVector{T}} <: AbstractVector{T}
    parent::P
    min::T
end

Base.IndexStyle(::Type{<:ClampedVec}) = IndexLinear()

Base.size(cv::ClampedVec) = size(cv.parent)

function Base.getindex(cv::ClampedVec, i::Int)
    value = cv.parent[i]
    ifelse(value < cv.min, zero(value), value)
end

function PointCloud(mrc::MRCData; threshold = typemin(eltype(mrc)))
    grid_points = GridPoints(voxelaxes(header(mrc)))
    clamped_values = ClampedVec(vec(mrc.data), threshold)
    PointCloud(grid_points, clamped_values)
end

end
