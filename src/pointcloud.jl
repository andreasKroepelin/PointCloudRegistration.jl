const VecOfSVec{N, T} = AbstractVector{<:SVector{N, T}}

function VecOfSVec(mat::AbstractMatrix)
    N = size(mat, 1)
    T = eltype(mat)
    reinterpret(reshape, SVector{N, T}, mat)
end

struct PointCloud{N, T, P <: VecOfSVec{N, T}, WT, W <: AbstractVector} <:
       AbstractMatrix{T}
    points::P
    weights::W
    sum_of_weights::WT
    mean::SVector{N, T}
end

function PointCloud(points::VecOfSVec, weights::AbstractVector)
    length(points) == length(weights) ||
        throw(ArgumentError("number of points must match number of weights"))
    sum_of_weights = sum(weights)
    mean = wsum(points, weights) / sum_of_weights
    PointCloud(points, weights, sum_of_weights, mean)
end

PointCloud(points::VecOfSVec) = PointCloud(points, Trues(length(points)))

PointCloud(points_mat::AbstractMatrix) = PointCloud(VecOfSVec(points_mat))
PointCloud(points_mat::AbstractMatrix, weights::AbstractVector) =
    PointCloud(VecOfSVec(points_mat), weights)

PointCloud(pc::PointCloud) = pc

Base.size(pc::PointCloud{N}) where {N} = (N, length(pc.points))
StaticArrays.Size(pc::PointCloud{N}) where {N} = Size(N, length(pc.points))
Base.@propagate_inbounds Base.getindex(pc::PointCloud, i, j) =
    getindex(getindex(pc.points, i), j)
Base.axes(pc::PointCloud{N}, i) where {N} =
    ifelse(i == 1, SOneTo(N), eachindex(pc.points))
# We use `x -> SVector(x)` instead of just `SVector` so that MappedArrays.jl
# can infer the eltype better.
# points(pc::PointCloud) = mappedarray(x -> SVector(x), eachcol(pc.points))
# points(X::HybridMatrix) = mappedarray(x -> SVector(x), eachcol(X))
# points(xs::AbstractVector{<: SVector}) = xs
# weights(pc::PointCloud) = pc.weights

nrows(pcs::PointCloud{N}) where {N} = N

(m::AbstractAffineMap)(pc::PointCloud) =
    PointCloud(m.(points), pc.weights, pc.sum_of_weights, m(mean))
