"""
    VecOfSVec{N, T}

Abbreviation for an `AbstractVector` of `SVectors` of length `N` and eltype `T`
"""
const VecOfSVec{N, T} = AbstractVector{<:SVector{N, T}}

"""
    VecOfSVec(::AbstractMatrix)

Potentially type unstable version of `VecOfSVec{N}(::AbstractMatrix)` when
the number of rows is not inferrable from the given matrix type
"""
VecOfSVec(mat::AbstractMatrix) = VecOfSVec{_size_1(mat)}(mat)

"""
    VecOfSVec{N}(mat::AbstractMatrix)

Returns a view of the given matrix as a vector of `SVector{N, eltype(mat)}`,
containing the entries of each column of `mat`.
The number of rows of `mat` must be `N`.
"""
function VecOfSVec{N}(mat::AbstractMatrix) where {N}
    T = eltype(mat)
    reinterpret(reshape, SVector{N, T}, mat)
end

_size_1(mat::AbstractMatrix) = _size_1(Size(mat), mat)
_size_1(::Size{Sz}, mat) where {Sz} = _size_1(first(Sz), mat)
_size_1(i::Int, mat) = i
_size_1(::StaticArrays.Dynamic, mat) = size(mat, 1)

function mean_cov_sumw(points, weights)
    sum_w = zero(eltype(eltype(points)))
    mean = zero(eltype(points))
    cov = mean * mean'

    for (x, w) in zip(points, weights)
        sum_w += w
        diff = x - mean
        mean += w / sum_w * diff
        cov += w * diff * (x - mean)'
    end

    cov /= sum_w

    mean, cov, sum_w
end

"""
Representation of a weighted point cloud.
An instance `pc` of `PointCloud{N, T}` stores `N`-dimensional points with
coordinate type `T`.
Also, it acts as an `AbstractMatrix{T}` with `N` rows and `length(pc.points)`
columns.
"""
struct PointCloud{
    N,
    T,
    P <: VecOfSVec{N, T},
    WT,
    W <: AbstractVector,
    EV <: SMatrix{N, N, T},
} <: AbstractMatrix{T}
    points::P
    weights::W
    sum_of_weights::WT
    mean::SVector{N, T}
    coveigvecs::EV
    coveigvals::SVector{N, T}
end

"""
    PointCloud(::VecOfSVec, ::AbstractVector)

Wrap a list of points and explicit weights as a `PointCloud`.
"""
function PointCloud(points::VecOfSVec, weights::AbstractVector)
    length(points) == length(weights) ||
        throw(ArgumentError("number of points must match number of weights"))
    mean, cov, sum_of_weights = mean_cov_sumw(points, weights)
    coveig = eigen(cov)
    PointCloud(
        points,
        weights,
        sum_of_weights,
        mean,
        coveig.vectors,
        coveig.values,
    )
end

"""
    PointCloud(::VecOfSVec)

Create a point cloud from the given points and use implicit unit weights.
"""
PointCloud(points::VecOfSVec) = PointCloud(points, Trues(length(points)))

"""
    PointCloud(::AbstractMatrix)
    PointCloud{N}(::AbstractMatrix)
    PointCloud(::AbstractMatrix, ::AbstractVector)
    PointCloud{N}(::AbstractMatrix, ::AbstractVector)

Create a point cloud from a matrix where every column represents one point.
If the static parameter `N` is given, it must match the number of rows of the
matrix, if not, this might not be type stable.
A vector of weights can be given explicitly, otherwise implicit unit weights are
used.
"""
PointCloud(points_mat::AbstractMatrix) = PointCloud(VecOfSVec(points_mat))
PointCloud(points_mat::AbstractMatrix, weights::AbstractVector) =
    PointCloud(VecOfSVec(points_mat), weights)
PointCloud{N}(points_mat::AbstractMatrix) where {N} =
    PointCloud(VecOfSVec{N}(points_mat))
PointCloud{N}(points_mat::AbstractMatrix, weights::AbstractVector) where {N} =
    PointCloud(VecOfSVec{N}(points_mat), weights)

PointCloud(pc::PointCloud) = pc

Base.size(pc::PointCloud{N}) where {N} = (N, length(pc.points))
StaticArrays.Size(pc::PointCloud{N}) where {N} = Size(N, length(pc.points))
Base.@propagate_inbounds Base.getindex(pc::PointCloud, i, j) =
    getindex(getindex(pc.points, j), i)
Base.@propagate_inbounds function Base.getindex(
    pc::PointCloud,
    idcs::AbstractVector{<: Integer},
)
    PointCloud(pc.points[idcs], pc.weights[idcs])
end
Base.axes(pc::PointCloud{N}, i) where {N} =
    ifelse(i == 1, SOneTo(N), eachindex(pc.points))

nrows(pcs::PointCloud{N}) where {N} = N

(m::AffineMap)(pc::PointCloud) = PointCloud(
    m.(pc.points),
    pc.weights,
    pc.sum_of_weights,
    m(pc.mean),
    m.linear * pc.coveigvecs,
    pc.coveigvals,
)

(m::LinearMap)(pc::PointCloud) = PointCloud(
    m.(pc.points),
    pc.weights,
    pc.sum_of_weights,
    m(pc.mean),
    m(pc.coveigvecs),
    pc.coveigvals,
)
