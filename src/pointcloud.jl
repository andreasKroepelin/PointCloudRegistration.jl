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

Returns a view of the given matrix where every column is a
`SVector{N, eltype(mat)}`.
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
    sum_w = zero(eltype(weights))
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

PointCloud(points::VecOfSVec) = PointCloud(points, Trues(length(points)))

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
Base.axes(pc::PointCloud{N}, i) where {N} =
    ifelse(i == 1, SOneTo(N), eachindex(pc.points))
# We use `x -> SVector(x)` instead of just `SVector` so that MappedArrays.jl
# can infer the eltype better.
# points(pc::PointCloud) = mappedarray(x -> SVector(x), eachcol(pc.points))
# points(X::HybridMatrix) = mappedarray(x -> SVector(x), eachcol(X))
# points(xs::AbstractVector{<: SVector}) = xs
# weights(pc::PointCloud) = pc.weights

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
