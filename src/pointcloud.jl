"""
    VecOfSVec{N, T}

Abbreviation for an `AbstractVector` of `SVectors` of length `N` and eltype `T`
"""
const VecOfSVec{N, T} = AbstractVector{<:SVector{N, T}}

function to_vec_of_svec(
    mat::AbstractMatrix{T},
    ::Val{N},
)::VecOfSVec{N, T} where {T, N}
    @argcheck N > 1 "can only handle two- and higher dimensional points"
    @argcheck N == size(mat, 1) "matrix must have given number of rows $N"
    reinterpret(reshape, SVector{N, T}, mat)
end

# mainly a workaround for HybridMatrix since `size(::HybridMatrix{N, ...}, 1)`
# does not *statically* return `N` (works for `StaticMatrix` though)
_size_1(mat::AbstractMatrix) = _size_1(Size(mat), mat)
_size_1(::Size{Sz}, mat) where {Sz} = _size_1(first(Sz), mat)
_size_1(i::Int, mat) = i
_size_1(::StaticArrays.Dynamic, mat) = size(mat, 1)

to_vec_of_svec(mat::AbstractMatrix) = to_vec_of_svec(mat, Val(_size_1(mat)))

function to_vec_of_svec(
    vecs::AbstractVector{<: AbstractVector{T}},
    ::Val{N},
)::VecOfSVec{N, T} where {T, N}
    @argcheck allequal(length, vecs) "all points must have same dimension"
    @argcheck N == length(first(vecs)) "points must have given dimension $N"

    map(SVector{N, T}, vecs)
end

to_vec_of_svec(vecs::AbstractVector{<: AbstractVector}) =
    to_vec_of_svec(vecs, Val(length(first(vecs))))

to_vec_of_svec(vecs::VecOfSVec{N}, ::Val{N}) where {N} = vecs

to_vec_of_svec(vecs::VecOfSVec) = vecs

to_vec_of_svec(arg::Any) =
    throw(ArgumentError("don't know how to interpret $arg as a list of points"))

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
PointCloud(points) = PointCloud(to_vec_of_svec(points))
PointCloud(points, weights::AbstractVector) =
    PointCloud(to_vec_of_svec(points), weights)
PointCloud{N}(points) where {N} = PointCloud(to_vec_of_svec(points, Val(N)))
PointCloud{N}(points, weights::AbstractVector) where {N} =
    PointCloud(to_vec_of_svec(points, Val(N)), weights)

PointCloud(pc::PointCloud) = pc

Base.size(pc::PointCloud{N}) where {N} = (N, length(pc.points))
StaticArrays.Size(pc::PointCloud{N}) where {N} = Size(N, StaticArrays.Dynamic())
Base.@propagate_inbounds Base.getindex(pc::PointCloud, i, j) =
    getindex(getindex(pc.points, j), i)
Base.@propagate_inbounds function Base.getindex(
    pc::PointCloud,
    idcs::AbstractVector{<: Integer},
)
    PointCloud(pc.points[idcs], pc.weights[idcs])
end
Base.axes(pc::PointCloud{N}) where {N} = (SOneTo(N), eachindex(pc.points))
dimension(::PointCloud{N}) where {N} = N

function bbox(xs::VecOfSVec)
    lo = hi = first(xs)
    for point in xs
        lo = min.(point, lo)
        hi = max.(point, hi)
    end
    lo, hi
end

bbox(pc::PointCloud) = bbox(pc.points)

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
