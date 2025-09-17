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

to_vec_of_svec(arg::Any) = throw(
    ArgumentError(
        "don't know how to interpret $(summary(arg)) as a list of points",
    ),
)

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

The purpose of this type is to bring the input data into a form that enables
efficient execution of the registration algorithms.
As for the public interface, you can consider `PointCloud` to be defined as
```julia
struct PointCloud{N, T} <: AbstractMatrix{T}
    points::AbstractVector{SVector{N, T}}
    weights::AbstractVector{<: Real}
    # ... private fields ...
end
```
with the `SVector` type from StaticArrays.jl.

# Simple usage

You can construct a `PointCloud` from a matrix or a vector of vectors:
```julia
julia> PointCloud([1. 2. 3.; 4. 5. 6.])
2-dimensional point cloud with 3 points of eltype Float64
 1.0  2.0  3.0
 4.0  5.0  6.0
and unit weights

julia> PointCloud([[1., 4.], [2., 5.], [3., 6.]])
2-dimensional point cloud with 3 points of eltype Float64
 1.0  2.0  3.0
 4.0  5.0  6.0
and unit weights
```

# Type stability

Since the dimensionality of the point cloud is part of the type, the two calls
above are not type stable (unless the dimensionality can be inferred, e. g. when
the number of rows of the matrix is statically known).
This is not a big deal because every function operating on `PointCloud`s is then
type stable
([*function barrier*](https://docs.julialang.org/en/v1/manual/performance-tips/#kernel-functions)).
If you really want the construction of a `PointCloud` to be type stable, you can
use the `PointCloud{N}` variant:
```julia
julia> PointCloud{2}([1. 2. 3.; 4. 5. 6.])
2-dimensional point cloud with 3 points of eltype Float64
 1.0  2.0  3.0
 4.0  5.0  6.0
and unit weights

julia> PointCloud{3}([1. 2. 3.; 4. 5. 6.])
ERROR: ArgumentError: matrix must have given number of rows 3
[...]

julia> isconcretetype(Core.Compiler.return_type(PointCloud, Tuple{Matrix{Float64}}))
false

julia> isconcretetype(Core.Compiler.return_type(PointCloud{2}, Tuple{Matrix{Float64}}))
true
```

# Weights

Optionally, each point can have an individual non-negative weight.
```julia
julia> PointCloud([1. 2. 3.; 4. 5. 6.], [.5, 1., .5])
2-dimensional point cloud with 3 points of eltype Float64
 1.0  2.0  3.0
 4.0  5.0  6.0
and weights
 3-element Vector{Float64}
 0.5  1.0  0.5
```
If no weights are specified, implicit unit weights are used.
(They are not explicitly stored and have minimal runtime cost for computations.)
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
    @argcheck length(points) == length(weights) "number of points must match number of weights"
    @argcheck all(>=(0), weights) "weights must be non-negative"
    mean, cov, sum_of_weights = mean_cov_sumw(points, weights)
    @argcheck sum_of_weights > 0 "weights cannot all be zero"
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

"""
    PointCloud(points)
    PointCloud{N}(points)

Create a [`PointCloud`](@ref) from the given points (matrix or vector of
vectors) and use implicit unit weights.
Optionally provide the dimensionality `N` for better type inference.
"""
PointCloud(points) = PointCloud(to_vec_of_svec(points))

"""
    PointCloud(points, weights)
    PointCloud{N}(points, weights)

Create a [`PointCloud`](@ref) from the given points (matrix or vector of
vectors) and weights.
Optionally provide the dimensionality `N` for better type inference.
"""
PointCloud(points, weights::AbstractVector) =
    PointCloud(to_vec_of_svec(points), weights)

PointCloud{N}(points) where {N} = PointCloud(to_vec_of_svec(points, Val(N)))
PointCloud{N}(points, weights::AbstractVector) where {N} =
    PointCloud(to_vec_of_svec(points, Val(N)), weights)

"""
    PointCloud(pc::PointCloud) = pc

Converting a [`PointCloud`](@ref) to a [`PointCloud`](@ref) just returns the
argument, i.e. `PointCloud(...)` is *idempotent*.
"""
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

function Base.show(
    io::IO,
    ::MIME"text/plain",
    pc::PointCloud{N, T},
) where {N, T}
    mat_io = IOContext(io, :limit => true, :compact => true)
    println(
        io,
        N,
        "-dimensional point cloud with ",
        length(pc.points),
        " points of eltype ",
        T,
    )
    Base.print_matrix(mat_io, pc)
    println(io)
    if pc.weights isa Trues || pc.weights isa Ones
        println(io, "and unit weights")
    else
        println(io, "and weights")
        print(io, " ")
        summary(io, pc.weights)
        println(io)
        Base.print_matrix(mat_io, pc.weights', " ")
        println(io)
    end
end

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
