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

to_matrix(vecs::VecOfSVec{N, T}) where {N, T} = reinterpret(reshape, T, vecs)

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
struct PointCloud{N, T, P <: VecOfSVec{N, T}, W <: AbstractVector}
    points::P
    weights::W

    function PointCloud(
        points::P,
        weights::W
    ) where {N, T, P <: VecOfSVec{N, T}, W <: AbstractVector}

        @argcheck length(points) == length(weights) "number of points must match number of weights"
        any(<(0), weights) && @warn "weights must be non-negative" minimum(weights)
        any(>(0), weights) || @warn "weights cannot all be zero"
        new{N, T, P, W}(points, weights)
    end
end

# """
#     PointCloud(::VecOfSVec, ::AbstractVector)

# Wrap a list of points and explicit weights as a `PointCloud`.
# """

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
PointCloud{N, T}() where {N, T} = PointCloud(SVector{N, T}[], Bool[])

"""
    PointCloud(pc::PointCloud) = pc

Converting a [`PointCloud`](@ref) to a [`PointCloud`](@ref) just returns the
argument, i.e. `PointCloud(...)` is *idempotent*.
"""
PointCloud(pc::PointCloud) = pc

"""
    getindex(pointcloud, idcs::AbstractVector{<:Integer})

Returns a new point cloud with points and weights chosen from `pointcloud`
according to `idcs`.

# Example
```julia
julia> pointcloud = PointCloud([1.0 2.0 3.0; 4.0 5.0 6.0], [7.0, 8.0, 9.0])
2-dimensional point cloud with 3 points of eltype Float64
 1.0  2.0  3.0
 4.0  5.0  6.0
and weights
 3-element Vector{Float64}
 7.0  8.0  9.0

julia> pointcloud[[1, 3]] # select indices
2-dimensional point cloud with 2 points of eltype Float64
 1.0  3.0
 4.0  6.0
and weights
 2-element Vector{Float64}
 7.0  9.0

julia> pointcloud[[false, true, true]] # boolean mask
2-dimensional point cloud with 2 points of eltype Float64
 2.0  3.0
 5.0  6.0
and weights
 2-element Vector{Float64}
 8.0  9.0
```
"""
Base.@propagate_inbounds function Base.getindex(
    pc::PointCloud,
    idcs::AbstractVector{<: Integer},
)
    PointCloud(pc.points[idcs], pc.weights[idcs])
end
Base.axes(pc::PointCloud{N}) where {N} = (SOneTo(N), eachindex(pc.points))
dimension(::PointCloud{N}) where {N} = N
Base.eltype(::PointCloud{N, T}) where {N, T} = T

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
    Base.print_matrix(mat_io, to_matrix(pc.points))
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

function mean_cov(pc::PointCloud)
    sum_w = zero(one(eltype(eltype(pc.points))))
    mean = zero(eltype(pc.points))
    cov = mean * mean'

    for (x, w) in zip(pc.points, pc.weights)
        iszero(w) && continue
        sum_w += w
        diff = x - mean
        mean += w / sum_w * diff
        cov += w * diff * (x - mean)'
    end

    cov /= sum_w

    mean, cov
end

function maxcoveigval(pc::PointCloud)
    _, cov = mean_cov(pc)
    cov ./ oneunit(eltype(cov)) |> Symmetric |> eigvals |> maximum
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

function bbox_hypervolume(pc::PointCloud)
    lo, hi = bbox(pc)
    prod(hi - lo)
end

(m::AffineMap)(pc::PointCloud) = PointCloud(m.(pc.points), pc.weights)

(m::LinearMap)(pc::PointCloud) = PointCloud(m.(pc.points), pc.weights)

function Base.vcat(pc1::PointCloud{N}, pc2::PointCloud{N}) where {N}
    PointCloud(
        vcat(pc1.points, pc2.points),
        vcat(pc1.weights, pc2.weights),
    )
end

function Base.isapprox(
    pc1::PointCloud{N},
    pc2::PointCloud{N};
    kwargs...,
) where {N}
    return all(fieldnames(PointCloud)) do fn
        f1 = getfield(pc1, fn)
        f2 = getfield(pc2, fn)
        isapprox(f1, f2; kwargs...)
    end
end

"""
    density2pointcloud(density::AbstractArray{T, N}) where {T, N}

Creates an `N`-dimensional point cloud that represents the `density` by placing
a point at every index of `density` with the corresponding value as the point's
weight.

# Example
```julia
julia> pixels = reshape(11:19, 3, 3)
3×3 reshape(::UnitRange{Int64}, 3, 3) with eltype Int64:
 11  14  17
 12  15  18
 13  16  19

julia> density2pointcloud(pixels)
2-dimensional point cloud with 9 points of eltype Float64
 1.0  2.0  3.0  1.0  2.0  3.0  1.0  2.0  3.0
 1.0  1.0  1.0  2.0  2.0  2.0  3.0  3.0  3.0
and weights
 9-element UnitRange{Int64}
 11  12  13  14  15  16  17  18  19

julia> voxels = reshape(11:18, 2, 2, 2)
2×2×2 reshape(::UnitRange{Int64}, 2, 2, 2) with eltype Int64:
[:, :, 1] =
 11  13
 12  14

[:, :, 2] =
 15  17
 16  18

julia> density2pointcloud(voxels)
3-dimensional point cloud with 8 points of eltype Float64
 1.0  2.0  1.0  2.0  1.0  2.0  1.0  2.0
 1.0  1.0  2.0  2.0  1.0  1.0  2.0  2.0
 1.0  1.0  1.0  1.0  2.0  2.0  2.0  2.0
and weights
 8-element UnitRange{Int64}
 11  12  13  14  15  16  17  18
```
"""
function density2pointcloud(density::AbstractArray)
    points = CartesianIndices(density) |> vec .|> Tuple .|> SVector .|> float
    weights = vec(density)
    PointCloud(points, weights)
end
