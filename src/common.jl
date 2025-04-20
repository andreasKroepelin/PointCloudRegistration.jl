function check_sizes(pointclouds...)
    allequal(size, pointclouds) ||
        throw(ArgumentError("point clouds must have same size"))
end

default_config() =
    (; iterations = 10, annealing = 5, restarts = 5, rng = Random.default_rng())

function bbox(X)
    lo = hi = first(points(X))
    for point in points(X)
        lo = min.(point, lo)
        hi = max.(point, hi)
    end
    lo, hi
end

function eigen_cov(X, mean_X = mean(points(X)))
    centered = mappedarray(Base.Fix2(-, mean_X), points(X))
    cov_X = mean(x -> x * x', centered) |> Symmetric

    eigen(cov_X)
end

maxvar(X::AbstractMatrix) = maxvar(eigen_cov(X))
maxvar(eig::Eigen) = maximum(eig.values)

nrows(Xs::AbstractMatrix...) = nrows(Size.(Xs)...)
nrows(::Size{Sz}) where {Sz} = first(Sz)::Int
function nrows(::Size{Sz}, szs::Size...) where {Sz}
    N = nrows(szs...)
    if first(Sz) == N
        N
    else
        throw(ArgumentError("nrows expects matrices of equal number of rows"))
    end
end

common_eltype(Xs...) = promote_type(eltype.(Xs)...)

statically_known_rows(X::AbstractMatrix) = statically_known_rows(Size(X), X)
function statically_known_rows(::Size{Sz}, X) where {Sz}
    N, M = Sz
    if N isa Int
        X
    else
        HybridMatrix{size(X, 1), M}(X)
    end
end

function rotation_type(source::AbstractMatrix, target::AbstractMatrix)
    N = nrows(source, target)
    T = common_eltype(source, target)
    rotation_type(Val(N), T)
end

rotation_type(::Val{N}, ::Type{T}) where {N, T} = SMatrix{N, N, T, N * N}

function translation_type(source::AbstractMatrix, target::AbstractMatrix)
    N = nrows(source, target)
    T = common_eltype(source, target)
    translation_type(Val(N), T)
end

translation_type(::Val{N}, ::Type{T}) where {N, T} = SVector{N, T}

function transformation_type(a, b)
    AffineMap{rotation_type(a, b), translation_type(a, b)}
end

struct WeightedPointCloud{T, P <: AbstractMatrix{T}, W <: AbstractVector} <:
       AbstractMatrix{T}
    points::P
    weights::W

    function WeightedPointCloud(points::AbstractMatrix, weights::AbstractVector)
        size(points, 2) == length(weights) || throw(
            ArgumentError("number of points must match number of weights"),
        )
        new{eltype(points), typeof(points), typeof(weights)}(points, weights)
    end
end

WeightedPointCloud(points::AbstractMatrix) =
    WeightedPointCloud(points, Ones(eltype(points), size(points, 2)))

Base.size(wpc::WeightedPointCloud, dims...) = size(wpc.points, dims...)
StaticArrays.Size(wpc::WeightedPointCloud) = Size(wpc.points)
Base.@propagate_inbounds Base.getindex(wpc::WeightedPointCloud, i...) =
    getindex(wpc.points, i...)
Base.axes(wpc::WeightedPointCloud, dims...) = axes(wpc.points, dims...)
# We use `x -> SVector(x)` instead of just `SVector` so that MappedArrays.jl
# can infer the eltype better.
points(wpc::WeightedPointCloud) =
    mappedarray(x -> SVector(x), eachcol(wpc.points))
weights(wpc::WeightedPointCloud) = wpc.weights

statically_known_rows(wpc::WeightedPointCloud) =
    WeightedPointCloud(statically_known_rows(wpc.points), wpc.weights)
nrows(wpcs::WeightedPointCloud...) =
    nrows(map(Base.Fix2(getfield, :points), wpcs)...)

weighted(wpc::WeightedPointCloud) = wpc
weighted(points::AbstractMatrix) = WeightedPointCloud(points)

struct TransformationWithCost{T <: Real, A <: AffineMap}
    cost::T
    transformation::A
end

function worst(
    A::Type{<:AffineMap{<:AbstractMatrix{T}, <:AbstractVector{T}}},
) where {T}
    TransformationWithCost(typemax(T), identity_transformation(A))
end

function better(
    t1::TransformationWithCost{T, A},
    t2::TransformationWithCost{T, A},
) where {T, A}
    if t1.cost < t2.cost
        t1
    else
        t2
    end
end

function annealing_plan(target_or_eigen, scale, n)
    if n < 2
        (scale^2,)
    else
        logrange(maxvar(target_or_eigen), scale^2; length = n)
    end
end
