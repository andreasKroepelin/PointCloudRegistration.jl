function check_sizes(pointclouds...)
    allequal(size, pointclouds) ||
        throw(ArgumentError("both point clouds must have same size"))
end

function bbox(X::AbstractMatrix{T}) where {T}
    lo = SVector(ntuple(_ -> typemax(T), Val(NRows(X))))
    hi = SVector(ntuple(_ -> typemin(T), Val(NRows(X))))
    for i in axes(X, 2)
        x = X[:, i]
        lo = min.(lo, x)
        hi = max.(hi, x)
    end
    (lo, hi)
end

function p2p(X)
    lo, hi = bbox(X)
    maximum(hi .- lo)
end

guess_diameter(X, Y) = NRows(X) * max(p2p(X), p2p(Y))^2

nrows(Xs::AbstractMatrix...) = nrows(Size.(Xs))
nrows(::Size{Sz}) where {Sz} = first(Sz)::Int
function nrows(::Size{Sz}, szs::Size...) where {Sz}
    N = nrows(szs...)
    if first(Sz) == N
        N
    else
        throw(ArgumentError("nrows expects matrices of equal number of rows"))
    end
end

common_eltype(X::AbstractMatrix...) = promote_type(eltype.(X)...)

function transformation_type(source, target)
    N = nrows(source, target)
    T = common_eltype(source, target)
    R = SMatrix{N, N, T, N * N}
    L = SVector{N, T}
    AffineMap{R, L}
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

Base.size(wpc::WeightedPointCloud) = size(wpc.points)
StaticArrays.Size(wpc::WeightedPointCloud) = Size(wpc.points)
Base.getindex(wpc::WeightedPointCloud, i...) = getindex(wpc.points, i...)
Base.axes(wpc::WeightedPointCloud, dims...) = axes(wpc.points, dims...)
points(wpc::WeightedPointCloud) = Iterators.map(SVector, eachcol(wpc.points))
weights(wpc::WeightedPointCloud) = wpc.weights

statically_known_rows(X::AbstractMatrix) = statically_known_rows(Size(X), X)
function statically_known_rows(::Size{Sz}, X) where {Sz}
    N, M = Sz
    if N isa Int
        X
    else
        HybridMatrix{size(X, 1), M}(X)
    end
end

statically_known_rows(wpc::WeightedPointCloud) =
    WeightedPointCloud(statically_known_rows(wpc.points), wpc.weights)

weighted(wpc::WeightedPointCloud) = wpc
weighted(points::AbstractMatrix) = WeightedPointCloud(points)

struct TransformationWithCost{T <: Real, A <: AffineMap}
    cost::T
    transformation::A
end

function worst(
    ::Type{A <: AffineMap{<:AbstractMatrix{T}, <:AbstractVector{T}}},
) where {A, T}
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
