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

statically_known_rows(X::AbstractMatrix) = statically_known_rows(Size(X), X)
function statically_known_rows(::Size{Sz}, X) where {Sz}
    N, M = Sz
    if N isa Int
        X
    else
        HybridMatrix{size(X, 1), M}(X)
    end
end

NRows(X::AbstractMatrix) = NRows(Size(X))
NRows(::Size{Sz}) where {Sz} = first(Sz)::Int

CommonType(X::AbstractMatrix...) = promote_type(eltype.(X)...)

struct WeightedPointCloud{T, P <: AbstractMatrix{T}, W <: AbstractVector} <:
       AbstractMatrix{T}
    points::P
    weights::W
end

WeightedPointCloud(points::AbstractMatrix) =
    WeightedPointCloud(points, Ones(eltype(points), size(points, 2)))

Base.size(wpc::WeightedPointCloud) = size(wpc.points)
Base.getindex(wpc::, i...) = getindex(wpc.points, i...)
