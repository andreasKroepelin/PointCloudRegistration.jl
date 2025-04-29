struct PointCloud{N, T, P <: AbstractMatrix{T}, WT, W <: AbstractVector} <:
       AbstractMatrix{T}
    points::P
    weights::W
    sum_of_weights::WT
    mean::SVector{N, T}
end

function PointCloud(points::AbstractMatrix, weights::AbstractVector)
    size(points, 2) == length(weights) ||
        throw(ArgumentError("number of points must match number of weights"))
    spoints = statically_known_rows(points)
    sum_of_weights = sum(weights)
    mean = wsum(spoints, weights) / sum_of_weights
    PointCloud(spoints, weights, sum_of_weights, mean)
end

PointCloud(points::AbstractMatrix) = PointCloud(points, Trues(size(points, 2)))

PointCloud(pc::PointCloud) = pc

Base.size(pc::PointCloud, dims...) = size(pc.points, dims...)
StaticArrays.Size(pc::PointCloud) = Size(pc.points)
Base.@propagate_inbounds Base.getindex(pc::PointCloud, i...) =
    getindex(pc.points, i...)
Base.axes(pc::PointCloud, dims...) = axes(pc.points, dims...)
# We use `x -> SVector(x)` instead of just `SVector` so that MappedArrays.jl
# can infer the eltype better.
points(pc::PointCloud) = mappedarray(x -> SVector(x), eachcol(pc.points))
points(X::HybridMatrix) = mappedarray(x -> SVector(x), eachcol(X))
weights(pc::PointCloud) = pc.weights

nrows(pcs::PointCloud{N}) where {N} = N

function (lm::LinearMap{<:StaticMatrix{N, N}})(pc::PointCloud{N}) where {N}
    new_points = similar(pc.points)
    mul!(new_points, lm.linear, pc.points)
    new_mean = lm.linear * pc.mean
    PointCloud(new_points, pc.weights, pc.sum_of_weights, new_mean)
end

function (am::AffineMap{<:StaticMatrix{N, N}, <: StaticVector{N}})(pc::PointCloud{N}) where {N}
    new_points = similar(pc.points)
    mul!(new_points, am.linear, pc.points)
    new_points .+= am.translation
    new_mean = am(pc.mean)
    PointCloud(new_points, pc.weights, pc.sum_of_weights, new_mean)
end
