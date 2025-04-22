struct PointCloud{N, T, P <: AbstractMatrix{T},WT,  W <: AbstractVector{WT}} <:
       AbstractMatrix{T}
    points::P
    weights::W
    sum_of_weights::T
    mean::SVector{N, T}

    function PointCloud(points::AbstractMatrix, weights::AbstractVector)
        size(points, 2) == length(weights) || throw(
            ArgumentError("number of points must match number of weights"),
        )
        spoints = statically_known_rows(points)
        sum_of_weights = sum(weights)
        mean = wsum(spoints, weights) / sum_of_weights
        new{nrows(spoints), eltype(spoints), typeof(spoints), eltype(weights), typeof(weights)}(spoints, weights, sum_of_weights, mean)
    end
end

PointCloud(points::AbstractMatrix) =
    PointCloud(points, Trues(size(points, 2)))

PointCloud(pc::PointCloud) = pc

Base.size(pc::PointCloud, dims...) = size(pc.points, dims...)
StaticArrays.Size(pc::PointCloud) = Size(pc.points)
Base.@propagate_inbounds Base.getindex(pc::PointCloud, i...) =
    getindex(pc.points, i...)
Base.axes(pc::PointCloud, dims...) = axes(pc.points, dims...)
# We use `x -> SVector(x)` instead of just `SVector` so that MappedArrays.jl
# can infer the eltype better.
points(pc::PointCloud) =
    mappedarray(x -> SVector(x), eachcol(pc.points))
weights(pc::PointCloud) = pc.weights

nrows(pcs::PointCloud{N}) where N = N


