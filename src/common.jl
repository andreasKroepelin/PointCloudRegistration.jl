@inline no_report(; kwargs...) = nothing

fillzeros!(arr) = fill!(arr, zero(eltype(arr)))

wsum(points, weights) = wsum(identity, points, weights)

function wsum(f, points::VecOfSVec, weights)
    s = zero(f(zero(eltype(points))))
    for (point, weight) in zip(points, weights)
        s += weight * f(point)
    end
    s
end

function zero_cov(source::PointCloud{N}, target::PointCloud{N}) where {N}
    a = zero(eltype(target.points))
    b = zero(eltype(source.points))
    zero(a * b')
end
