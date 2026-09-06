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

function zero_cov(source::PointCloud{N, TS}, target::PointCloud{N, TT}) where {N, TS, TT}
    a = zero(SVector{N, TS})
    b = zero(SVector{N, TT})
    zero(a * b')
end
