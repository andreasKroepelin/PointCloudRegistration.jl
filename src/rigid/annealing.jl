function avg_nn_dist(pc::PointCloud)
    tree = KDTree(points(pc))
    _idcs, dists = allnn(tree)
    mean(dists)
end

function min_nn_dist(pc::PointCloud)
    tree = KDTree(points(pc))
    _idcs, dists = allnn(tree)
    minimum(dists)
end

function avg_nn_dist_fast(pc::PointCloud)
    test_points = rand(points(pc), 3)
    some_dist = sqeuclidean(test_points[1], test_points[2])
    mins = map(Returns(typemax(some_dist)), test_points)
    discount_zero(x) = ifelse(iszero(x), typemax(x), x)
    for p in points(pc)
        mins = min.(mins, discount_zero.(sqeuclidean.(test_points, (p,))))
    end
    mean(sqrt, mins)
end

"""
    TargetScales([steps = 2])

Annealing plan starting from the largest standard deviation of the target point
cloud in any direction, going down to average nearest neighbor distance in the
target, in `steps` steps with logarithmic progression.
"""
struct TargetScales
    steps::Int
end

TargetScales() = TargetScales(2)

"""
    LogAnnealingTo(scale, [steps = 2])

Annealing plan starting from `2^(steps - 1) * scale`, going down to `scale` in
`steps` steps, halfing the scale at each step.
"""
struct LogAnnealingTo{T <: Number}
    scale::T
    steps::Int
end

LogAnnealingTo(scale) = LogAnnealingTo(scale, 2)

"""
    LogAnnealingToNearestNeighborDistance([steps = 2])

Annealing plan going down to average nearest neighbor distance in the
target, starting from `2^(steps - 1)` times that value in `steps` steps, halfing
the scale at each step.
"""
struct LogAnnealingToNearestNeighborDistance
    steps::Int
end

LogAnnealingToNearestNeighborDistance() =
    LogAnnealingToNearestNeighborDistance(2)

const ScaleType = Union{
    T,
    <: AbstractVector{T},
    TargetScales,
    LogAnnealingTo{T},
    LogAnnealingToNearestNeighborDistance,
} where {T <: Number}

annealing_plan(::PointCloud{N, T}, scale::Number) where {N, T} =
    tuple(T(scale)^2)

annealing_plan(::PointCloud{N, T}, scales::AbstractVector) where {N, T} =
    T.(scales) .^ 2

function annealing_plan(target, ann::TargetScales)
    lo = avg_nn_dist(target)^2
    hi = maxcoveigval(target)
    return _logrange(hi, lo; length = ann.steps)
end

function annealing_plan(target, ann::LogAnnealingToNearestNeighborDistance)
    nnd = avg_nn_dist(target)
    return annealing_plan(target, LogAnnealingTo(nnd, ann.steps))
end

function annealing_plan(target::PointCloud, ann::LogAnnealingTo)
    T = eltype(target)
    lo = T(ann.scale)^2
    return [2^i * lo for i in reverse(0:(ann.steps - 1))]
end

_logrange(start::Real, stop::Real; length) = logrange(start, stop; length)
# Fallback for types not covered by stdlib logrange
function _logrange(start::Number, stop::Number; length)
    factor = (stop / start)^inv(length - 1)
    start .* factor .^ (0:(length - 1))
end
