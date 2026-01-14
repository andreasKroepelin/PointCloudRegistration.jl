function avg_nn_dist(pc::PointCloud)
    tree = KDTree(pc.points)
    _idcs, dists = allnn(tree)
    mean(dists)
end

function min_nn_dist(pc::PointCloud)
    tree = KDTree(pc.points)
    _idcs, dists = allnn(tree)
    minimum(dists)
end

function avg_nn_dist_fast(pc::PointCloud)
    i1 = rand(eachindex(pc.points))
    i2 = rand(eachindex(pc.points))
    i3 = rand(eachindex(pc.points))
    test_points = map(i -> pc.points[i], (i1, i2, i3))
    some_dist = sqeuclidean(test_points[1], test_points[2])
    mins = map(Returns(typemax(some_dist)), test_points)
    discount_zero(x) = ifelse(iszero(x), typemax(x), x)
    for p in pc.points
        mins = min.(mins, discount_zero.(sqeuclidean.(test_points, (p, ))))
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
    DownTo(scale, [steps = 2])

Annealing plan starting from the largest standard deviation of the target point
cloud in any direction, going down to `scale` in `steps` steps with logarithmic
progression.
"""
struct DownTo{T <: Number}
    scale::T
    steps::Int
end

DownTo(scale) = DownTo(scale, 2)

const ScaleType =
    Union{T, <: AbstractVector{T}, TargetScales, DownTo{T}} where {T <: Number}

annealing_plan(::PointCloud{N, T}, scale::Number) where {N, T} =
    tuple(T(scale)^2)

annealing_plan(::PointCloud{N, T}, scales::AbstractVector) where {N, T} =
    T.(scales) .^ 2

annealing_plan(target, ann::TargetScales) =
    annealing_plan(target, DownTo(avg_nn_dist_fast(target) / 2, ann.steps))

function annealing_plan(target::PointCloud{N, T}, ann::DownTo) where {N, T}
    hi = maxcoveigval(target)
    lo = T(ann.scale) ^ 2
    _logrange(hi, lo; length = ann.steps)
end

_logrange(start::Real, stop::Real; length) = logrange(start, stop; length)
# Fallback for types not covered by stdlib logrange
function _logrange(start::Number, stop::Number; length)
    factor = (stop / start)^inv(length - 1)
    start .* factor .^ (0:(length - 1))
end
