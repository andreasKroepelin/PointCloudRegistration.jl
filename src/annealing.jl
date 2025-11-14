function avg_nn_dist(pc::PointCloud)
    tree = KDTree(pc.points, Euclidean(); reorder = true)
    # the nearest neighbour is always the point itself, so query for the
    # nearest two
    _, dists2 = knn(tree, pc.points, 2)
    mean(first, dists2)
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
    annealing_plan(target, DownTo(avg_nn_dist(target) / 2, ann.steps))

function annealing_plan(target::PointCloud{N, T}, ann::DownTo) where {N, T}
    hi = maximum(target.coveigvals)
    lo = T(ann.scale) ^ 2
    _logrange(hi, lo; length = ann.steps)
end

_logrange(start::Real, stop::Real; length) = logrange(start, stop; length)
# Fallback for types not covered by stdlib logrange
function _logrange(start::Number, stop::Number; length)
    factor = (stop / start)^inv(length - 1)
    start .* factor .^ (0:(length - 1))
end
