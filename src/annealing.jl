function avg_nn_dist(pc::PointCloud)
    tree = KDTree(pc.points, Euclidean(); reorder = true)
    # the nearest neighbour is always the point itself, so query for the
    # nearest two
    _, dists2 = knn(tree, pc.points, 2)
    mean(first, dists2)
end

"""
    TargetScales([steps = 5])

Annealing plan starting from the largest standard deviation of the target point
cloud in any direction, going down to average nearest neighbor distance in the
target, in `steps` steps with logarithmic progression.
"""
struct TargetScales
    steps::Int
end

TargetScales() = TargetScales(5)

"""
    DownTo(scale, [steps = 5])

Annealing plan starting from the largest standard deviation of the target point
cloud in any direction, going down to `scale` in `steps` steps with logarithmic
progression.
"""
struct DownTo{T <: Real}
    scale::T
    steps::Int
end

DownTo(scale) = DownTo(scale, 5)

const ScaleType =
    Union{T, <: AbstractVector{T}, TargetScales, DownTo{T}} where {T <: Real}

annealing_plan(::PointCloud{N, T}, scale::Number) where {N, T} =
    tuple(T(scale)^2)

annealing_plan(::PointCloud{N, T}, scales::AbstractVector) where {N, T} =
    T.(scales) .^ 2

function annealing_plan(target, ann::TargetScales)
    hi = maximum(target.coveigvals)
    lo = avg_nn_dist(target) ^ 2
    logrange(hi, lo; length = ann.steps)
end

function annealing_plan(target::PointCloud{N, T}, ann::DownTo) where {N, T}
    hi = maximum(target.coveigvals)
    lo = T(ann.scale) ^ 2
    logrange(hi, lo; length = ann.steps)
end
