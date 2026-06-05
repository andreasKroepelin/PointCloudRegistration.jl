struct NoUpperBound <: Number end

Base.:<(::Number, ::NoUpperBound) = true
Base.:<(::NoUpperBound, ::Number) = false

"""
    IterativeClosestPoint([; distance_cutoff, restarts, iterations, report_pair, report_iteration, report_restart])

Rigid registration of point clouds that does not need any a priori
correspondence information.

The algorithm iteratively assumes correspondences between closest points and
then minimizes the RMSD (see [`Kabsch`](@ref)).

[Iterative Closest Point on Wikipedia](https://en.wikipedia.org/wiki/Iterative_closest_point)

# Parameters
- `distance_cutoff`: Even if two points are closest to eachother, they are not
  considered to be corresponding if their distance is farther than this cutoff.
  No cutoff by default.
- `restarts`: Determines how to restart the optimization to avoid local optima,
  see Section [Restarts](@ref).
  Default: `RandomRestarts(50)`
- `iterations`: How many iterations to perform at most, might stop earlier if
  convergence is detected.
  Default: `100`
- `report_pair`: Callback to run on every correspondence pair.
  Must accept the following keyword arguments:
  - `source_idx`: Index in the source point cloud of the first point.
  - `target_idx`: Index in the target point cloud of the second point.
  - `distance`: Their distance.
  Default: `(; kwargs...) -> nothing`
- `report_iteration`: Callback to run on every iteration.
  Must accept the following keyword arguments:
  - `iter`: Number of the current iteration.
  - `cost`: Cost of the current solution candidate.
  - `transformation`: Currently best found transformation.
  Default: `(; kwargs...) -> nothing`
- `report_restart`: Callback to run on every restart.
  Must accept the following keyword arguments:
  - `restart`: Number of the current restart.
  - `cost`: Cost of the optimum found in this restart.
  - `transformation`: Optimal transformation found in this restart.
  Default: `(; kwargs...) -> nothing`
"""
@kwdef struct IterativeClosestPoint{D <: Number, R <: AbstractRestarts, RP, RI, RR}
    distance_cutoff::D = NoUpperBound()
    restarts::R = RandomRestarts(50)
    iterations::Int = 100
    report_pair::RP = no_report
    report_iteration::RI = no_report
    report_restart::RR = no_report
end

"""
    rigid_registration(source, target, algorithm::IterativeClosestPoint)

Perform rigid registration via [`IterativeClosestPoint`](@ref).
See [here](@ref rigid_registration(::Any, ::Any, ::Any)) for general info about
this function.
"""
function rigid_registration(source, target, alg::IterativeClosestPoint, flip::FlipMarker = NoFlip())
    @argcheck alg.iterations >= 1

    pc_source = PointCloud(source)
    pc_target = PointCloud(target)

    _rigid_icp(
        pc_source,
        pc_target,
        flip,
        alg.distance_cutoff,
        alg.restarts,
        alg.iterations,
        alg.report_pair,
        alg.report_iteration,
        alg.report_restart,
    )
end

function _rigid_icp(
    source::PointCloud{N, TS},
    target::PointCloud{N, TT},
    flip,
    dist_cutoff,
    restarts,
    iterations,
    report_pair,
    report_iteration,
    report_restart,
) where {N, TS, TT}
    T = promote_type(TS, TT)

    target_tree = KDTree(target.points)
    best = worst(T, transformation_type(Val(N), T, flip))
    restarts_iter = restarts_iterator(source, target, restarts, flip)
    for (restart, transformation) in enumerate(restarts_iter)
        prev_transformation = identity_transformation(transformation)
        cost = zero(T)
        for iter in 1:iterations
            source_mean = zero(eltype(source.points))
            target_mean = zero(eltype(target.points))
            covariance = target_mean * source_mean'
            sum_w = zero(T)
            cost = zero(T)

            for j in eachindex(source.points)
                src = source.points[j]
                i, dist = nn(target_tree, transformation(src))
                if dist > dist_cutoff
                    continue
                end
                trg = target.points[i]
                w = source.weights[j] * target.weights[i]

                source_mean += w * src
                target_mean += w * trg
                covariance += w * trg * src'
                sum_w += w
                cost += dist^2

                report_pair(; source_idx = j, target_idx = i, distance = dist)
            end
            source_mean /= sum_w
            target_mean /= sum_w
            covariance /= sum_w
            cost /= sum_w
            covariance -= target_mean * source_mean'

            transformation = transformation_from_moments(
                covariance,
                source_mean,
                target_mean,
                flip,
            )

            report_iteration(; iter, cost, transformation)

            if iter > 1 && isapprox(transformation, prev_transformation)
                break
            end
            prev_transformation = transformation
        end
        best = better(best, TransformationWithCost(cost, transformation))
        report_restart(; restart, cost, transformation)
    end

    return best.transformation
end
