function register_gmc(
    source,
    target;
    restarts::AbstractRestarts = default_restarts(),
    iterations::Int = default_iterations(),
    # use type parameters `RI` and `RR` here to force specialization
    report_iteration::RI = no_report,
    report_restart::RR = no_report,
) where {RI, RR}
    @argcheck iterations >= 1

    pc_source = PointCloud(source)
    pc_target = PointCloud(target)

    _register_icp(
        pc_source,
        pc_target,
        restarts,
        iterations,
        report_iteration,
        report_restart,
    )
end

function _register_icp(
    source::PointCloud{N, TS},
    target::PointCloud{N, TT},
    restarts,
    iterations,
    report_iteration,
    report_restart,
) where {N, TS, TT}
    T = promote_type(TS, TT)

    target_tree = KDTree(target.points)
    idcs = [1]
    dists = [zero(T)]
    best = worst(transformation_type(Val(N), T))
    restarts_iter = restarts_iterator(source, target, restarts)
    for (restart, transformation) in enumerate(restarts_iter)
        prev_transformation = identity_transformation(transformation)
        cost = zero(T)
        for iter in 1:iterations
            source_mean = zero(eltype(source.points))
            target_mean = zero(eltype(target.points))
            covariance = zero(rotation_type(source, target))
            cost = zero(T)

            for j in eachindex(source.points)
                src = source.points[j]
                knn!(idcs, dists, target_tree, transformation(src), 1)
                dist = only(dists)
                i = only(idcs)
                trg = target.points[i]
                w = source.weights[j] * target.weights[i]

                source_mean += w * src
                target_mean += w * trg
                covariance += w * trg * src'
                sum_w += w
                cost += dist^2
            end
            source_mean /= sum_w
            target_mean /= sum_w
            covariance /= sum_w
            covariance -= target_mean * source_mean'

            transformation = transformation_from_moments(
                covariance,
                source_mean,
                target_mean,
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
