struct GemanMcclure{T}
    sqscale::T
end

mm_weight(gm::GemanMcclure, x::AbstractVector, y::AbstractVector) =
    mm_weight(gm, sqeuclidean(x, y))
mm_weight(gm::GemanMcclure, sqdist::Number) =
    gm.sqscale / (gm.sqscale + sqdist)^2

function mm_weight_type(gm::GemanMcclure, Y::PointCloud, X::PointCloud)
    typeof(mm_weight(gm, first(Y.points), first(X.points)))
end

cost(gm::GemanMcclure, x::AbstractVector, y::AbstractVector) =
    cost(gm, sqeuclidean(x, y))
cost(gm::GemanMcclure, sqdist::Number) = sqdist / (gm.sqscale + sqdist)

cost_type(gm::GemanMcclure, Y::PointCloud, X::PointCloud) =
    typeof(cost(gm, first(Y.points), first(X.points)))

"""
    rigid_gmc(source, target[; scale, restarts, iterations, rng, accumulator])

$REGISTER_DOCS_START
minimizes the Geman-McClure loss to `target`,
i.e.
```math
\\sum_{i = 1}^n \\operatorname{GMC}_\\rho(\\Vert R y_i + t - x_i \\Vert)
```
where
```math
\\operatorname{GMC}_\\rho(r) = \\frac{r^2}{\\rho^2 + r^2}
```
$REGISTER_DOCS_SYMBOLS
$REGISTER_DOCS_EQUAL

$REGISTER_DOCS_TYPES

The Geman-McClure loss has a scale parameter ``\\rho`` that defines the range of
distances that affect the loss (making it robust against outliers).
``\\rho`` corresponds to the `scale` keyword argument and you can learn about
how to use the keyword arguments in [this section](#Common-keyword-arguments).

Use this function if you expect outliers or wrong correspondences.
"""
function rigid_gmc(
    source,
    target;
    scale::ScaleType = default_scale(),
    restarts::AbstractRestarts = default_restarts(),
    iterations::Int = default_iterations(),
    smm = NoSmm(),
    # use type parameters `RI` and `RR` here to force specialization
    report_iteration::RI = no_report,
    report_restart::RR = no_report,
) where {RI, RR}
    @argcheck iterations >= 1

    pc_source = PointCloud(source)
    pc_target = PointCloud(target)
    @argcheck size(pc_source) == size(pc_target)

    sqscales = annealing_plan(pc_target, scale)
    _rigid_gmc(
        pc_source,
        pc_target,
        sqscales,
        restarts,
        iterations,
        smm,
        report_iteration,
        report_restart,
    )
end

function _rigid_gmc(
    source::PointCloud{N},
    target::PointCloud{N},
    sqscales,
    restarts,
    iterations,
    smm,
    report_iteration,
    report_restart,
) where {N}
    gm = GemanMcclure(oneunit(eltype(sqscales)))
    SrcT = eltype(source.points)
    TrgT = eltype(target.points)
    CostT = cost_type(gm, source, target)
    WeightT = mm_weight_type(gm, source, target)
    best = worst(CostT, transformation_type(source, target))
    gm_cost = zero(CostT)
    restarts_iter = restarts_iterator(source, target, restarts)
    source_iter = smm_iterator(smm, source)
    for (restart, transformation) in enumerate(restarts_iter)
        for sqscale in sqscales
            # double `sqscale` such that the loss function has the same
            # quadratic behavior for small distances as the kernel correlation
            # loss with `sqscale`
            gm = GemanMcclure(2sqscale)
            prev_transformation = identity_transformation(transformation)
            for this_source_iter in (source_iter, non_stochastic(source_iter))
                for iter in 1:iterations
                    sum_w = zero(WeightT)
                    source_mean = sum_w * zero(SrcT)
                    target_mean = sum_w * zero(TrgT)
                    covariance = sum_w * zero(TrgT) * zero(SrcT)'
                    gm_cost = zero(CostT)

                    for source_element in this_source_iter
                        src = source_element.point
                        trg = target.points[source_element.idx]
                        w_src = source_element.weight
                        w_trg = target.weights[source_element.idx]
                        sqdist = sqeuclidean(transformation(src), trg)
                        w_src_w_trg = w_src * w_trg
                        w = w_src_w_trg * mm_weight(gm, sqdist)
                        gm_cost += w_src_w_trg * cost(gm, sqdist)
                        source_mean += w * src
                        target_mean += w * trg
                        covariance += w * trg * src'
                        sum_w += w
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

                    report_iteration(;
                        iter,
                        annealing_level = sqscale,
                        cost = gm_cost,
                        transformation,
                    )

                    if iter > 1 && isapprox(transformation, prev_transformation)
                        break
                    end
                    prev_transformation = transformation
                end
            end
        end
        best = better(best, TransformationWithCost(gm_cost, transformation))
        report_restart(; restart, cost = gm_cost, transformation)
    end

    return best.transformation
end

function rigid_mad(
    source,
    target;
    iterations::Int = default_iterations(),
    # use type parameter `RI` here to force specialization
    report_iteration::RI = no_report,
) where {RI}
    @argcheck iterations >= 1

    pc_source = PointCloud(source)
    pc_target = PointCloud(target)
    @argcheck size(pc_source) == size(pc_target)

    _rigid_mad(pc_source, pc_target, iterations, report_iteration)
end

function _rigid_mad(
    source::PointCloud{N},
    target::PointCloud{N},
    iterations,
    report_iteration,
) where {N}
    SrcT = eltype(source.points)
    TrgT = eltype(target.points)
    transformation = simple_transformation(source, target)
    prev_transformation = identity_transformation(transformation)
    for iter in 1:iterations
        sum_w = float(zero(eltype(source.weights)) * zero(eltype(target.weights)))
        source_mean = sum_w * zero(SrcT)
        target_mean = sum_w * zero(TrgT)
        covariance = sum_w * zero(TrgT) * zero(SrcT)'

        for j in eachindex(source.points, target.points)
            src = source.points[j]
            trg = target.points[j]
            w_src = source.weights[j]
            w_trg = target.weights[j]
            sqdist = sqeuclidean(transformation(src), trg)
            w_src_w_trg = w_src * w_trg
            w = w_src_w_trg / (sqdist + one(sqdist) / 100)
            source_mean += w * src
            target_mean += w * trg
            covariance += w * trg * src'
            sum_w += w
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

        report_iteration(; iter, transformation)

        if iter > 1 && isapprox(transformation, prev_transformation)
            break
        end
        prev_transformation = transformation
    end

    return transformation
end
