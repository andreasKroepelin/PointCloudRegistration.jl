struct GemanMcclure{T}
    sqscale::T
end

mm_weight(gm::GemanMcclure, x::AbstractVector, y::AbstractVector) =
    mm_weight(gm, sqeuclidean(x, y))
mm_weight(gm::GemanMcclure, sqdist::Real) = gm.sqscale / (gm.sqscale + sqdist)^2

cost(gm::GemanMcclure, x::AbstractVector, y::AbstractVector) =
    cost(gm, sqeuclidean(x, y))
cost(gm::GemanMcclure, sqdist::Real) = sqdist / (gm.sqscale + sqdist)

"""
    register_gmc(source, target[; scale, restarts, iterations, rng, accumulator])

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
function register_gmc(
    source,
    target;
    scale::ScaleType = default_scale(),
    restarts::Int = default_restarts(),
    iterations::Int = default_iterations(),
    rng = Random.default_rng(),
    # use type parameters `RI` and `RR` here to force specialization
    report_iteration::RI = no_report,
    report_restart::RR = no_report,
) where {RI, RR}
    @argcheck restarts >= 0
    @argcheck iterations >= 1

    pc_source = PointCloud(source)
    pc_target = PointCloud(target)
    @argcheck size(pc_source) == size(pc_target)

    sqscales = annealing_plan(pc_target, scale)
    _register_gmc(
        pc_source,
        pc_target,
        sqscales,
        restarts,
        iterations,
        rng,
        report_iteration,
        report_restart,
    )
end

function _register_gmc(
    source::PointCloud{N, TS},
    target::PointCloud{N, TT},
    sqscales,
    restarts,
    iterations,
    rng,
    report_iteration,
    report_restart,
) where {N, TS, TT}
    T = promote_type(TS, TT)

    best = worst(transformation_type(Val(N), T))
    gm_cost = zero(T)
    transformation = simple_transformation(source, target)
    for restart in 0:restarts
        for sqscale in sqscales
            # double `sqscale` such that the loss function has the same
            # quadratic behavior for small distances as the kernel correlation
            # loss with `sqscale`
            gm = GemanMcclure(2sqscale)
            prev_transformation = identity_transformation(transformation)
            for iter in 1:iterations
                source_mean = zero(eltype(source.points))
                target_mean = zero(eltype(target.points))
                covariance = zero(rotation_type(source, target))
                sum_w = zero(T)
                gm_cost = zero(T)

                correspondences = zip(
                    source.points,
                    source.weights,
                    target.points,
                    target.weights,
                )
                for (src, w_src, trg, w_trg) in correspondences
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
        best = better(best, TransformationWithCost(gm_cost, transformation))
        report_restart(; restart, cost = gm_cost, transformation)
        transformation = rand_transformation(rng, source, target)
    end

    return best.transformation
end
