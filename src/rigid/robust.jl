struct GemanMcClureCost{T}
    sqscale::T
end

mm_weight(gm::GemanMcClureCost, x::AbstractVector, y::AbstractVector) =
    mm_weight(gm, sqeuclidean(x, y))
mm_weight(gm::GemanMcClureCost, sqdist::Number) =
    gm.sqscale / (gm.sqscale + sqdist)^2

function mm_weight_type(gm::GemanMcClureCost, Y::PointCloud, X::PointCloud)
    typeof(mm_weight(gm, first(Y.points), first(X.points)))
end

cost(gm::GemanMcClureCost, x::AbstractVector, y::AbstractVector) =
    cost(gm, sqeuclidean(x, y))
cost(gm::GemanMcClureCost, sqdist::Number) = sqdist / (gm.sqscale + sqdist)

cost_type(gm::GemanMcClureCost, Y::PointCloud, X::PointCloud) =
    typeof(cost(gm, first(Y.points), first(X.points)))

"""
    GemanMcClureMM([; scale, restarts, iterations, batching, report_iteration, report_restart])

Rigid registration of point clouds with known correspondences that is robust
against outliers and some wrong correspondences.

This algorithm performs majorization minimization of the Geman-McClure loss.
The Geman-McClure loss of two weighted point clouds
``x_1, \\dots, x_I \\in \\mathbb{R}^D`` with weights ``p_1, \\dots, p_I`` and
``y_1, \\dots, y_I \\in \\mathbb{R}^D`` with weights ``q_1, \\dots, q_I``
is given as
```math
\\sum_{i = 1}^I p_i q_i \\rho(\\Vert y_i - x_i \\Vert)
```
with
```math
\\rho(r) = \\frac{r^2}{2 \\sigma^2 + r^2} .
```
The scale parameter ``\\sigma`` determines the range of distances the loss is
sensitive to.
``\\rho(r)`` behaves like ``r^2`` for ``r \\ll \\sigma`` and flattens out for
large ``r``.

Note that this assumes both point clouds having the same size and points with
equal index are supposed to correspond to each other.

# Parameters
- `scale`: Determines the value of ``\\sigma`` (see above).
  Can be set to a specific number/collection of numbers or chosen heuristically,
  see Section [Scale parameter](@ref).
  Default: [`LogAnnealingToNearestNeighborDistance()`](@ref)
- `restarts`: Determines how to restart the optimization to avoid local optima,
  see Section [Restarts](@ref).
  Default: `RandomRestarts(5)`
- `iterations`: How many iterations to perform at most, might stop earlier if
  convergence is detected.
  Default: `50`
- `batching`: How to select points from the source in each iteration, see
  Section [Batching](@ref).
  Default: `FullBatch()`
- `report_iteration`: Callback to run on every iteration.
  Must accept the following keyword arguments:
  - `iter`: Number of the current iteration.
  - `annealing_level`: Current value of ``\\sigma^2``.
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
@kwdef struct GemanMcClureMM{S <: ScaleType, R <: AbstractRestarts, B <: AbstractBatch, RI, RR}
    scale::S = LogAnnealingToNearestNeighborDistance()
    restarts::R = RandomRestarts(5)
    iterations::Int = 50
    batching::B = FullBatch()
    report_iteration::RI = no_report
    report_restart::RR = no_report
end

"""
    rigid_registration(source, target, algorithm::GemanMcClureMM, [flip = NoFlip()])

Perform rigid registration via [`GemanMcClureMM`](@ref).
See [here](@ref rigid_registration(::Any, ::Any, ::Any, ::Any)) for general info
about this function.

# Example
This demonstrates the robustness against outliers of the Geman-McClure cost by
registering a point cloud with itself but creating one severe outlier.
In the end, we still obtain an identity matrix and a zero vector as rotation and
translation, respectively.
```julia
julia> source = randn(2, 50)
2×50 Matrix{Float64}:
  0.288016   1.3289   1.07965  -1.14399   1.48149    0.2409    1.11254  …  -0.799255  -0.454939  0.600287  -0.0679249  -2.12803   -1.42089
 -0.673326  -1.38039  0.5257    1.64189  -0.317935  -0.834046  1.33805      0.476037  -0.801183  0.148831  -1.16099     0.882929  -1.94498

julia> target = copy(source);

julia> target[:, 1] .+= 1000; # severe outlier

julia> target
2×50 Matrix{Float64}:
 1000.29    1.3289   1.07965  -1.14399   1.48149    0.2409    1.11254  …  -0.799255  -0.454939  0.600287  -0.0679249  -2.12803   -1.42089
  999.327  -1.38039  0.5257    1.64189  -0.317935  -0.834046  1.33805      0.476037  -0.801183  0.148831  -1.16099     0.882929  -1.94498

julia> transformation = rigid_registration(source, target, GemanMcClureMM());

julia> transformation.linear
2×2 RotMatrix2{Float64} with indices SOneTo(2)×SOneTo(2):
 1.0         -4.68232e-6
 4.68232e-6   1.0

julia> transformation.translation
2-element StaticArraysCore.SVector{2, Float64} with indices SOneTo(2):
 1.3242532771004512e-5
 1.3159778492727314e-5
```
"""
function rigid_registration(source, target, alg::GemanMcClureMM, flip::FlipMarker = NoFlip())
    @argcheck alg.iterations >= 1

    source_pc = PointCloud(source)
    target_pc = PointCloud(target)
    @argcheck size(source_pc) == size(target_pc)

    sqscales = annealing_plan(target_pc, alg.scale)
    _rigid_gmc(
        source_pc,
        target_pc,
        flip,
        sqscales,
        alg.restarts,
        alg.iterations,
        alg.batching,
        alg.report_iteration,
        alg.report_restart,
    )
end

function _rigid_gmc(
    source::PointCloud{N},
    target::PointCloud{N},
    flip,
    sqscales,
    restarts,
    iterations,
    batching,
    report_iteration,
    report_restart,
) where {N}
    gm = GemanMcClureCost(oneunit(eltype(sqscales)))
    SrcT = eltype(source.points)
    TrgT = eltype(target.points)
    CostT = cost_type(gm, source, target)
    WeightT = mm_weight_type(gm, source, target)
    best = worst(CostT, transformation_type(source, target, flip))
    gm_cost = zero(CostT)
    restarts_iter = restarts_iterator(source, target, restarts, flip)
    source_iter = point_cloud_iterator(batching, source)
    for (restart, transformation) in enumerate(restarts_iter)
        for sqscale in sqscales
            # double `sqscale` such that the loss function has the same
            # quadratic behavior for small distances as the kernel correlation
            # loss with `sqscale`
            gm = GemanMcClureCost(2sqscale)
            prev_transformation = identity_transformation(transformation)
            for iter in 1:iterations
                if 10iter > 9iterations
                    source_iter = non_stochastic(source_iter)
                end
                sum_w = zero(WeightT)
                source_mean = sum_w * zero(SrcT)
                target_mean = sum_w * zero(TrgT)
                covariance = sum_w * zero(TrgT) * zero(SrcT)'
                gm_cost = zero(CostT)

                for source_element in source_iter
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
                    flip,
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
    end

    return best.transformation
end

"""
Rigid registration of point clouds with known correspondences that is somewhat
robust against outliers and wrong correspondences (more robust than
[`Kabsch`](@ref), less robust than [`GemanMcClureMM`](@ref)).

This algorithm performs majorization minimization of the mean absolute
deviation.
For two weighted point clouds
``x_1, \\dots, x_I \\in \\mathbb{R}^D`` with weights ``p_1, \\dots, p_I`` and
``y_1, \\dots, y_I \\in \\mathbb{R}^D`` with weights ``q_1, \\dots, q_I``
it is given as
```math
\\frac{1}{\\sum_{i = 1}^I p_i q_i}
\\sum_{i = 1}^I p_i q_i \\Vert y_i - x_i \\Vert
```

The optimisation problem is convex and only has one local optimum so this needs
no restarts or annealing.

# Parameters
- `iterations`: How many iterations to perform at most, might stop earlier if
  convergence is detected.
  Default: `50`
- `report_iteration`: Callback to run on every iteration.
  Must accept the following keyword arguments:
  - `iter`: Number of the current iteration.
  - `transformation`: Currently best found transformation.
  Default: `(; kwargs...) -> nothing`
"""
@kwdef struct MeanAbsoluteDeviationMM{RI}
    iterations::Int = 50
    report_iteration::RI = no_report
end

"""
    rigid_registration(source, target, algorithm::MeanAbsoluteDeviationMM, [flip = NoFlip()])

Perform rigid registration via [`MeanAbsoluteDeviationMM`](@ref).
See [here](@ref rigid_registration(::Any, ::Any, ::Any, ::Any)) for general info
about this function.
"""
function rigid_registration(source, target, alg::MeanAbsoluteDeviationMM, flip::FlipMarker = NoFlip())
    @argcheck alg.iterations >= 1

    pc_source = PointCloud(source)
    pc_target = PointCloud(target)
    @argcheck size(pc_source) == size(pc_target)

    _rigid_mad(pc_source, pc_target, flip, alg.iterations, alg.report_iteration)
end

function _rigid_mad(
    source::PointCloud{N},
    target::PointCloud{N},
    flip,
    iterations,
    report_iteration,
) where {N}
    SrcT = eltype(source.points)
    TrgT = eltype(target.points)
    transformation = simple_transformation(source, target, flip)
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
            flip,
        )

        report_iteration(; iter, transformation)

        if iter > 1 && isapprox(transformation, prev_transformation)
            break
        end
        prev_transformation = transformation
    end

    return transformation
end
