using .Adam

struct NeighborGraph{T}
    edges::Vector{NTuple{2, Int}}
    distances::Vector{T}
end

struct NeighborEdge{T}
    from_to::NTuple{2, Int}
    distance::T
end

struct PreparedSourceDistPres{T, Es <: AbstractVector{NeighborEdge{T}}}
    neighbor_edges::Es
    init_sqsigma::T
end

"""
    prepare_source_distancepreserving(source; max_edge_length)

Perform all the target independent precomputation for the source that is used in
[`nonrigid_registration(source, target, ::DistancePreserving)`](@ref).
This function is especially useful if you plan to register the same source to
multiple targets.

For the meaning of the keyword arguments, see [`DistancePreserving`](@ref).

# Example
Say, you have the three point clouds `source`, `target1`, and `target2` where
you want to register `source` to `target1` and `target2`:
```julia
prep = prepare_source_coherentpointdrift(source; max_edge_length = 10)
transformation1 = nonrigid_registration(source, target1, DistancePreserving(sensitivity = 1.8, rel_deviation = 1e-2); source_preparation = prep)
transformation2 = nonrigid_registration(source, target2, DistancePreserving(sensitivity = 1.2, rel_deviation = 3e-4); source_preparation = prep)
```
"""
function prepare_source_distancepreserving(source; max_edge_length)
    source_pc = PointCloud(source)
    _prepare_source_distancepreserving(source_pc, max_edge_length)
end

function _prepare_source_distancepreserving(source::PointCloud, max_edge_length)
    tree = KDTree(points(source))

    edges = inrange_pairs(tree, max_edge_length)
    distances =
        [euclidean(source[j1].coords, source[j2].coords) for (j1, j2) in edges]
    neighbor_edges = StructArray{NeighborEdge{eltype(distances)}}((edges, distances))

    init_sqsigma = let
        _idcs, dists = allnn(tree)
        (1 * mean(dists))^2
    end

    return PreparedSourceDistPres(neighbor_edges, init_sqsigma)
end

"""
    DistancePreserving(; max_edge_length, sensitivity, rel_deviation[, iterations, report_iteration])

Non-rigid registration of point clouds that tries to preserve distances of
neighboring points in the source.

The algorithm finds new source points by optimizing the regularized matching
score via gradient descent (ADAM).

The matching of (registered) source and target is quantified by a Gaussian
Mixture Model likelihood.
That is, for two weighted point clouds
``x_1, \\dots, x_I \\in \\mathbb{R}^D`` with weights ``p_1, \\dots, p_I`` and
``y_1, \\dots, y_J \\in \\mathbb{R}^D`` with weights ``q_1, \\dots, q_J``,
their matching is
```math
\\prod_{i = 1}^I \\left( \\sum_{j = 1}^J q_j \\exp(-\\Vert x_i - y_j \\Vert^2 / 2 \\sigma^2) \\right)^{p_i}
.
```

For regularization, it is assumed that the ratio of a distance between two
points in the transformed source to their distance in the original source
follows a generalized log-normal distribution.
That is,
for two source points ``y_j`` and ``y_k``
with distance ``d_{j k} = \\Vert y_j - y_k \\Vert``
and the corresponding transformed points ``\\hat{y}_j`` and ``\\hat{y}_k``
with distance ``\\hat{d}_{j k} = \\Vert \\hat{y}_j - \\hat{y}_k \\Vert``
we quantify the deviation of ``\\hat{d}_{j k}`` from ``d_{j k}`` as
```math
\\frac{1}{\\lambda \\hat{d}_{j k} / d_{j k}}
\\exp(- \\vert \\log(\\hat{d}_{j k} / d_{j k}) \\vert^\\beta / \\lambda^\\beta)
.
```

# Parameters
* `max_edge_length`: All pairs of points in the source with a distance up to
  `max_edge_length` are considered for regularization.
  It therefore expresses the length scale of rigid units in the source.
  Registration tends to work better when setting this value rather large.
* `sensitivity`: This value is the ``\\beta`` in the regularizer explained
  above.
  Values between one and two are sensible.
  The parameter deterimines how sensitive the regularizer is regarding
  outliers of distance ratios.
  With `sensitivity = 1`, some larger deviations of neighbor distances are
  permitted by the regularizer, while with `sensitivity = 2`, necessary
  deviations get more evenly distributed between the neighbor edges.
* `rel_deviation`: In terms of the explanation above, this corresponds to
  the average expected ``\\vert 1 - \\hat{d}_{j k} / d_{j k} \\vert``.
  It determines the value of ``\\lambda``.
  That is, `rel_deviation = 0.1` means "distances between neighbors will be
  10 % larger or smaller in the registered source compared to the original".
* `iterations`: How many iterations to perform at most, might stop earlier if
  convergence is detected.
  Default: `100_000`
* `report_iteration`: Callback to run on every iteration.
  Must accept the following keyword arguments:
  * `iteration`: Number of the current iteration.
  * `new_source_points`: Vector of points of the current candidate for the
    registered source.
  * `sqsigma`: Current value of ``\\sigma^2``.
  Default: `(; kwargs...) -> nothing`
"""
@kwdef struct DistancePreserving{
    E <: Union{Number, Nothing},
    S <: Real,
    D <: Real,
    B <: AbstractBatch,
    N <: Number,
    RI,
}
    max_edge_length::E = nothing
    sensitivity::S
    rel_deviation::D
    iterations::Int = 100_000
    batching::B = StochasticBatch(50)
    init_noise::N = false
    report_iteration::RI = no_report
end

"""
    nonrigid_registration(source, target, algorithm::DistancePreserving[; source_preparation])

Perform non-rigid registration via [`DistancePreserving`](@ref).
See [here](@ref nonrigid_registration(::Any, ::Any, ::Any)) for general info
about this function.

This method returns a
[`Displacement`](@ref PointCloudRegistration.Displacement)
that can only be applied to `source`.

# Performance
Some of the necessary computation depends only on the source and can thus be
reused for different targets.
To exploit this, use [`prepare_source_distancepreserving`](@ref) and provide its
result to the `source_preparation` keyword argument.
In this case, the `max_edge_length` parameter of [`DistancePreserving`](@ref)
does not have to be provided (and is ignored if provided).

# Example
```julia
nonrigid_registration(source, target, DistancePreserving(max_edge_length = 10, sensitivity = 1.3, rel_deviation = 1e-3))
```
"""
function nonrigid_registration(
    source,
    target,
    alg::DistancePreserving;
    source_preparation::Union{Nothing, PreparedSourceDistPres} = nothing,
)
    source_pc = PointCloud(source)
    target_pc = PointCloud(target)
    if isnothing(source_preparation)
        @argcheck !isnothing(alg.max_edge_length) "without `source_preparation`, `max_edge_length` must be specified"
        source_preparation =
            prepare_source_distancepreserving(source_pc; alg.max_edge_length)
    end
    _nonrigid_distancepreserving(
        source_pc,
        target_pc,
        source_preparation,
        GeneralizedLogNormalRegularizer(alg.sensitivity, alg.rel_deviation),
        alg.iterations,
        alg.batching,
        alg.init_noise,
        alg.report_iteration,
    )
end

function _nonrigid_distancepreserving(
    source::PointCloud{N},
    target::PointCloud{N},
    prepd_source::PreparedSourceDistPres,
    regularizer,
    iterations,
    batching,
    init_noise,
    report_iteration::RI,
) where {N, RI}
    (; neighbor_edges, init_sqsigma) = prepd_source

    I = length(target)
    J = length(source)

    function extract_points(arr)
        section = @view arr[2:end]
        mat = reshape(section, N, :)
        return to_vec_of_svec(mat, Val(N))
    end

    state = zeros(1 + N * J)
    gradient = similar(state)
    new_source_points = extract_points(state)
    convergence_checker = PointsConvergenceChecker(
        new_source_points,
        1000,
        # iterations ÷ 100,
        minimum(neighbor_edges.distance) / 20,
    )
    points_gradient = extract_points(gradient)
    points_gradient_per_trg = similar(points_gradient)
    adam = Adam.State(state)

    new_source_points .= points(source)
    if !iszero(init_noise)
        V = eltype(new_source_points)
        for j in eachindex(new_source_points)
            new_source_points[j] += init_noise * randn(V)
        end
    end

    target_wsum = sum_of_weights(target)

    state[1] = log(init_sqsigma) / 2

    target_iter = batched(batching, target)
    edge_iter = batched(batching, neighbor_edges)

    for iteration in 1:iterations
        invsqsigma = exp(-2 * state[1])
        exp_factor = -invsqsigma / 2

        report_iteration(; iteration, new_source_points, sqsigma = inv(invsqsigma))

        converged =
            update_and_check!(convergence_checker, new_source_points, iteration)
        # converged && break

        fillzeros!(gradient)
        logsigma_gradient = zero(invsqsigma)

        target_batch_wsum = zero(target_wsum)
        
        for (trg, trg_w) in batch(target_iter, iteration)
            iszero(trg_w) && continue

            sum_of_coeffs = eps(invsqsigma)
            logsigma_gradient_per_trg = zero(logsigma_gradient)
            for (j, (_src, src_w)) in enumerate(source)
                nsrc = new_source_points[j]
                delta = nsrc - trg
                sqdist = norm_sqr(delta)
                coeff = src_w * exp(exp_factor * sqdist)
                sum_of_coeffs += coeff
                points_gradient_per_trg[j] = coeff * delta
                logsigma_gradient_per_trg += coeff * sqdist
            end
            target_batch_wsum += trg_w
            points_gradient .+= (invsqsigma * trg_w / sum_of_coeffs) .* points_gradient_per_trg
            logsigma_gradient += trg_w * (N - invsqsigma * logsigma_gradient_per_trg / sum_of_coeffs)
        end
        # @info "batch" target_batch_wsum
        gradient .*= 2 / target_batch_wsum

        regularize_neighbor_distances!(
            points_gradient,
            new_source_points,
            batch(edge_iter, iteration),
            batch_length_ratio(edge_iter),
            regularizer,
        )

        gradient[1] = logsigma_gradient

        # iteration % 1000 == 0 && @info "magnitude" norm(gradient)

        Adam.step!(adam, gradient, state, iteration)
    end

    Displacement(points(source), new_source_points)
end

struct GeneralizedLogNormalRegularizer{B <: Number}
    beta::B
    coefficient::Float64

    function GeneralizedLogNormalRegularizer(beta, expected_rel_deviation)
        lambda =
            log1p(expected_rel_deviation) * sqrt(gamma(1/beta) / gamma(3/beta))
        coefficient = beta / lambda^beta
        new{typeof(beta)}(beta, coefficient)
    end
end

function regularize_neighbor_distances!(
    points_gradient,
    points,
    neighbor_edges::AbstractVector{<: NeighborEdge},
    additional_factor,
    glnr::GeneralizedLogNormalRegularizer,
)
    (; beta, coefficient) = glnr

    for edge in neighbor_edges
        j1, j2 = edge.from_to
        orig_dist = edge.distance
        p1 = points[j1]
        p2 = points[j2]
        dist = euclidean(p1, p2)
        log_ratio = log(dist / orig_dist)
        factor = 1 + coefficient * abs(log_ratio)^(beta - 1) * sign(log_ratio)
        # d = additional_factor * factor / dist^2 * (p1 - p2)
        d = factor / dist^2 * (p1 - p2) / length(neighbor_edges)
        points_gradient[j1] += d
        points_gradient[j2] -= d
    end
end
