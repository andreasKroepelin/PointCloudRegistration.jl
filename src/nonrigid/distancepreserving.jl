using .Adam

struct NeighborGraph{T}
    edges::Vector{NTuple{2, Int}}
    distances::Vector{T}
end

struct PreparedSourceDistPres{T}
    neighbor_graph::NeighborGraph{T}
    init_sqsigma::T
end

function prepare_source_distancepreserving(source; max_edge_length)
    source_pc = PointCloud(source)
    _prepare_source_distancepreserving(source_pc, max_edge_length)
end

function _prepare_source_distancepreserving(
    source::PointCloud,
    max_edge_length
)
    tree = KDTree(source.points)

    edges = inrange_pairs(tree, max_edge_length)
    distances = [
        euclidean(source.points[j1], source.points[j2])
        for (j1, j2) in edges
    ]

    init_sqsigma = let
        _idcs, dists = allnn(tree)
        (1 * mean(dists))^2
    end

    return PreparedSourceDistPres(NeighborGraph(edges, distances), init_sqsigma)
end

@kwdef struct DistancePreserving{E <: Number, S <: Real, D <: Real, N <: Number, RI}
    max_edge_length::E
    sensitivity::S = 2
    rel_deviation::D = 0.01
    iterations::Int = 20_000
    init_noise::N = false
    report_iteration::RI = no_report
end

function nonrigid_registration(
    source,
    target,
    alg::DistancePreserving;
    source_preparation::Union{Nothing, PreparedSourceDistPres} = nothing,
)
    source_pc = PointCloud(source)
    target_pc = PointCloud(target)
    if isnothing(source_preparation)
        source_preparation = prepare_source_distancepreserving(source_pc; alg.max_edge_length)
    end
    _nonrigid_distancepreserving(
        source_pc,
        target_pc,
        source_preparation,
        GeneralizedLogNormalRegularizer(alg.sensitivity, alg.rel_deviation),
        alg.iterations,
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
    init_noise,
    report_iteration::RI,
) where {N, RI}
    (; neighbor_graph, init_sqsigma) = prepd_source

    I = length(target.points)
    J = length(source.points)

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
        minimum(neighbor_graph.distances) / 20
    )
    points_gradient = extract_points(gradient)
    adam = Adam.State(state)

    new_source_points .= source.points
    if !iszero(init_noise)
        V = eltype(new_source_points)
        for j in eachindex(new_source_points)
            new_source_points[j] += init_noise * randn(V)
        end
    end

    C = float.(target.weights .* source.weights')
    s = sum(C; dims = 2)
    R = pairwise(sqeuclidean, target.points, source.points)

    sqsigma = init_sqsigma
    state[1] = log(sqsigma)

    @info "before loop"
    for iter in 1:iterations
        sqsigma = exp(state[1])

        report_iteration(; iter, new_source_points, sqsigma)

        converged = update_and_check!(convergence_checker, new_source_points, iter)
        converged && break

        if false && iter % 5000 == 0
            V = eltype(new_source_points)
            noise = sqrt(sqsigma) / 2
            for j in eachindex(new_source_points)
                new_source_points[j] += noise * randn(V)
            end
        end

        exp_factor = -inv(2sqsigma)
        pairwise!(R, sqeuclidean, target.points, new_source_points)
        @. C = source.weights' * exp(exp_factor * R)
        sum!(s, C)
        @. C *= target.weights / s

        fillzeros!(gradient)

        for j in eachindex(source.points)
            nsrc = new_source_points[j]
            for i in eachindex(target.points)
                trg = target.points[i]
                trg_w = target.weights[i]
                points_gradient[j] += trg_w * C[i, j] * (nsrc - trg)
            end
        end
        points_gradient ./= sqsigma

        regularize_neighbor_distances!(
            points_gradient,
            new_source_points,
            neighbor_graph,
            regularizer,
        )

        logsqsigma_gradient = -dot(vec(C), vec(R)) / 2sqsigma + N * target.sum_of_weights
        # logsqsigma_gradient /= 2sqsigma

        gradient[1] = logsqsigma_gradient

        Adam.step!(adam, gradient, state, iter)
    end

    Displacement(source.points, new_source_points)
end

struct GeneralizedLogNormalRegularizer{B <: Number}
    beta::B
    coefficient::Float64

    function GeneralizedLogNormalRegularizer(beta, expected_rel_deviation)
        lambda = log1p(expected_rel_deviation) * sqrt(gamma(1/beta) / gamma(3/beta))
        coefficient = beta / lambda^beta
        new{typeof(beta)}(beta, coefficient)
    end
end

function regularize_neighbor_distances!(
    points_gradient,
    points,
    neighbor_graph::NeighborGraph,
    glnr::GeneralizedLogNormalRegularizer,
)
    (; beta, coefficient) = glnr
    (; edges, distances) = neighbor_graph

    for (orig_dist, (j1, j2)) in zip(distances, edges)
        p1 = points[j1]
        p2 = points[j2]
        dist = euclidean(p1, p2)
        log_ratio = log(dist / orig_dist)
        factor = 1 + coefficient * abs(log_ratio)^(beta - 1) * sign(log_ratio)
        d = factor / dist^2 * (p1 - p2)
        points_gradient[j1] += d
        points_gradient[j2] -= d
    end
end
