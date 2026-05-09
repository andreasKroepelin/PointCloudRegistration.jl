using .Adam

struct DistancePreservingRegistration{N, T, PC1 <: PointCloud{N, T}, PC2 <: PointCloud{N, T}}
    source::PC1
    new_source::PC2

    function DistancePreservingRegistration(source::PointCloud{N, T}, new_source_points::VecOfSVec{N, T}) where {N, T}
        new_source = PointCloud(new_source_points, source.weights)
        new{N, T, typeof(source), typeof(new_source)}(source, new_source)
    end
end

function (dpr::DistancePreservingRegistration)(pc::PointCloud)
    @argcheck dpr.source == pc "Distance preserving registration result can only be applied to the source it was computed for."
    return dpr.new_source
end

struct NeighborGraph{T}
    edges::Vector{NTuple{2, Int}}
    distances::Vector{T}
end

struct PreparedSourceDistPres{N, T, PC <: PointCloud{N, T}}
    source::PC
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

    return PreparedSourceDistPres(source, NeighborGraph(edges, distances), init_sqsigma)
end

function nonrigid_distancepreserving(
    source,
    target;
    max_edge_length,
    regularizer = GeneralizedLogNormalRegularizer(2, 1.01),
    iterations = 10_000,
    init_noise = false,
    report_iteration::RI = no_report
) where {RI}
    prepd_source = prepare_source_distancepreserving(source; max_edge_length)
    target_pc = PointCloud(target)
    _nonrigid_distancepreserving(
        prepd_source,
        target_pc,
        regularizer,
        iterations,
        init_noise,
        report_iteration,
    )
end

function nonrigid_distancepreserving(
    prepd_source::PreparedSourceDistPres,
    target;
    regularizer = GeneralizedLogNormalRegularizer(2, 1.01),
    iterations = 10_000,
    init_noise = false,
    report_iteration::RI = no_report
) where {RI}
    target_pc = PointCloud(target)
    _nonrigid_distancepreserving(
        prepd_source,
        target_pc,
        regularizer,
        iterations,
        init_noise,
        report_iteration,
    )
end

function _nonrigid_distancepreserving(
    prepd_source::PreparedSourceDistPres{N},
    target::PointCloud{N},
    regularizer,
    iterations,
    init_noise,
    report_iteration::RI,
) where {N, RI}
    (; source, neighbor_graph, init_sqsigma) = prepd_source

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
    state[1] = sqsigma

    convergence_checker = ConvergenceChecker(state[1], 1000)

    for iter in 1:iterations
        sqsigma = state[1]

        report_iteration(; iter, new_source_points, sqsigma)

        convergence_checker, converged = update_and_check(convergence_checker, sqsigma, iter)
        converged && break

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

        sqsigma_gradient = -dot(vec(C), vec(R)) / sqsigma + N * target.sum_of_weights
        sqsigma_gradient /= 2sqsigma

        gradient[1] = sqsigma_gradient

        Adam.step!(adam, gradient, state, iter)
    end

    DistancePreservingRegistration(new_source_points)
end

struct GeneralizedLogNormalRegularizer{B <: Number}
    beta::B
    coefficient::Float64

    function GeneralizedLogNormalRegularizer(beta, expected_rel_deviation)
        lambda = log(expected_rel_deviation)
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
