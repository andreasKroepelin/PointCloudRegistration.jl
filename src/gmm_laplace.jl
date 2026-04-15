using .Adam

function nonrigid_gmml(source::PointCloud{N}, target::PointCloud{N}; max_spring_length, report_iteration = no_report) where {N}
    I = length(target.points)
    J = length(source.points)

    source_kdtree = KDTree(source.points)
    spring_pairs = inrange_pairs(source_kdtree, max_spring_length)
    equil_dists = [
        euclidean(source.points[j1], source.points[j2])
        for (j1, j2) in spring_pairs
    ]

    sqlambda = let
        _idcs, nn_dists = allnn(source_kdtree)
        minimum(nn_dists)^2
    end

    function extract_points(arr)
        section = @view arr[3:end]
        mat = reshape(section, N, :)
        return to_vec_of_svec(mat, Val(N))
    end

    state = zeros(2 + N * J)
    gradient = similar(state)
    new_source_points = extract_points(state)
    points_gradient = extract_points(gradient)
    adam = Adam.State(state)

    new_source_points .= source.points
    # for j in eachindex(new_source_points)
    #     new_source_points[j] += randn(eltype(new_source_points))
    # end

    C = float.(target.weights .* source.weights')
    s = sum(C; dims = 2)
    R = pairwise(sqeuclidean, target.points, source.points)

    sqsigma = 5.0^2 # dot(vec(C), vec(R)) / (2 * N * sum(s))
    state[1] = sqsigma
    state[2] = sqlambda

    for iter in 1:50_000
        sqsigma = state[1]
        sqlambda = state[2]
        
        report_iteration(; iter, new_source_points, sqsigma, sqlambda)

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

        ssq_displacement = zero(sqlambda)
        for (equil_dist, (j1, j2)) in zip(equil_dists, spring_pairs)
            nsrc1 = new_source_points[j1]
            nsrc2 = new_source_points[j2]
            dist = euclidean(nsrc1, nsrc2)
            displacement = dist - equil_dist
            d = inv(sqlambda) * displacement / dist * (nsrc1 - nsrc2)
            points_gradient[j1] += d
            points_gradient[j2] -= d
            ssq_displacement += displacement^2
        end

        sqsigma_gradient = -dot(vec(C), vec(R)) / sqsigma + N * target.sum_of_weights
        sqsigma_gradient /= 2sqsigma
        sqlambda_gradient = -ssq_displacement / sqlambda + length(spring_pairs)
        sqlambda_gradient /= 2sqlambda

        gradient[1] = sqsigma_gradient
        gradient[2] = sqlambda_gradient

        Adam.step!(adam, gradient, state, iter)
    end

    PointCloud(new_source_points, source.weights)
end
