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

    beta = 2

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
    lambda = 5e-2
    state[1] = sqsigma
    state[2] = lambda

    for iter in 1:5_000
        sqsigma = state[1]
        lambda = state[2]
        
        report_iteration(; iter, new_source_points, sqsigma, lambda)

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

        sum_log_ratios = zero(lambda)
        beta_div_lambda_pow_beta = beta / lambda^beta
        for (equil_dist, (j1, j2)) in zip(equil_dists, spring_pairs)
            nsrc1 = new_source_points[j1]
            nsrc2 = new_source_points[j2]
            dist = euclidean(nsrc1, nsrc2)
            log_ratio = log(dist / equil_dist)
            factor = 1 + beta_div_lambda_pow_beta * abs(log_ratio)^(beta - 1) * sign(log_ratio)
            d = factor / dist^2 * (nsrc1 - nsrc2)
            points_gradient[j1] += d
            points_gradient[j2] -= d
            sum_log_ratios += abs(log_ratio)^beta
        end

        sqsigma_gradient = -dot(vec(C), vec(R)) / sqsigma + N * target.sum_of_weights
        sqsigma_gradient /= 2sqsigma
        lambda_gradient = -beta_div_lambda_pow_beta * sum_log_ratios + length(spring_pairs)
        lambda_gradient /= lambda

        gradient[1] = sqsigma_gradient
        gradient[2] = 0 # lambda_gradient

        Adam.step!(adam, gradient, state, iter)
    end

    PointCloud(new_source_points, source.weights)
end
