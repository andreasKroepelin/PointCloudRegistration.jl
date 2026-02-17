struct BcpdRegistration{C, D}
    correspondences::C
    displacements::D
end

displacements(bcpd::BcpdRegistration) = bcpd.displacements

correspondences(bcpd::BcpdRegistration) = bcpd.correspondences

function register_bcpd(
    source,
    target;
    corr_length,
    expected_displacement,
    outlier_proportion
)
    source_pc = PointCloud(source)
    target_pc = PointCloud(target)
    lambda = dimension(source_pc) / expected_displacement^2
    sqbeta = corr_length^2
    _register_bcpd(source, target, lambda, sqbeta, outlier_proportion)
end

function _register_bcpd(
    source::PointCloud{N, TS},
    target::PointCloud{N, TT},
    lambda,
    sqbeta,
    outlier_p,
) where {N, TS, TT}
    # links:
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8985307
    # https://ieeexplore.ieee.org/ielx7/34/9448371/8985307/supp1-2971687.pdf?arnumber=8985307
    # https://proceedings.neurips.cc/paper/2000/file/19de10adbaa1b2ee13f77f679fa1483a-Paper.pdf
    # https://en.wikipedia.org/wiki/Low-rank_matrix_approximations

    displaced_source_points = similar(source.points)
    displacements = similar(source.points)
    unregularized_displacements = similar(source.points)
    fill!(displacements, zero(eltype(displacements)))
    aligned_target_points = similar(source.points)
    I = length(target.points)
    J = length(source.points)
    R = [sqeuclidean(src, trg) for trg in target.points, src in source.points]
    sqsigma = sum(R) / (I * J * N)
    prev_sqsigma = typemax(sqsigma)
    lambda_invG = let
        G = [
            exp(sqeuclidean(src1, src2) / (-2sqbeta))
            for src1 in source.points, src2 in source.points
        ]
        cholesky_G = cholesky!(Symmetric(G))
        invG = LinearAlgebra.inv!(cholesky_G)
        Symmetric(lmul!(lambda, invG))
    end
    invSigma = similar(lambda_invG)
    b = fill(exp(-N / 2sqsigma) / J, J)
    outlier_preterm = outlier_p / (1 - outlier_p) / bbox_hypervolume(target)
    C = float.(target.weights .* source.weights')
    e = sum(C; dims = 1)
    d = sum(C; dims = 2)
    Z = sum(d)
    avg_displacement_sqsigma = 0

    convergence_counter = 0
    for iter in 1:1000
        # @info "iteration" iter sqrt(sqsigma) convergence_counter
        for j in 1:J
            dsrc = source.points[j] + displacements[j]
            for i in 1:I
                trg = target.points[i]
                R[i, j] = sqeuclidean(dsrc, trg)
            end
        end
        sqsigma = dot(vec(R), vec(C)) / (Z * N) + avg_displacement_sqsigma
        if abs(1 - sqrt(sqsigma / prev_sqsigma)) < 1e-3
            convergence_counter += 1
        else
            convergence_counter = 0
        end
        if convergence_counter > 10
            break
        end
        prev_sqsigma = sqsigma
        expfactor = inv(-2 * sqsigma)
        @. C = target.weights * source.weights' * b' * exp(expfactor * R)
        sum!(d, C)
        outlier_term = outlier_preterm * (2pi * sqsigma)^(N // 2)
        C ./= outlier_term .+ d
        d ./= outlier_term .+ d
        sum!(e, C)
        vec_e = vec(e)
        Z = sum(e)
        mul!(to_matrix(aligned_target_points), to_matrix(target.points), C)
        unregularized_displacements .= aligned_target_points .- vec_e .* source.points

        Sigma = let
            copyto!(invSigma, lambda_invG)
            diagview(invSigma) .+= vec_e ./ sqsigma
            cholesky_invSigma = cholesky!(invSigma)
            Symmetric(LinearAlgebra.inv!(cholesky_invSigma))
        end
        mul!(to_matrix(displacements), to_matrix(unregularized_displacements), Sigma)
        displacements ./= sqsigma
        diag_Sigma = diagview(Sigma)
        @. b = exp(digamma(1 + vec_e) - digamma(J + Z) + expfactor * N * diag_Sigma)
        avg_displacement_sqsigma = dot(diag_Sigma, e) / Z
    end

    return BcpdRegistration(C, displacements)
end
