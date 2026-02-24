struct BcpdPreparedSource{T, PC <: PointCloud}
    lambda::T
    source::PC
    invlambda_gram_eigvals::Vector{T}
    lambda_gram_inveigvals::Vector{T}
    gram_eigvecs::Matrix{T}
end

function prepare_source_bcpd(source; corr_length, expected_displacement)
    source_pc = PointCloud(source)
    lambda = dimension(source_pc) / expected_displacement^2
    sqbeta = corr_length^2
    _prepare_source_bcpd(source_pc, lambda, sqbeta)
end

function _prepare_source_bcpd(source::PointCloud, lambda, sqbeta)
    G = [
        exp(sqeuclidean(src1, src2) / (-2sqbeta))
        for src1 in source.points, src2 in source.points
    ]
    copy_G = copy(G)
    eigen_G = eigen!(G)
    maxeigval = last(eigen_G.values)
    threshold = maxeigval / 10_000
    first_idx = findfirst(>(threshold), eigen_G.values)
    invlambda_gram_eigvals = eigen_G.values[first_idx:end] ./ lambda
    lambda_gram_inveigvals = inv.(invlambda_gram_eigvals)
    gram_eigvecs = eigen_G.vectors[:, first_idx:end]
    reconstructed_G = gram_eigvecs * Diagonal(invlambda_gram_eigvals) * gram_eigvecs'
    @info "gram approx" first_idx size(gram_eigvecs) extrema(inv(lambda) * copy_G .- reconstructed_G)
    BcpdPreparedSource(lambda, source, invlambda_gram_eigvals, lambda_gram_inveigvals, gram_eigvecs)
end

struct BcpdSigmaBuffers{T}
    buffer1::Matrix{T}
    buffer2::Matrix{T}
    buffer3::Matrix{T}
    buffer4::Matrix{T}
    buffer5::Matrix{T}
    buffer6::Matrix{T}
end

function BcpdSigmaBuffers(prepd_source::BcpdPreparedSource{T}) where {T}
    J, K = size(prepd_source.gram_eigvecs)
    buffer1 = Matrix{T}(undef, J, K)
    buffer2 = Matrix{T}(undef, K, K)
    buffer3 = Matrix{T}(undef, K, K)
    buffer4 = Matrix{T}(undef, K, K)
    buffer5 = Matrix{T}(undef, K, J)
    buffer6 = Matrix{T}(undef, J, J)
    BcpdSigmaBuffers(buffer1, buffer2, buffer3, buffer4, buffer5, buffer6)
end

function compute_Sigma!(buffers::BcpdSigmaBuffers, prepd_source::BcpdPreparedSource, sqsigma, e)
    (; lambda, invlambda_gram_eigvals, lambda_gram_inveigvals, gram_eigvecs) = prepd_source
    (; buffer1, buffer2, buffer3, buffer4, buffer5, buffer6) = buffers
    mul!(buffer1, Diagonal(e), gram_eigvecs)
    mul!(buffer2, gram_eigvecs', buffer1)
    copyto!(buffer3, buffer2)
    diagview(buffer3) .+= sqsigma .* lambda_gram_inveigvals
    cholesky_buffer3 = cholesky!(Symmetric(buffer3))
    inv_buffer3 = LinearAlgebra.inv!(cholesky_buffer3)
    mul!(buffer4, buffer2, inv_buffer3)
    buffer4 .*= -1
    diagview(buffer4) .+= 1
    for (ilge, row) in zip(invlambda_gram_eigvals, eachrow(buffer4))
        row .*= ilge
    end
    mul!(buffer5, buffer4, gram_eigvecs')
    mul!(buffer6, gram_eigvecs, buffer5)

    # invSigma = gram_eigvecs * Diagonal(lambda_gram_inveigvals) * gram_eigvecs' + Diagonal(e ./ sqsigma)
    # @info "Sigma approx" extrema(invSigma - inv(buffer6))

    return buffer6
end

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
    outlier_proportion = 0
)
    source_prepd = prepare_source_bcpd(source; corr_length, expected_displacement)
    target_pc = PointCloud(target)
    _register_bcpd(source_prepd, target, outlier_proportion)
end

function register_bcpd(
    source_prepd::BcpdPreparedSource,
    target;
    outlier_proportion = 0
)
    target_pc = PointCloud(target)
    _register_bcpd(source_prepd, target, outlier_proportion)
end

function _register_bcpd(
    prepd_source::BcpdPreparedSource{T, <: PointCloud{N, TS}},
    target::PointCloud{N, TT},
    outlier_p,
) where {T, N, TS, TT}
    # links:
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8985307
    # https://ieeexplore.ieee.org/ielx7/34/9448371/8985307/supp1-2971687.pdf?arnumber=8985307
    # https://proceedings.neurips.cc/paper/2000/file/19de10adbaa1b2ee13f77f679fa1483a-Paper.pdf
    # https://en.wikipedia.org/wiki/Low-rank_matrix_approximations

    (; source) = prepd_source
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
    buffers = BcpdSigmaBuffers(prepd_source)
    b = fill(exp(-N / 2sqsigma) / J, J)
    outlier_preterm = outlier_p / (1 - outlier_p) / bbox_hypervolume(target)
    C = float.(target.weights .* source.weights')
    e = sum(C; dims = 1)
    d = sum(C; dims = 2)
    Z = sum(d)
    avg_displacement_sqsigma = 0

    convergence_counter = 0
    for iter in 1:1000
        @info "iteration" iter sqrt(sqsigma) convergence_counter maximum(norm, displacements)
        for j in 1:J
            dsrc = source.points[j] + displacements[j]
            for i in 1:I
                trg = target.points[i]
                R[i, j] = sqeuclidean(dsrc, trg)
            end
        end
        sqsigma = dot(vec(R), vec(C)) / (Z * N) + avg_displacement_sqsigma
        if abs(1 - sqrt(sqsigma / prev_sqsigma)) < 1e-5
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

        Sigma = compute_Sigma!(buffers, prepd_source, sqsigma, vec_e)
        mul!(to_matrix(displacements), to_matrix(unregularized_displacements), Sigma)
        displacements ./= sqsigma
        diag_Sigma = diagview(Sigma)
        @. b = exp(digamma(1 + vec_e) - digamma(J + Z) + expfactor * N * diag_Sigma)
        avg_displacement_sqsigma = dot(diag_Sigma, e) / Z
    end

    return BcpdRegistration(C, displacements)
end
