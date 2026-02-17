struct PreparedSourceCPD{N, T, PC <: PointCloud{N, T}, G <: Matrix}
    source::PC
    gram::G
end

function prepare_source_cpd(
    source::PointCloud{N, T};
    regularizer_lengthscale,
) where {N, T}
    neginv2sqregscale = -inv(2 * regularizer_lengthscale^2)
    # GT = exp(neginv2sqregscale * sqeuclidean())
    G = map(Iterators.product(source.points, source.points)) do (x, y)
        exp(neginv2sqregscale * sqeuclidean(x, y))
    end
    # G = Matrix{}
    # sqdists = pairwise(sqeuclidean, source.points)
    # G = exp.(neginv2sqregscale .* sqdists)
    PreparedSourceCPD(source, G)
end

@kwdef struct CpdResult{N, T, T1}
    displacement::Vector{SVector{N, T}}
    correspondences::Matrix{T1}
    target_representatives::Vector{SVector{N, T}}
end

function register_cpd(
    source::PointCloud{N, T},
    target::PointCloud{N, T};
    scale,
    outlier_proportion,
    regularizer_strength,
    regularizer_lengthscale,
) where {N, T}
    prepd_source = prepare_source_cpd(source; regularizer_lengthscale)
    _register_cpd(
        prepd_source,
        target,
        scale^2,
        outlier_proportion,
        regularizer_strength,
    )
end

function register_cpd(
    prepd_source::PreparedSourceCPD{N, T},
    target::PointCloud{N, T};
    scale,
    outlier_proportion,
    regularizer_strength,
) where {N, T}
    _register_cpd(
        prepd_source,
        target,
        scale^2,
        outlier_proportion,
        regularizer_strength,
    )
end

function _register_cpd(
    prepd_source::PreparedSourceCPD{N, T},
    target::PointCloud{N, T},
    sqscale,
    outlier_proportion,
    regularizer_strength,
) where {N, T}
    T1 = typeof(one(T))
    source = prepd_source.source
    displacement_spanning_set = [zero(src) for src in source.points]
    displaced_source = [zero(src) for src in source.points]
    ideal_displacement = [zero(src) for src in source.points]
    target_representatives = [zero(src) for src in source.points]

    outlier_term = (
        outlier_proportion / (1 - outlier_proportion) *
        sqrt(2pi * sqscale) ^ N *
        length(source.points) / bbox_hypervolume(target)
    )

    P = zeros(T1, length(target.points), length(source.points))
    P_rowsums = zeros(T1, length(target.points), 1)
    P_colsums = zeros(T1, 1, length(source.points))
    GP = similar(prepd_source.gram)
    w = zeros(T, length(source.points))
    d = zeros(T, length(source.points))
    neginv2sqscale = -inv(2 * sqscale)
    regularizer_strength_sqscale = regularizer_strength * sqscale
    # @info "type" outlier_term regularizer_strength_sqscale

    for iter in 1:10_000
        try
            for j in eachindex(source.points)
                src = source.points[j]
                coeffs = @view prepd_source.gram[:, j]
                displaced_source[j] =
                    src + wsum(displacement_spanning_set, coeffs)
            end
            for ci in CartesianIndices(P)
                i, j = Tuple(ci)
                sqdist = sqeuclidean(target.points[i], displaced_source[j])
                w_src_w_trg = source.weights[j] * target.weights[i]
                P[ci] = w_src_w_trg * exp(neginv2sqscale * sqdist)
            end
            # map!(P, Iterators.product(target.points, displaced_source)) do (x, y)
            #                                     exp(neginv2sqscale * sqeuclidean(x, y))
            #                                                     end
            # pairwise!(P, sqeuclidean, target.points, displaced_source)
            # P .= exp.(neginv2sqscale .* P)
            sum!(P_rowsums, P)
            P ./= P_rowsums .+ outlier_term

            sum!(P_colsums, P)
            change = sqeuclidean(
                zero(eltype(target_representatives)),
                zero(eltype(target_representatives)),
            )
            for j in eachindex(source.points)
                src = source.points[j]
                coeffs = @view P[:, j]
                s = P_colsums[j]
                iszero(s) &&
                    error("No target point contributes to source point $j")
                tr = wsum(target.points, coeffs) / s
                change += sqeuclidean(tr, target_representatives[j])
                target_representatives[j] = tr
                ideal_displacement[j] = tr - src
            end
            relchange = change / length(source.points)
            # @info "iteration" iter change sqrt(relchange)
            if relchange < sqscale / 10_000
                # @info "converged" iter
                break
            end

            p = vec(P_colsums)
            copyto!(GP, prepd_source.gram)
            GP[diagind(GP)] .+= regularizer_strength_sqscale ./ p
            chol = cholesky!(Symmetric(GP))
            W = reinterpret(reshape, T, displacement_spanning_set)
            D = reinterpret(reshape, T, ideal_displacement)
            for (D_row, W_row) in zip(eachrow(D), eachrow(W))
                # Need to go via extra vectors because `ldiv( , ::Cholesky, )`
                # expects continuous arrays whereas `D_row` and `W_row` have stride
                # `N`.
                copyto!(d, D_row)
                ldiv!(w, chol, d)
                copyto!(W_row, w)
            end
        catch e
            if e isa InterruptException
                break
            else
                throw(e)
            end
        end
    end

    CpdResult(;
        displacement = map(eachcol(prepd_source.gram)) do coeffs
            wsum(displacement_spanning_set, coeffs)
        end,
        correspondences = P,
        target_representatives,
    )
end

struct LargeAffinityMatrixPrep
    target_kdtree
    target_affinity_matrix
    sigma_threshold
end

struct NystromAffinityApproximation
    source_sample_affinity_matrix
    inv_sample_affinity_matrix
    sample_target_affinity_matrix
end

function NystromAffinityApproximation(prep, source, target, sigma)
    num_samples = 1000
end

struct SparseAffinityApproximation{T}
    affinity_matrix::SparseMatrixCSC{T, Int}
end

function SparseAffinityApproximation(prep, source, target, sigma)
    idcss = inrange(prep.target_kdtree, source.points, 3sigma)
    sort!.(idcss)
    values = mapreduce(vcat, idcss, source.points) do idcs, src
        map(idcs) do idx
            exp(sqeuclidean(src, target.points[idx]) / (-2sigma^2))
        end
    end
    matrix = SparseMatrixCSC(
        length(target.points),
        length(source.points),
        [1; cumsum(length.(idcss)) .+ 1],
        reduce(vcat, idcss),
        values
    )

    return SparseAffinityApproximation(matrix)
end

function build_affinity_matrix(prep, source, target, sigma)
    if sigma < prep.sigma_threshold
        SparseAffinityApproximation(prep, source, target, sigma)
    else
        NystromAffinityApproximation(prep, source, target, sigma)
    end
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
