struct PreparedSourceCPD{N, T, PC <: PointCloud{N, T}, IGEVL, GEVT}
    source::PC
    gram::Matrix{T}
    inv_gram_eigvalues::IGEVL
    gram_eigvectors::GEVT
end

function prepare_source_cpd(
    source::PointCloud{N, T};
    regularizer_lengthscale,
) where {N, T}
    _prepare_source_cpd(source, regularizer_lengthscale)
end

function _prepare_source_cpd(
    source::PointCloud{N, T},
    regularizer_lengthscale,
) where {N, T}
    neginv2sqregscale = -inv(2 * regularizer_lengthscale^2)
    G = pairwise(sqeuclidean, source.points)
    G .= exp.(neginv2sqregscale .* G)
    eig = eigen(Symmetric(G))
    rk = floor(Int, length(source.points) ^ (1 / 3))
    PreparedSourceCPD(
        source,
        G,
        inv.(eig.values[(end - rk):end]),
        permutedims(eig.vectors[:, (end - rk):end]),
    )
end

function register_cpd(
    source::PointCloud{N, T},
    target::PointCloud{N, T};
    scale,
    outlier_proportion,
    regularizer_strength,
    regularizer_lengthscale,
) where {N, T}
    prepd_source = _prepare_source_cpd(source, regularizer_lengthscale)
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
    source = prepd_source.source
    displacement_spanning_set = [zero(src) for src in source.points]
    displaced_source = [zero(src) for src in source.points]
    ideal_displacement = [zero(src) for src in source.points]
    target_representatives = [zero(src) for src in source.points]

    outlier_term = (
        outlier_proportion / (1 - outlier_proportion) *
        sqrt(2pi * sqscale) ^ N *
        length(source.points) / length(target.points)
    )

    P = zeros(T, length(target.points), length(source.points))
    P_rowsums = zeros(T, length(target.points), 1)
    P_colsums = zeros(T, 1, length(source.points))
    QP = similar(prepd_source.gram_eigvectors)
    rk = size(prepd_source.gram_eigvectors, 1)
    QPQ = similar(prepd_source.gram_eigvectors, rk, rk)
    prior_coeffs = similar(prepd_source.gram_eigvectors, rk)
    neginv2sqscale = -inv(2 * sqscale)

    for iter in 1:100
        for j in eachindex(source.points)
            src = source.points[j]
            coeffs = @view prepd_source.gram[:, j]
            displaced_source[j] = src + wsum(displacement_spanning_set, coeffs)
        end
        pairwise!(P, sqeuclidean, target.points, displaced_source)
        P .= exp.(neginv2sqscale .* P)
        sum!(P_rowsums, P)
        P ./= P_rowsums .+ outlier_term

        sum!(P_colsums, P)
        converged = true
        for j in eachindex(source.points)
            src = source.points[j]
            coeffs = @view P[:, j]
            s = P_colsums[j]
            iszero(s) && error("No target point contributes to source point $j")
            tr = wsum(target.points, coeffs) / s
            if sqeuclidean(tr, target_representatives[j]) > eps(T)
                converged = false
            end
            target_representatives[j] = tr
            ideal_displacement[j] = tr - src
        end
        if converged
            @info "converged" iter
            break
        end

        P_colsums ./= regularizer_strength * sqscale
        mul!(QP, prepd_source.gram_eigvectors, Diagonal(vec(P_colsums)))
        mul!(QPQ, QP, prepd_source.gram_eigvectors')
        QPQ .+= Diagonal(prepd_source.inv_gram_eigvalues)
        small_inv = LinearAlgebra.inv!(lu!(QPQ))
        # prior_matrix =
        #     inv(G + regularizer_strength * sqscale * inv(Diagonal(vec(P_colsums))))
        for j in eachindex(source.points)
            mul!(prior_coeffs, small_inv, view(QP, :, j))
            # coeffs = @view prior_matrix[:, j]
            displacement_spanning_set[j] =
                wsum(ideal_displacement, prior_coeffs)
        end
    end

    V = map(eachcol(prepd_source.gram)) do coeffs
        wsum(displacement_spanning_set, coeffs)
    end
    # V = ideal_displacement
    V, P, target_representatives

    #=
    function(y)
        coeffs = map(source.points) do src
            exp(neginv2sqregscale * sqeuclidean(y, src))
        end
        wsum(displacement_spanning_set, coeffs)
    end
    =#
end
