struct PreparedSourceCPD{N, T, PC <: PointCloud{N, T}, IGEVL, GEVT}
    source::PC
    gram::Matrix{T}
    inv_gram_eigvalues::IGEVL
    gram_eigvectors::GEVT
end

function _prepare_source_cpd(
    source::PointCloud{N, T},
    regularizer_lengthscale,
)
    neginv2sqregscale = -inv(2 * regularizer_lengthscale^2)
    G = pairwise(sqeuclidean, source.points)
    G .= exp.(neginv2sqregscale .* G)
    eig = eigen(Symmetric(G))
    rk = floor(Int, length(source.points) ^ (1 / 3))
    PreparedSourceCPD(source, G, inv.(eig.values[1:rk]), eig.vectors[:, 1:rk])
end

function register_cpd(source, target)
end

function _register_cpd(
    prepd_source::PreparedSourceCPD{N, T},
    target::PointCloud{N, T},
    sqscale,
    outlier_proportion,
    regularizer_strength,
) where {N, T}
    displacement_spanning_set = [zero(src) for src in source.points]
    displaced_source = [zero(src) for src in source.points]
    ideal_displacement = [zero(src) for src in source.points]
    target_representatives = [zero(src) for src in source.points]

    outlier_term = (
        outlier_proportion / (1 - outlier_proportion)
        * sqrt(2pi * sqscale) ^ N
        * length(source.points) / length(target.points)
    )

    P = zeros(T, length(target.points), length(source.points))
    P_rowsums = zeros(T, length(target.points), 1)
    P_colsums = zeros(T, 1, length(source.points))
    QP = similar(prepd_source.gram_eigvectors)
    neginv2sqscale = -inv(2 * sqscale)

    for iter in 1:100
        for j in eachindex(source.points)
            src = source.points[j]
            coeffs = @view prepd_source.gram[:, j]
            displaced_source[j] = src + wsum(displacement_spanning_set, coeffs)
        end
        pairwise!(P, sqeuclidean, target.points, displaced_source)
        # any(ps -> all(iszero, ps), eachcol(P)) && @error "a column with all zeros"
        # any(isinf, P) && @error "found Inf in distances"
        # any(isnan, P) && @error "found NaN in distances"
        P .= exp.(neginv2sqscale .* P)
        # any(isinf, P) && @error "found Inf in kernels"
        # any(isnan, P) && @error "found NaN in kernels"
        sum!(P_rowsums, P)
        P ./= P_rowsums .+ outlier_term
        # any(isinf, P) && @error "found Inf in normalised P"
        # any(isnan, P) && @error "found NaN in normalised P"

        sum!(P_colsums, P)
        converged = true
        for j in eachindex(source.points)
            src = source.points[j]
            coeffs = @view P[:, j]
            s = P_colsums[j]
            # iszero(s) && @error "zero colsum" j
            # ideal_displacement[j] = wsum(target.points, coeffs) / s - src
            tr = wsum(target.points, coeffs) / s
            if sqeuclidean(tr, target_representative[j]) > eps(T)
                converged = false
            end
            target_representatives[j] = tr
            ideal_displacement[j] = tr - src
            # any(isnan, ideal_displacement[j]) && @error "NaN in ideal_displacement" j ideal_displacement[j]
        end
        if converged
            @info "converged" iter
            break
        end

        # if any(isnan, P_colsums) || any(isnan, G)
        #     @error "found NaN" any(isnan, P_colsums) any(isnan, G)
        # end
        prior_matrix =
            inv(G + regularizer_strength * sqscale * inv(Diagonal(vec(P_colsums))))
        # if any(isnan, prior_matrix) || any(isinf, prior_matrix)
        #     @error "found NaN or Inf in prior" any(isnan, prior_matrix) any(isinf, prior_matrix)
        # end
        for j in eachindex(source.points)
            coeffs = @view prior_matrix[:, j]
            displacement_spanning_set[j] = wsum(ideal_displacement, coeffs)
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
