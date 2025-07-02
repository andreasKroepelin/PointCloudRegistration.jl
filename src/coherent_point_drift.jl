function register_cpd(
    source::PointCloud{N, T},
    target::PointCloud{N, T},
    sqscale,
    outlier_proportion,
    regularizer_strength,
    regularizer_lengthscale,
) where {N, T}
    neginv2sqregscale = -inv(2 * regularizer_lengthscale^2)
    G = pairwise(sqeuclidean, source.points)
    G .= exp.(neginv2sqregscale .* G)

    displacement_spanning_set = [zero(src) for src in source.points]
    displaced_source = [zero(src) for src in source.points]
    ideal_displacement = [zero(src) for src in source.points]

    outlier_term = zero(T) # TODO: implement proper term

    P = zeros(T, length(target.points), length(source.points))
    P_rowsums = zeros(T, length(target.points), 1)
    P_colsums = zeros(T, 1, length(source.points))
    neginv2sqscale = -inv(2 * sqscale)

    for iter in 1:10
        for j in eachindex(source.points)
            src = source.points[j]
            coeffs = @view G[:, j]
            displaced_source[j] = src + wsum(displacement_spanning_set, coeffs)
        end
        pairwise!(P, sqeuclidean, target.points, displaced_source)
        P .= exp.(neginv2sqscale .* P)
        sum!(P_rowsums, P)
        P ./= P_rowsums .+ outlier_term

        sum!(P_colsums, P)
        for j in eachindex(source.points)
            src = source.points[j]
            coeffs = @view P[:, j]
            s = P_colsums[j]
            ideal_displacement[j] = wsum(target.points, coeffs) / s - src
        end

        prior_matrix = inv(G + regularizer_strength * sqscale * Diagonal(vec(P_colsums)))
        for j in eachindex(source.points)
            coeffs = @view prior_matrix[:, j]
            displacement_spanning_set[j] = wsum(ideal_displacement, coeffs)
        end
    end

    V = map(eachcol(G)) do coeffs
        wsum(displacement_spanning_set, coeffs)
    end
    V, P

    #=
    function(y)
        coeffs = map(source.points) do src
            exp(neginv2sqregscale * sqeuclidean(y, src))
        end
        wsum(displacement_spanning_set, coeffs)
    end
    =#
end
