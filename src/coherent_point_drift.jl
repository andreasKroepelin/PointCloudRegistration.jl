function register_cpd(
    source::PointCloud{N, T},
    target::PointCloud{N, T};
    outlier_proportion,
    regularizer_strength,
    regularizer_lengthscale,
) where {N, T}
    G = let
        G = pairwise(sqeuclidean, source.points)
        factor = -inv(2 * regularizer_lengthscale^2)
        G .= exp.(factor .* G)
        G
    end

    displacement_basis = [zero(src) for src in source.points]
    displaced_source = [zero(src) for src in source.points]

    outlier_term = zero(T) # TODO: implement proper term

    P = zeros(T, length(source.points), length(target.points))
    P_colsums = zeros(T, 1, length(target.points))
    neginv2sqscale = -inv(2 * sqscale)

    for iter in 1:10
        for j in eachindex(source.points)
            src = source.points[j]
            coeffs = @view G[:, j]
            displaced_source[j] = src + wsum(displacement_basis, coeffs)
        end
        pairwise!(P, sqeuclidean, target.points, displaced_source)
        P .= exp.(neginv2sqscale .* P)
        sum!(P_colsums, P)
        P ./= P_colsums .+ outlier_term

        P1 = Diagonal(sum(P, dims = 2))
        
    end
end
