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
    sample_affinity_matrix
    sample_target_affinity_matrix
end

struct SparseAffinityApproximation{T}
    affinity_matrix::SparseMatrixCSC{T, Int}
end

function build_affinity_matrix(prep, source, target, sigma)
    if sigma < prep.sigma_threshold
        idcss = inrange(prep.target_kdtree, source.points, 3sigma)
        sort!.(idcss)
        values = mapreduce(vcat, idcss, source.points) do idcs, src
            map(idcs) do idx
                exp(sqeuclidean(src, target.points[idx]) / (-2sigma^2))
            end
        end
        return SparseMatrixCSC(
            length(target.points),
            length(source.points),
            [1; cumsum(length.(idcss)) .+ 1],
            reduce(vcat, idcss),
            values
        )
    else
    end
end

function _register_bcpd()
    # links:
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8985307
    # https://ieeexplore.ieee.org/ielx7/34/9448371/8985307/supp1-2971687.pdf?arnumber=8985307
    # https://proceedings.neurips.cc/paper/2000/file/19de10adbaa1b2ee13f77f679fa1483a-Paper.pdf
    # https://en.wikipedia.org/wiki/Low-rank_matrix_approximations
end
