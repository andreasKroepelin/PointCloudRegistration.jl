function transformation_from_moments(covariance, source_mean, target_mean)
    (; U, Vt) = svd(covariance)
    if xor(det(U) < 0, det(Vt) < 0)
        U = negative_last_column(U)
    end
    rotation = U * Vt
    translation = target_mean - rotation * source_mean
    AffineMap(rotation, translation)
end

function negative_last_column(A::SMatrix)
    for i in axes(A, 1)
        @reset A[i, end] *= -1
    end
    A
end

function register_rmsd(source, target)
    _register_naively(PointCloud(source), PointCloud(target))
end

function _register_rmsd(
    source::PointCloud{N, TS},
    target::PointCloud{N, TT},
) where {N, TS, TT}
    check_sizes(source, target)
    T = promote_type(TS, TT)
    covariance = zero(SMatrix{N, N, T})
    for i in eachindex(source.points)
        src = source.points[i] - source.mean
        trg = target.points[i] - target.mean
        w_src = source.weights[i]
        w_trg = target.weights[i]
        covariance += w_src * w_trg * src * trg'
    end
    transformation_from_moments(covariance, source.mean, target.mean)
end
