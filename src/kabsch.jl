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

function register_naively(source, target)
    _register_naively(PointCloud(source), PointCloud(target))
end

function _register_naively(source::PointCloud{N, TS}, target::PointCloud{N, TT}) where {N, TS, TT}
    check_sizes(source, target)
    T = promote_type(TS, TT)
    covariance = zero(SMatrix{N, N, T})
    for i in eachindex(weights(source))
        src = points(source)[i] - source.mean
        trg = points(target)[i] - target.mean
        w_src = weights(source)[i]
        w_trg = weights(target)[i]
        covariance += w_src * w_trg * src * trg'
    end
    transformation_from_moments(covariance, source.mean, target.mean)
end
