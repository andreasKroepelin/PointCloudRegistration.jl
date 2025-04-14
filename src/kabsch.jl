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
