rot_from_cov!!(R::AbstractMatrix, C::AbstractMatrix) =
    _rot_from_cov!!(Size(Cov), R, C)

_rot_from_cov!!(::Size{Sz}, R, C) where {Sz} = _rot_from_cov!!(Sz, R, C)

function static_rot_from_cov(C)
    (; U, Vt) = svd(C)
    if xor(det(U) < 0, det(Vt) < 0)
        U = negative_last_column(U)
    end
    U * Vt
end

function negative_last_column(A::SMatrix)
    for i in axes(A, 1)
        @reset A[i, end] *= -1
    end
    A
end

function dynamic_rot_from_cov!(R, C)
    (; U, Vt) = svd!(C)
    mul!(R, U, Vt)
    if det(R) < 0
        @view(U[:, end]) .*= -1
        mul!(R, U, Vt)
    end
    R
end
