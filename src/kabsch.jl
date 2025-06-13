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

"""
    register_rmsd(source, target)

$REGISTER_DOCS_START
minimizes the Root Mean Square Distance to
`target`, i.e.
```math
% \\operatorname{arg\\;min}\\limits_{
%     \\text{rotation } R \\text{ and  translation } t
% }
\\sqrt{ \\frac{1}{n} \\sum_{i = 1}^n \\Vert R y_i + t - x_i \\Vert^2 }
```
$REGISTER_DOCS_SYMBOLS
This assumes that the ``i``-th point in `source` corresponds to the ``i``-th
point in `target`, so `source` and `target` must have the same size.

`source` and `target` can each either be matrices with one point per column
or [`PointCloud`](@ref)s.

This is a konvex optimization problem with a closed form solution
([Kabsch algorithm](https://en.wikipedia.org/wiki/Kabsch_algorithm))
but is susceptible to outliers or wrong correspondences.

Use this function if you do not expect outliers or wrong correspondences and
you need maximum speed.
"""
function register_rmsd(source, target)
    _register_rmsd(PointCloud(source), PointCloud(target))
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
