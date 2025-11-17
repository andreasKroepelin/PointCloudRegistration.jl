function transformation_from_moments(covariance, source_mean, target_mean)
    (; U, Vt) = try
        svd(covariance ./ oneunit(eltype(covariance)))
    catch e
        if e isa LAPACKException
            @warn "Failed SVD!" e covariance
            (U = one(covariance), Vt = one(covariance))
        else
            throw(e)
        end
    end
    if xor(det(U) < 0, det(Vt) < 0)
        U = negative_last_column(U)
    end
    rotation = RotMatrix(U * Vt)
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
$REGISTER_DOCS_EQUAL

$REGISTER_DOCS_TYPES

This is a konvex optimization problem with a closed form solution
([Kabsch algorithm](https://en.wikipedia.org/wiki/Kabsch_algorithm))
but is susceptible to outliers or wrong correspondences.

Use this function if you do not expect outliers or wrong correspondences and
you need maximum speed.
"""
function register_rmsd(source, target)
    pc_source = PointCloud(source)
    pc_target = PointCloud(target)
    @argcheck size(pc_source) == size(pc_target)

    _register_rmsd(pc_source, pc_target)
end

function _register_rmsd(
    source::PointCloud{N, TS},
    target::PointCloud{N, TT},
) where {N, TS, TT}
    @argcheck length(source.points) == length(target.points)
    SrcT = eltype(source.points)
    TrgT = eltype(target.points)
    sum_w = 2 * zero(eltype(source.weights)) * zero(eltype(target.weights))
    covariance = zero_cov(source, target)
    source_mean = sum_w * zero(SrcT)
    target_mean = sum_w * zero(TrgT)
    for i in eachindex(source.points)
        src = source.points[i]
        trg = target.points[i]
        w_src = source.weights[i]
        w_trg = target.weights[i]
        w = w_src * w_trg
        source_mean += w * src
        target_mean += w * trg
        covariance += w * trg * src'
        sum_w += w
    end

    source_mean /= sum_w
    target_mean /= sum_w
    covariance /= sum_w
    covariance -= target_mean * source_mean'
    transformation_from_moments(covariance, source_mean, target_mean)
end
