# vendored from Rotations.jl since the upstream version currently has issues
# with the return type
function nearest_rotation(M::StaticMatrix{N,N}) where N
    u, _, v = svd(M)
    s = sign(det(u * v'))
    d = @SVector ones(eltype(M), N-1)
    R = u * Diagonal(push(d,s)) * v'
    return RotMatrix{N}(R)
end

function transformation_from_moments(covariance, source_mean, target_mean)
    unit_free_covariance = covariance ./ oneunit(eltype(covariance))
    rotation = nearest_rotation(unit_free_covariance)
    translation = target_mean - rotation * source_mean
    AffineMap(rotation, translation)
end

"""
    rigid_rmsd(source, target)

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
function rigid_rmsd(source, target)
    pc_source = PointCloud(source)
    pc_target = PointCloud(target)
    @argcheck size(pc_source) == size(pc_target)

    _rigid_rmsd(pc_source, pc_target)
end

function _rigid_rmsd(
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

function evaluate_rmsd(source::PointCloud{N}, target::PointCloud{N}, transformation = identity_transformation(source, target)) where {N}
    @argcheck eachindex(source.points) == eachindex(target.points)
    ssd = mapreduce(+, source.points, source.weights, target.points, target.weights) do src, src_w, trg, trg_w
        src_w * trg_w * sqeuclidean(transformation(src), trg)
    end
    sqrt(ssd / length(source.points))
end
