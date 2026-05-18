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
    Kabsch()

Rigid registration that minimizes the squared distances of corresponding points.

For two weighted point clouds
``x_1, \\dots, x_I \\in \\mathbb{R}^D`` with weights ``p_1, \\dots, p_I`` and
``y_1, \\dots, y_I \\in \\mathbb{R}^D`` with weights ``q_1, \\dots, q_I``
their *Root Mean Square Displacement* (RMSD) is defined as
```math
\\sqrt{
    \\frac{1}{\\sum_{i = 1}^I p_i q_i}
    \\sum_{i = 1}^I p_i q_i \\Vert y_i - x_i \\Vert^2
} .
```
To find the rotation and translation minimising the RMSD between two point
clouds, there exists a closed form solution in form of the Kabsch algorithm.
Therefore, this method is very fast and performs only a single pass over the
point clouds.
However, it requires both point clouds having the same size and points with
equal index corresponding to each other.
It is also very susceptible to outliers.

**Unless you are very sure what you are doing, prefer [`GemanMcClureMM`](@ref)
over `Kabsch` for rigid registration with known correspondences.**
"""
struct Kabsch end

"""
    rigid_registration(source, target, algorithm::Kabsch)

Perform rigid registration via [`Kabsch`](@ref).
See [here](@ref rigid_registration(::Any, ::Any, ::Any)) for general info about
this function.

# Example
We create a similar situation to the example
[for the more robust Geman-McClure loss](@ref rigid_registration(::Any, ::Any, ::GemanMcClureMM))
with an even smaller outlier.
Still, we can observe that it significantly influences the Kabsch algorithm and
the rigid registration returns a result visibly different from the identity
transformation.
```julia
julia> source = randn(2, 50)
2×50 Matrix{Float64}:
  1.77965   0.179855  1.07821  -0.938728  -1.06811   0.560114  -1.03644   …  -1.93775  -0.188104  -1.22085  -1.39789  0.976599  -0.117102
 -0.236629  0.669804  0.51261   0.609834   2.47462  -0.443925  -0.288543      1.00906  -1.16603   -2.34421   1.22799  0.652395   0.370768

julia> target = copy(source);

julia> target[:, 1] .+= 10;

julia> target
2×50 Matrix{Float64}:
 11.7797   0.179855  1.07821  -0.938728  -1.06811   0.560114  -1.03644   …  -1.93775  -0.188104  -1.22085  -1.39789  0.976599  -0.117102
  9.76337  0.669804  0.51261   0.609834   2.47462  -0.443925  -0.288543      1.00906  -1.16603   -2.34421   1.22799  0.652395   0.370768

julia> transformation = rigid_registration(source, target, Kabsch());

julia> transformation.linear
2×2 RotMatrix2{Float64} with indices SOneTo(2)×SOneTo(2):
 0.984029  -0.17801
 0.17801    0.984029

julia> transformation.translation
2-element StaticArraysCore.SVector{2, Float64} with indices SOneTo(2):
 0.22562894646802503
 0.20799830682893525
```
"""
function rigid_registration(source, target, ::Kabsch)
    pc_source = PointCloud(source)
    pc_target = PointCloud(target)
    @argcheck size(pc_source) == size(pc_target)

    _rigid_kabsch(pc_source, pc_target)
end

function _rigid_kabsch(
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
