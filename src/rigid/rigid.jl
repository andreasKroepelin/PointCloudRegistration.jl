"""
    WithFlip()

Indicates that rigid registration will search for the optimal rotation,
translation, *and reflection*.
That is, the full Euclidean group is eligible for the returned motion.
"""
struct WithFlip end

"""
    NoFlip()

Indicates that rigid registration will search for the optimal rotation and
translation, *disallowing reflections*.
That is, only the Special Euclidean group is eligible for the returned motion.
"""
struct NoFlip end
const FlipMarker = Union{WithFlip, NoFlip}

function transformation_from_moments(
    covariance,
    source_mean,
    target_mean,
    flip::FlipMarker,
)
    unit_free_covariance = covariance ./ oneunit(eltype(covariance))
    rotation = nearest_orthogonal(unit_free_covariance, flip)
    translation = target_mean - rotation * source_mean
    AffineMap(rotation, translation)
end

"""
    rigid_registration(source, target, [flip = NoFlip()]; correspondences = Unknown())

Find a [`CoordinateTransformations.AffineMap`](@extref) that rotates and
translates (and possibly reflects if `flip isa WithFlip`) the point cloud
`source` to "match" the point cloud `target`.

Both `source` and `target` can be given in a form described in Section
[Representing point clouds](@ref).
They must have matching dimensions (both 2D or both 3D and so forth).

By default, it is assumed that no correspondences between points in `source` and
`target` are known (`correspondences = Unknown()`).
If you do have such correspondences available, it is strongly recommended to
utilize them.
For example, if `source` and `target` are organized such that points with equal
index correspond to each other (1st point in `source` corresponds to 1st point
in `target` etc.), set `correspondences = Ordered()`.
This makes the registration faster and more precise.

!!! details "Default algorithms"
    Depending on the keyword argument `correspondences`, `rigid_registration`
    uses these algorithms:
    - `correspondences = Unknown()`: Maximization of the Kernel
      Correlation ([`KernelCorrelationMM`](@ref)), tries to maximize the
      similarity of densities obtained by Gaussian "blurring" of `source` and
      `target`.
    - Otherwise: Minimization of the Geman-McClure cost
      ([`GemanMcClureMM`](@ref)), brings corresponding points close together but
      is robust against outliers.
"""
function rigid_registration(
    source,
    target,
    flip::FlipMarker = NoFlip();
    correspondences = Unknown(),
)
    source_pc = PointCloud(source)
    target_pc = PointCloud(target)
    @argcheck dimension(source_pc) == dimension(target_pc)

    if correspondences isa Unknown
        return rigid_registration(source_pc, target_pc, KernelCorrelationMM(), flip)
    else
        return rigid_registration(source_pc, target_pc, GemanMcClureMM(), flip; correspondences)
    end
end

"""
    rigid_registration(source, target, algorithm, [flip = NoFlip()])

Find a [`CoordinateTransformations.AffineMap`](@extref) that rotates and
translates (and possibly reflects if `flip isa WithFlip`) the point cloud
`source` to "match" the point cloud `target` using
`algorithm`.
See [here](#Rigid-registration-algorithms) for a list of available algorithms.

This method is intended for more fine grained control over the registration.
Alternatively, [`rigid_registration(source, target; correspondences)`](@ref) is
available for leaving the choice of the algorithm to a rule of thumb.

Both `source` and `target` can be given in a form described in Section
[Representing point clouds](@ref).
They must have matching dimensions (both 2D or both 3D and so forth).
"""
function rigid_registration(source, target, alg, flip)
    error("Unsopported algorithm of type ", typeof(alg))
end

"""
    rigidly_registered(source, target, args...; kwargs...)

Performs rigid registration via [`rigid_registration`](@ref) and directly
applies the computed motion to the source.
That is, this function returns the registered source.
Additional (keyword) arguments are propagated to [`rigid_registration`](@ref).
"""
function rigidly_registered(source, target, args...; kwargs...)
    source_pc = PointCloud(source)
    motion = rigid_registration(source_pc, target, args...; kwargs...)
    return motion(source_pc)
end
