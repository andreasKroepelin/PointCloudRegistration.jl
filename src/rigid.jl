"""
    rigid_registration(source, target; ordered::Bool = false)

Find a [`CoordinateTransformations.AffineMap`](@extref) that rotates and
translates the point cloud `source` to "match" the point cloud `target`.

Both `source` and `target` can be given in a form described in Section
[Representing point clouds](@ref).
They must have matching dimensions (both 2D or both 3D and so forth).

By default, it is assumed that no correspondences between points in `source` and
`target` are known (`ordered = false`).
If you do have such correspondences available, it is strongly recommended to
organize `source` and `target` such that points with equal index correspond to
each other (1st point in `source` corresponds to 1st point in `target` etc.) and
set `ordered = true`.
This makes the registration faster and more precise.

!!! details "Default algorithms"
    Depending on the keyword argument `ordered`, `rigid_registration` uses these
    algorithms:
    - `ordered = false`: Majorization Minimization of the Kernel Correlation
      ([`KernelCorrelationMM`](@ref)), tries to maximize the similarity of
      densities obtained by Gaussian "blurring" `source` and `target`.
    - `ordered = true`: Majorization Minimization of the Geman-McClure cost
      ([`GemanMcClureMM`](@ref)), brings corresponding points close together but
      is robust against outliers.
"""
function rigid_registration(source, target; ordered = false)
    source_pc = PointCloud(source)
    target_pc = PointCloud(target)
    @argcheck dimension(source_pc) == dimension(target_pc)

    if ordered
        rigid_registration(source_pc, target_pc, GemanMcClureMM())
    else
        rigid_registration(source_pc, target_pc, KernelCorrelationMM())
    end
end

"""
    rigid_registration(source, target, algorithm)

Find a [`CoordinateTransformations.AffineMap`](@extref) that rotates and
translates the point cloud `source` to "match" the point cloud `target` using
`algorithm`.
See [here](#Rigid-registration-algorithms) for a list of available algorithms.

This method is intended for more fine grained control over the registration.
Alternatively, `rigid_registration(source, target; ordered)` is available for
leaving the choice of the algorithm to a rule of thumb.

Both `source` and `target` can be given in a form described in Section
[Representing point clouds](@ref).
They must have matching dimensions (both 2D or both 3D and so forth).
"""
function rigid_registration(source, target, alg)
    error("Unsopported algorithm of type ", typeof(alg))
end
