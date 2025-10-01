default_iterations() = 50
default_restarts() = RandomRestarts(5)
default_scale() = TargetScales()

@inline no_report(; kwargs...) = nothing

wsum(points, weights) = wsum(identity, points, weights)

function wsum(f, points::VecOfSVec, weights)
    s = zero(f(zero(eltype(points))))
    for (point, weight) in zip(points, weights)
        s += weight * f(point)
    end
    s
end

function zero_cov(source::PointCloud{N}, target::PointCloud{N}) where {N}
    a = zero(eltype(target.points))
    b = zero(eltype(source.points))
    zero(a * b')
end

const REGISTER_DOCS_START = """
Find a [`CoordinateTransformations.AffineMap`](@extref) that rotates and
translates `source` in a way that """

const REGISTER_DOCS_SYMBOLS = """
for source points ``y_i``, target points ``x_i``, rotation ``R``, and
translation ``t``. """

const REGISTER_DOCS_EQUAL = """
This assumes that the ``i``-th point in `source` corresponds to the ``i``-th
point in `target`, so `source` and `target` must have the same size.
"""

const REGISTER_DOCS_TYPES = """
`source` and `target` can each either be matrices with one point per column
or [`PointCloud`](@ref)s.
"""
