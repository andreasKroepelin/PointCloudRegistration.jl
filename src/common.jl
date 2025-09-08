function check_sizes(pointclouds...)
    allequal(size, pointclouds) ||
        throw(ArgumentError("point clouds must have same size"))
end

default_config() = (;
    iterations = 50,
    scale = TargetScales(),
    restarts = 5,
    rng = Random.default_rng(),
    accumulator = BestTransformation,
    axisalign = true,
    features = 100,
)

wsum(points, weights) = wsum(identity, points, weights)

function wsum(f, points::VecOfSVec, weights)
    s = zero(f(zero(eltype(points))))
    for (point, weight) in zip(points, weights)
        s += weight * f(point)
    end
    s
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
