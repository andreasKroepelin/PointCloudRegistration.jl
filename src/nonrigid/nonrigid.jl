"""
    nonrigid_registration(source, target, algorithm)

Find a non-rigid transformation that transforms the point cloud `source` to
"match" the point cloud `target` using `algorithm`.
See [here](#Non-rigid-registration-algorithms) for a list of available
algorithms.

Both `source` and `target` can be given in a form described in Section
[Representing point clouds](@ref).
They must have matching dimensions (both 2D or both 3D and so forth).
"""
function nonrigid_registration(source, target, algorithm)
    error("Unsopported algorithm of type ", typeof(algorithm))
end

"""
    Displacement(origin, result)

A displacement that displaces every vector in `origin` to the corresponding
vector in `result`.

This type is public but not exported.

    (::Displacement)(pointcloud::PointCloud)

Apply the displacement to `pointcloud`, returning a new `PointCloud`.
This only works if the displacement originates exactly at `pointcloud.points`.
"""
struct Displacement{N, O <: VecOfSVec{N}, R <: VecOfSVec{N}}
    origin::O
    result::R

    function Displacement(
        origin::O,
        result::R,
    ) where {N, O <: VecOfSVec{N}, R <: VecOfSVec{N}}
        @argcheck length(origin) == length(result)
        return new{N, O, R}(origin, result)
    end
end

dimension(::Displacement{N}) where {N} = N

function (displacement::Displacement{N})(pc::PointCloud{N}) where {N}
    @argcheck displacement.origin == points(pc) "Displacement can only be applied to the source it was computed for."
    return PointCloud(displacement.result, weights(pc))
end

function Base.show(
    io::IO,
    ::MIME"text/plain",
    displacement::Displacement{N},
) where {N}
    mat_io = IOContext(io, :limit => true, :compact => true)
    vectors = displacement.result .- displacement.origin
    println(
        io,
        N,
        "-dimensional displacement with ",
        length(vectors),
        " vectors of eltype ",
        eltype(eltype(vectors)),
    )
    Base.print_matrix(mat_io, to_matrix(vectors))
end

"""
    EarthMover(; optimizer)

Non-rigid registration of point clouds by solving an _Optimal Transport_
problem (minimizing the so-called _Earth Mover Distance_).

Using this algorithm requires the `ExactOptimalTransport.jl` package to
be loaded.
Additionally, the keyword argument `optimizer` must be set to an optimizer
supporting the MathOptInterface.
See [`ExactOptimalTransport.emd`](@extref) for details.


For two weighted point clouds
``x_1, \\dots, x_I \\in \\mathbb{R}^D`` with normalized weights
``p_1, \\dots, p_I`` and
``y_1, \\dots, y_J \\in \\mathbb{R}^D`` with normalized weights
``q_1, \\dots, q_J``,
the method first finds an optimal transport plan
``\\gamma \\in \\mathbb{R}^{J \\times I}`` with
``\\sum_{i = 1}^I \\gamma_{i j} = q_j`` and
``\\sum_{i = j}^J \\gamma_{i j} = p_i``
that minimizes
``\\sum_{i j} \\gamma_{i j} \\Vert x_i - y_j \\Vert``
and then computes the registered source points as
``\\hat{y}_j = \\frac{1}{q_j} \\sum_i \\gamma_{i j} x_i``.
"""
@kwdef struct EarthMover{O}
    optimizer::O
end

"""
    DivergenceFree(; [scale, degree, iterations])

Non-rigid registration of point clouds that ensures that the displacement field
is divergence free.
The method is presented in _"Divergence-Free Shape Correspondence by
Deformation"_ by Eisenberger, Lähner, and Cremers
(<https://doi.org/10.1111/cgf.13785>).

Using this algorithm requires the `Mooncake.jl` package to be loaded.

The algorithm first shifts and scales the point clouds such that they are
contained in the unit (hyper-) cube.
Then, it estimates a displacement field as a linear combination of increasingly
finegrained "swirls" that each have zero divergence across this unit cube.
This is done using gradient descent (ADAM).

# Parameters
- `scale`: The residual standard deviation for the matching score.
  Default: average nearest neighbor distance of target.
- `degree`: The maximum number of "swirls" per direction.
  Default: `3`.
- `iterations`: How many optimizing iterations to perform.
  Default: `100`.
"""
@kwdef struct DivergenceFree{S}
    scale::S = nothing
    degree::Int = 3
    iterations::Int = 100
end

struct DivFreeDisplacement{S, V}
    ubs::S
    invubs::S
    velocity::V

    function DivFreeDisplacement(ubs, velocity)
        new{typeof(ubs), typeof(velocity)}(ubs, inv(ubs), velocity)
    end
end
