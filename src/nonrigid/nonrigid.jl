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

struct Displacement{N, O <: VecOfSVec{N}, R <: VecOfSVec{N}}
    origin::O
    result::R

    function Displacement(origin::O, result::R) where {N, O <: VecOfSVec{N}, R <: VecOfSVec{N}}
        @argcheck length(origin) == length(result)
        return new{N, O, R}(origin, result)
    end
end

dimension(::Displacement{N}) where {N} = N

function (displacement::Displacement{N})(pc::PointCloud{N}) where {N}
    @argcheck displacement.origin == pc.points "Displacement can only be applied to the source it was computed for."
    return PointCloud(displacement.result, pc.weights)
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

function nonrigid_sinkhorn end
function nonrigid_divfree end
