function rand_rotation(rng, ::PointCloud{N, T}, ::PointCloud{N, T}) where {N, T}
    rand_rotation(rng, Val(N), T)
end

function rand_rotation(rng, ::Val{N}, ::Type{T}) where {N, T}
    M = randn(rng, rotation_type(Val(N), T))
    orthogonal, _ = qr(M)
    if det(orthogonal) < 0
        negative_last_column(orthogonal)
    else
        orthogonal
    end
end

function rand_transformation(
    rng,
    source::PointCloud{N},
    target::PointCloud{N},
) where {N}
    rotation = rand_rotation(rng, source, target)

    translation = target.mean - rotation * source.mean

    AffineMap(rotation, translation)::transformation_type(source, target)
end

function simple_transformation(
    source::PointCloud{N, T},
    target::PointCloud{N, T},
) where {N, T}
    rotation = one(rotation_type(Val(N), T))
    translation = target.mean - rotation * source.mean
    AffineMap(rotation, translation)::transformation_type(source, target)
end

function identity_transformation(::Type{<:AffineMap{R, L}}) where {R, L}
    AffineMap(one(R), zero(L))
end
