function rand_rotation(rng, src::StaticVector{N, T}, trg::StaticVector{N, T}) where {N, T}
    M = rand(rng, rotation_type(Val(N), T))
    orthogonal, _ = qr(M)
    if det(orthogonal) < 0
        negative_last_column(orthogonal)
    else
        orthogonal
    end
end

function rand_transformation(rng, src, trg)
    rotation = rand_rotation(rng, src, trg)

    translation = trg - rotation * src

    AffineMap(rotation, translation)::transformation_type(source, target)
end

function simple_transformation(src::StaticVector{N, T}, trg::StaticVector{N, T}) where {N, T}
    rotation = one(rotation_type(Val(N), T))
    translation = trg - rotation * src
    AffineMap(rotation, translation)::transformation_type(source, target)
end


function identity_transformation(::Type{<:AffineMap{R, L}}) where {R, L}
    AffineMap(one(R), zero(L))
end
