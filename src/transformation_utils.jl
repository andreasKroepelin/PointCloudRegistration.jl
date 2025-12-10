function rotation_type(
    source::PointCloud{N, TS},
    target::PointCloud{N, TT},
) where {N, TS, TT}
    T = promote_type(TS, TT)
    T1 = typeof(one(T))
    rotation_type(Val(N), T1)
end

rotation_type(::Val{N}, ::Type{T}) where {N, T} = RotMatrix{N, T, N * N}

function translation_type(
    source::PointCloud{N, TS},
    target::PointCloud{N, TT},
) where {N, TS, TT}
    T = promote_type(TS, TT)
    translation_type(Val(N), T)
end

translation_type(::Val{N}, ::Type{T}) where {N, T} = SVector{N, T}

function transformation_type(a, b)
    AffineMap{rotation_type(a, b), translation_type(a, b)}
end

function rand_rotation(
    rng,
    source::PointCloud{N},
    target::PointCloud{N},
) where {N}
    rand(rng, rotation_type(source, target))
end

function rand_rotation(rng, SM::Type{<: SMatrix{N, N, T}}) where {N, T}
    rand(rng, RotMatrix{N, T})
end

function rand_rotation(rng, ::Val{N}, ::Type{T}) where {N, T}
    rand(rng, RotMatrix{N, T})
end

function rand_transformation(
    rng,
    source::PointCloud{N},
    target::PointCloud{N},
) where {N}
    rotation = rand_rotation(rng, source, target)

    i = rand(rng, eachindex(source.points))
    j = rand(rng, eachindex(target.points))
    translation = target.points[j] - rotation * source.points[i]

    AffineMap(rotation, translation)::transformation_type(source, target)
end

function simple_transformation(
    source::PointCloud{N},
    target::PointCloud{N},
) where {N}
    rotation = one(rotation_type(source, target))
    target_mean, _ = mean_cov(target)
    source_mean, _ = mean_cov(source)
    translation = target_mean - rotation * source_mean
    AffineMap(rotation, translation)::transformation_type(source, target)
end

function identity_transformation(::Type{<:AffineMap{R, L}}) where {R, L}
    AffineMap(one(R), zero(L))
end
identity_transformation(::A) where {A <: AffineMap} = identity_transformation(A)
function identity_transformation(
    source::PointCloud{N},
    target::PointCloud{N},
) where {N}
    AffineMap(one(rotation_type(source, target)), zero(eltype(target.points)) - zero(eltype(source.points)))
end

const RigidTransformation{N} = AffineMap{Rot, Transl} where {Rot <: Rotation{N}, Transl <: StaticVector{N}}
