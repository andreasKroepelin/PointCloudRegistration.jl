function orthogonal_type(
    source::PointCloud{N, TS},
    target::PointCloud{N, TT},
    flip::FlipMarker,
) where {N, TS, TT}
    T = promote_type(TS, TT)
    T1 = typeof(one(T))
    orthogonal_type(Val(N), T1, flip)
end

orthogonal_type(::Val{N}, ::Type{T}, ::NoFlip) where {N, T} =
    RotMatrix{N, T, N * N}
orthogonal_type(::Val{N}, ::Type{T}, ::WithFlip) where {N, T} =
    OrthogonalMatrix{N, T, N * N}

function flip_matrix(::Val{N}, ::Type{T}) where {N, T}
    o = @SVector ones(T, N - 1)
    negone = oftype(one(T), -1) # inspired by Base.sign
    return Diagonal(push(o, negone))
end

function translation_type(
    source::PointCloud{N, TS},
    target::PointCloud{N, TT},
) where {N, TS, TT}
    T = promote_type(TS, TT)
    translation_type(Val(N), T)
end

translation_type(::Val{N}, ::Type{T}) where {N, T} = SVector{N, T}

function transformation_type(a, b, flip)
    AffineMap{orthogonal_type(a, b, flip), translation_type(a, b)}
end

function rand_orthogonal(
    rng,
    source::PointCloud{N},
    target::PointCloud{N},
    flip::FlipMarker,
) where {N}
    rand_orthogonal(rng, orthogonal_type(source, target, flip))
end

rand_orthogonal(rng, R::Type{<: RotMatrix}) = rand(rng, R)

function rand_orthogonal(rng, O::Type{<: OrthogonalMatrix{N, T}}) where {N, T}
    m = rand(rng, SMatrix{N, N, T})
    return nearest_orthogonal(m, WithFlip())
end

function rand_rotation(rng, SM::Type{<: SMatrix{N, N, T}}) where {N, T}
    rand(rng, RotMatrix{N, T})
end

function rand_rotation(rng, ::Val{N}, ::Type{T}) where {N, T}
    rand(rng, RotMatrix{N, T})
end

function rand_transformation(
    rng::AbstractRNG,
    source::PointCloud{N},
    target::PointCloud{N},
    flip::FlipMarker = NoFlip(),
) where {N}
    orthogonal = rand_orthogonal(rng, source, target, flip)

    i = rand(rng, eachindex(source.points))
    j = rand(rng, eachindex(target.points))
    translation = target.points[j] - orthogonal * source.points[i]

    AffineMap(
        orthogonal,
        translation,
    )::transformation_type(source, target, flip)
end

function rand_transformation(
    source::PointCloud{N},
    target::PointCloud{N},
    flip::FlipMarker = NoFlip(),
) where {N}
    rand_transformation(Random.default_rng(), source, target, flip)
end

rand_transformation(pc::PointCloud, flip::FlipMarker = NoFlip()) =
    rand_transformation(Random.default_rng(), pc, pc, flip)

function simple_transformation(
    source::PointCloud{N},
    target::PointCloud{N},
    flip::FlipMarker = NoFlip(),
) where {N}
    orthogonal = one(orthogonal_type(source, target, flip))
    target_mean, _ = mean_cov(target)
    source_mean, _ = mean_cov(source)
    translation = target_mean - orthogonal * source_mean
    AffineMap(
        orthogonal,
        translation,
    )::transformation_type(source, target, flip)
end

function identity_transformation(::Type{<:AffineMap{R, L}}) where {R, L}
    AffineMap(one(R), zero(L))
end
identity_transformation(::A) where {A <: AffineMap} = identity_transformation(A)
function identity_transformation(
    source::PointCloud{N},
    target::PointCloud{N},
    flip::FlipMarker = NoFlip(),
) where {N}
    AffineMap(
        one(orthogonal_type(source, target, flip)),
        zero(eltype(target.points)) - zero(eltype(source.points)),
    )
end

const RigidTransformation{N} =
    AffineMap{Rot, Transl} where {Rot <: Rotation{N}, Transl <: StaticVector{N}}
