struct TransformationSampler{N, T, XT, YT}
    X::XT
    Y::YT
    function TransformationSampler(X::AbstractMatrix, Y::AbstractMatrix)
        N = NRows(X)
        T = promote_type(eltype(X), eltype(Y))

        new{N, T, typeof(X), typeof(Y)}(X, Y)
    end
end

function rand_transformation(rng, Y, X)
    N = NRows(X)
    T = promote_type(eltype(X), eltype(Y))
    M = @SMatrix rand(rng, T, N, N)
    rotation, _ = qr(M)
    if det(rotation) < 0
        rotation = negative_last_column(rotation)
    end

    x = X[:, rand(rng, axes(X, 2))]
    y = Y[:, rand(rng, axes(Y, 2))]
    translation = x - rotation * y

    AffineMap(rotation, translation)
end

# function rand_transformation(dti::DynamicTransformationIter)
#     (; X, Y, matrix, translation) = dti
#     rand!(matrix)
#     rotation, _ = qr!(matrix)
#     rotation_m = Matrix(rotation)
#     if det(rotation) < 0
#         @view(rotation_m[:, end]) .*= -1
#     end

#     x = view(X, :, rand(axes(X, 2)))
#     y = view(Y, :, rand(axes(Y, 2)))
#     mul!(translation, rotation_m, y)
#     translation .*= -1
#     translation .+ x

#     AffineMap(rotation_m, translation)
# end

struct RandomRestarts{Rng}
    number::Int
    rng::Rng
end

RandomRestarts(number::Int) = RandomRestarts(number, Random.default_rng())

struct IdentityInitialization end

struct RandomTransformationIterator{TX, TY, Rng}
    number::Int
    X::TX
    Y::TY
    rng::Rng
end

function Base.iterate(rti::RandomTransformationIterator, i = 1)
    i > rti.number && return nothing

    (rand_transformation(rti.rng, rti.Y, rti.X), i + 1)
end

function identity_transformation(X, Y)
    N = NRows(X)
    T = promote_type(eltype(X), eltype(Y))
    AffineMap(one(SMatrix{N, N, T}), zero(SVector{N, T}))
end

function transformation_iterator(::IdentityInitialization, X, Y)
    (identity_transformation(X, Y),)
end

function transformation_iterator(rr::RandomRestarts, X, Y)
    RandomTransformationIterator(rr.number, X, Y, rr.rng)
end
