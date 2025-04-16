function rand_rotation(rng, source, target)
    M = rand(rng, rotation_type(source, target))
    rotation, _ = qr(M)
    if det(rotation) < 0
        rotation = negative_last_column(rotation)
    end
    rotation
end

function rand_transformation(rng, source, target)
    rotation = rand_rotation(rng, source, target)

    trg = target[:, rand(rng, axes(target, 2))]
    src = source[:, rand(rng, axes(source, 2))]
    translation = trg - rotation * src

    AffineMap(rotation, translation)::transformation_type(source, target)
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

struct SimpleInitialization end

struct RandomTransformationIterator{TT, TS, Rng}
    number::Int
    source::TS
    target::TT
    rng::Rng
end

function Base.iterate(rti::RandomTransformationIterator, i = 1)
    i > rti.number && return nothing

    (rand_transformation(rti.rng, rti.source, rti.target), i + 1)
end

function simple_transformation(source, target)
    rotation = one(rotation_type(source, target))
    translation = target[:, rand(axes(target, 2))] - rotation * source[:, rand(axes(source, 2))]
    AffineMap(rotation, translation)::transformation_type(source, target)
end

function transformation_iterator(::SimpleInitialization, source, target)
    (simple_transformation(source, target),)
end

function transformation_iterator(rr::RandomRestarts, source, target)
    RandomTransformationIterator(rr.number, source, target, rr.rng)
end

function identity_transformation(::Type{<:AffineMap{R, L}}) where {R, L}
    AffineMap(one(R), zero(L))
end
