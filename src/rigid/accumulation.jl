struct TransformationWithCost{T <: Number, A <: AffineMap}
    cost::T
    transformation::A
end

function better(
    t1::TransformationWithCost{T, A},
    t2::TransformationWithCost{T, A},
) where {T, A}
    if t1.cost < t2.cost
        t1
    else
        t2
    end
end

function worst(C::Type{<: Number}, A::Type{<:AffineMap})
    TransformationWithCost(typemax(C), identity_transformation(A))
end
