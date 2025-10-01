struct TransformationWithCost{T <: Real, A <: AffineMap}
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

function worst(
    A::Type{<:AffineMap{<:AbstractMatrix{TL}, <:AbstractVector{TT}}},
) where {TL, TT}
    TransformationWithCost(typemax(TL), identity_transformation(A))
end
