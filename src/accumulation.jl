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

abstract type AbstractTransformationAccumulator end

"""
Stores the best transformation found.

Makes `register_gmc` and `register_kc` return a single transformation.
"""
struct BestTransformation{TWC <: TransformationWithCost} <:
       AbstractTransformationAccumulator
    best::TWC
end

function BestTransformation(
    A::Type{<:AffineMap{<:AbstractMatrix{T}, <:AbstractVector{T}}},
) where {T}
    BestTransformation(
        TransformationWithCost(typemax(T), identity_transformation(A)),
    )
end

function update(bt::BestTransformation{TWC}, t::TWC) where {TWC}
    BestTransformation(better(bt.best, t))
end

result(bt::BestTransformation) = bt.best
function result(bt::BestTransformation, lm::LinearMap)
    @reset bt.best.transformation = lm ∘ bt.best.transformation
    bt.best
end

"""
Stores all transformations found.

Makes `register_gmc` and `register_kc` return a `Vector` of transformations.
"""
struct AllTransformations{TWC <: TransformationWithCost} <:
       AbstractTransformationAccumulator
    transformations::Vector{TWC}
end

function AllTransformations(
    A::Type{<:AffineMap{<:AbstractMatrix{T}, <:AbstractVector{T}}},
) where {T}
    AllTransformations(TransformationWithCost{T, A}[])
end

function update(at::AllTransformations{TWC}, t::TWC) where {TWC}
    push!(at.transformations, t)
    at
end

result(at::AllTransformations) = at.transformations
function result(at::AllTransformations, lm::LinearMap)
    if isone(lm.linear)
        at.transformations
    else
        map(at.transformations) do twc
            @set twc.transformation = lm ∘ twc.transformation
        end
    end
end
