# copied and adapted from the definition of `RotMatrix` in Rotations.jl

struct OrthogonalMatrix{N, T, L} <: StaticMatrix{N, N, T}
    mat::SMatrix{N, N, T, L}
    OrthogonalMatrix{N, T, L}(x::AbstractArray) where {N, T, L} = new{N, T, L}(convert(SMatrix{N, N, T, L}, x))
    # fixes #49 ambiguity introduced in StaticArrays 0.6.5
    OrthogonalMatrix{N, T, L}(x::StaticArray) where {N, T, L} = new{N, T, L}(convert(SMatrix{N, N, T, L}, x))
    OrthogonalMatrix{N}(tpl::NTuple{L, T}) where {N, T, L} = new{N, T, L}(SMatrix{N, N, T, L}(tpl))
end

OrthogonalMatrix(x::SMatrix{N, N, T, L}) where {N, T, L} = OrthogonalMatrix{N, T, L}(x)

Base.@propagate_inbounds Base.getindex(om::OrthogonalMatrix, i::Int) = getindex(om.mat, i)
@inline Base.Tuple(om::OrthogonalMatrix) = Tuple(om.mat)

Base.one(::Type{<: OrthogonalMatrix{N, T, L}}) where {N, T, L} = OrthogonalMatrix(one(SMatrix{N, N, T, L}))
Base.zero(::Type{<: OrthogonalMatrix}) = error("There is no zero orthogonal matrix.")
Base.one(::OM) where {OM <: OrthogonalMatrix} = one(OM)
Base.zero(::OM) where {OM <: OrthogonalMatrix} = zero(OM)
Base.inv(om::OrthogonalMatrix) = OrthogonalMatrix(om.mat')

Base.:*(o1::OrthogonalMatrix, o2::OrthogonalMatrix) = OrthogonalMatrix(o1.mat * o2.mat)
Base.:*(o::OrthogonalMatrix, r::RotMatrix) = OrthogonalMatrix(o.mat * r.mat)
Base.:*(r::RotMatrix, o::OrthogonalMatrix) = OrthogonalMatrix(r.mat * o.mat)

isflip(om::OrthogonalMatrix{N, T}) where {N, T} = det(om) < zero(T)
isflip(::RotMatrix) = false

function nearest_orthogonal(M::StaticMatrix{N, N}, flip::Flip) where {N, Flip <: FlipMarker}
    u, _, v = svd(M)
    if flip isa NoFlip
        s = det(u * v')
        R = if s < zero(s)
            u * flip_matrix(Val(N), eltype(M)) * v'
        else
            u * v'
        end
        return RotMatrix{N}(R)
    else
        # we know that `flip::WithFlip` here, so negative determinant is okay
        return OrthogonalMatrix(u * v')
    end
end
