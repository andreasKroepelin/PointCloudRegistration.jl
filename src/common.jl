function check_sizes(pointclouds...)
    allequal(size, pointclouds) ||
        throw(ArgumentError("point clouds must have same size"))
end

default_config() =
    (; iterations = 10, annealing = 5, restarts = 5, rng = Random.default_rng())

function bbox(X)
    lo = hi = first(points(X))
    for point in points(X)
        lo = min.(point, lo)
        hi = max.(point, hi)
    end
    lo, hi
end

function eigen_cov(pc::PointCloud)
    cov_X = wsum(pc.points, pc.weights) do x
        xc = x - pc.mean
        xc * xc'
    end
    cov_X /= pc.sum_of_weights

    eigen(Symmetric(cov_X))
end

maxvar(X::PointCloud) = maxvar(eigen_cov(X))
maxvar(eig::Eigen) = maximum(eig.values)

nrows(Xs::AbstractMatrix...) = nrows(Size.(Xs)...)
nrows(::Size{Sz}) where {Sz} = first(Sz)::Int
function nrows(::Size{Sz}, szs::Size...) where {Sz}
    N = nrows(szs...)
    if first(Sz) == N
        N
    else
        throw(ArgumentError("nrows expects matrices of equal number of rows"))
    end
end

common_eltype(Xs...) = promote_type(eltype.(Xs)...)

statically_known_rows(X::AbstractMatrix) = statically_known_rows(Size(X), X)
function statically_known_rows(::Size{Sz}, X) where {Sz}
    N, M = Sz
    if N isa Int
        X
    else
        HybridMatrix{size(X, 1), M}(X)
    end
end

function rotation_type(source::AbstractMatrix, target::AbstractMatrix)
    N = nrows(source, target)
    T = common_eltype(source, target)
    rotation_type(Val(N), T)
end

rotation_type(::Val{N}, ::Type{T}) where {N, T} = SMatrix{N, N, T, N * N}

function translation_type(source::AbstractMatrix, target::AbstractMatrix)
    N = nrows(source, target)
    T = common_eltype(source, target)
    translation_type(Val(N), T)
end

translation_type(::Val{N}, ::Type{T}) where {N, T} = SVector{N, T}

function transformation_type(a, b)
    AffineMap{rotation_type(a, b), translation_type(a, b)}
end

wsum(hmatrix, weights) = wsum(identity, hmatrix, weights)

function wsum(f, hmatrix::HybridMatrix{N, M, T}, weights) where {N, M, T}
    s = zero(f(zero(SVector{N, T})))
    @inbounds for i in eachindex(eachcol(hmatrix), weights)
        s += weights[i] * f(hmatrix[:, i]::SVector{N, T})
    end
    s
end

struct TransformationWithCost{T <: Real, A <: AffineMap}
    cost::T
    transformation::A
end

function worst(
    A::Type{<:AffineMap{<:AbstractMatrix{T}, <:AbstractVector{T}}},
) where {T}
    TransformationWithCost(typemax(T), identity_transformation(A))
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

function annealing_plan(target_or_eigen, scale, n)
    mv2sqs(mv) = mv / 100
    if n < 2
        if ismissing(scale)
            (mv2sqs(maxvar(target_or_eigen)), )
        else
            (scale^2,)
        end
    else
        mv = maxvar(target_or_eigen)
        if ismissing(scale)
            logrange(mv, mv2sqs(mv); length = n)
        else
            logrange(mv, scale^2; length = n)
        end
    end
end
