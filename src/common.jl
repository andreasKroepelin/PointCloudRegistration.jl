function check_sizes(pointclouds...)
    allequal(size, pointclouds) ||
        throw(ArgumentError("point clouds must have same size"))
end

default_config() =
    (; iterations = 10, annealing = 5, restarts = 5, rng = Random.default_rng())

function bbox(xs::VecOfSVec)
    lo = hi = first(xs)
    for point in xs
        lo = min.(point, lo)
        hi = max.(point, hi)
    end
    lo, hi
end

bbox(pc::PointCloud) = bbox(pc.points)

function diameter(xs::VecOfSVec{N}, ε) where {N}
    if iszero(ε)
        return diameter_exhaustive(xs)
    end
    lo, hi = bbox(xs)
    Δ = (ε / 2sqrt(N)) * (hi - lo)
    grid_points = Set(round.((x - lo) ./ Δ) for x in xs) |> collect
    diameter_exhaustive(grid_points)
end

function diameter_exhaustive(xs::VecOfSVec{N, T}) where {N, T}
    max_v = zero(eltype(xs))
    max_d = typemin(T)
    for i in 1:length(xs)
        xi = xs[i]
        for j in (i + 1):length(xs)
            xj = xs[j]
            v = xj - xi
            d = LinearAlgebra.norm_sqr(v)
            if d > max_d
                max_v = v
                max_d = d
            end
        end
    end
    max_v, max_d
end

function min_volume_bbox(xs, ε, ::Val{N} = Val(length(eltype(xs)))) where {N}
    if N == 0
        return SMatrix{length(eltype(xs)), 0, eltype(eltype(xs))}()
    end

    v, d = diameter(xs, ε)
    # v = normalize(v)
    # P = I - v * v'
    # xs_projected = mappedarray(LinearMap(P), xs)
    # hcat(v, min_volume_bbox(xs_projected, ε, Val(N - 1)))

    # v, d = diameter_exhaustive(eachcol(X))
    R = qr(v * ones(typeof(v))').Q
    if det(R) < 0
        R = negative_last_column(R)
    end
    R = R'
    two_to_N = ntuple(i -> i + 1, Val(N - 1)) |> SVector
    PR = R[two_to_N, :]
    # xs_rot = mappedarray(LinearMap(PR), xs)
    xs_rot = map(LinearMap(PR), xs)
    R_lower_dim = min_volume_bbox(xs_rot, ε)
    add_identity_dim(R_lower_dim) * R
end

# makes
#  1 2
#  3 4
# into
#  1 0 0
#  0 1 2
#  0 3 4
function add_identity_dim(A::StaticMatrix{N, N}) where {N}
    z = zero(SVector{N, eltype(A)})
    vcat(hcat(one(eltype(A)), z'), hcat(z, A))
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

wsum(points, weights) = wsum(identity, points, weights)

function wsum(f, points::VecOfSVec, weights)
    s = zero(f(zero(eltype(points))))
    for (point, weight) in zip(points, weights)
        s += weight * f(point)
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
            (mv2sqs(maxvar(target_or_eigen)),)
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
