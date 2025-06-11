function check_sizes(pointclouds...)
    allequal(size, pointclouds) ||
        throw(ArgumentError("point clouds must have same size"))
end

default_config() = (;
    iterations = 10,
    scale = DefaultAnnealing(),
    restarts = 5,
    rng = Random.default_rng(),
    accumulator = BestTransformation,
    axisalign = true,
)

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

function rotation_type(
    source::PointCloud{N, TS},
    target::PointCloud{N, TT},
) where {N, TS, TT}
    T = promote_type(TS, TT)
    rotation_type(Val(N), T)
end

rotation_type(::Val{N}, ::Type{T}) where {N, T} = SMatrix{N, N, T, N * N}

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

result(bt::BestTransformation, lm::LinearMap) =
    if isone(lm.linear)
        bt.best.transformation
    else
        lm ∘ bt.best.transformation
    end

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

function result(at::AllTransformations, lm::LinearMap)
    if isone(lm.linear)
        at.transformations
    else
        map(at.transformations) do twc
            @set twc.transformation = lm ∘ twc.transformation
        end
    end
end

function avg_nn_dist(pc::PointCloud)
    tree = KDTree(pc.points, Euclidean(); reorder = true)
    # the nearest neighbour is always the point itself, so query for the
    # nearest two
    _, dists2 = knn(tree, pc.points, 2)
    # dists = maximum.(dists2)
    mean(first, dists2)
end

function avg_nn_dist2(pc::PointCloud{N, T}) where {N, T}
    sum_of_dists = zero(T)
    for i in eachindex(pc.points)
        xi = pc.points[i]
        min_sqdist = typemax(T)
        for j in eachindex(pc.points)
            i == j && continue
            xj = pc.points[j]
            d = sqeuclidean(xi, xj)
            min_sqdist = min(d, min_sqdist)
        end
        sum_of_dists += sqrt(min_sqdist)
    end
    sum_of_dists / length(pc.points)
end

"""
    DefaultAnnealing([steps = 5])

Annealing plan starting from the largest standard deviation of the target point
cloud in any direction, going down to average nearest neighbor distance in the
target, in `steps` steps with logarithmic progression.
"""
struct DefaultAnnealing
    steps::Int
end

DefaultAnnealing() = DefaultAnnealing(5)

"""
    DownTo(scale, [steps = 5])

Annealing plan starting from the largest standard deviation of the target point
cloud in any direction, going down to `scale` in `steps` steps with logarithmic
progression.
"""
struct DownTo{T <: Real}
    scale::T
    steps::Int
end

DownTo(scale) = DownTo(scale, 5)

const ScaleType = Union{
    T,
    <: AbstractVector{T},
    DefaultAnnealing,
    DownTo{T},
} where {T <: Real}

annealing_plan(_, scale::Number) = tuple(scale^2)

annealing_plan(_, scales::AbstractVector) = scales .^ 2

function annealing_plan(target, ann::DefaultAnnealing)
    hi = maximum(target.coveigvals)
    lo = avg_nn_dist(target) ^ 2
    logrange(hi, lo; length = ann.steps)
end

function annealing_plan(target, ann::DownTo)
    hi = maximum(target.coveigvals)
    lo = ann.scale ^ 2
    logrange(hi, lo; length = ann.steps)
end
