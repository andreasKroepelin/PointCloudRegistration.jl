function _static_mean(X, weights)
    m = zero(X[:, 1])
    @inbounds for i in eachindex(weights)
        m += weights[i] * X[:, i]
    end
    m
end

function geman_mcclure_weight(scale, x, y)
    sq_dist = sqeuclidean(x, y)
    (scale / (scale + sq_dist))^2
end

# function opt_geman_mcclure(Y, X, scales, init_transformation)
#     if equal_num_rows(X, Y) isa Val
#         _static_opt_geman_mcclure(Y, X, scales, init_transformation)
#     else
#         _dynamic_opt_geman_mcclure(Y, X, scales, init_transformation)
#     end
# end

struct RobustProblem{TX, TY, S, TT}
    X::TX
    Y::TY
    scales::S
    init_transformation::TT
end

function optimize_transformation(problem::RobustProblem, buf)
    (; X, Y, scales, init_transformation) = problem
    rotation = init_transformation.linear
    translation = init_transformation.translation
    @no_escape buf begin
        weights = @alloc(promote_type(eltype(X), eltype(Y)), size(X, 2))

        for scale in scales
            for i in eachindex(weights)
                x = X[:, i]
                y = Y[:, i]
                weights[i] =
                    geman_mcclure_weight(scale, x, rotation * y + translation)
            end
            weights ./= sum(weights)

            mean_x = _static_mean(X, weights)
            mean_y = _static_mean(Y, weights)
            covariance = zero(rotation)
            for (x, y, weight) in zip(eachcol(X), eachcol(Y), weights)
                centered_x = x - mean_x
                centered_y = y - mean_y
                covariance += weight * centered_x * centered_y'
            end
            rotation = static_rot_from_cov(covariance)
            translation = mean_x - rotation * mean_y
        end
    end

    AffineMap(rotation, translation)
end

#=
function optimize_transformation(problem::DensitiesProblem, buf)
    (; X, Y, scales, init_transformation) = problem
    rotation = init_transformation.linear
    translation = init_transformation.translation
    @no_escape buf begin
        weights = @alloc(promote_type(eltype(X), eltype(Y)), size(X, 2), size(Y, 2))

        for scale in scales
            for i in eachindex(weights)
                x = X[:, i]
                y = Y[:, i]
                weights[i] =
                    geman_mcclure_weight(scale, x, rotation * y + translation)
            end
            weights ./= sum(weights)

            mean_x = _static_mean(X, weights)
            mean_y = _static_mean(Y, weights)
            covariance = zero(rotation)
            for (x, y, weight) in zip(eachcol(X), eachcol(Y), weights)
                centered_x = x - mean_x
                centered_y = y - mean_y
                covariance += weight * centered_x * centered_y'
            end
            rotation = static_rot_from_cov(covariance)
            translation = mean_x - rotation * mean_y
        end
    end

    AffineMap(rotation, translation)
end
=#

function evaluate_geman_mcclure(scale, Y, X, transformation)
    cost = zero(CommonType(X, Y))
    for i in axes(X, 2)
        x = X[:, i]
        y = Y[:, i]
        sqdist = sqeuclidean(x, transformation(y))
        cost += sqdist / (scale + sqdist)
    end
    cost *= scale
    cost
end

# function optimize_transformation(problem::RobustProblem{:dynamic}, buf)
#     (; X, Y, scales, transformation, weights) = sdp
#     rotation = init_transformation.linear
#     translation = init_transformation.translation
#     @no_escape buf begin
#         weights = @alloc(promote_type(eltype(X), eltype(Y)), size(X, 2))
#         transformed_Y = @alloc(eltype(Y), size(Y)...)
#         mean_x = @alloc(eltype(X), size(X, 1))
#         mean_y = @alloc(eltype(Y), size(Y, 1))
#         centered_x = @alloc(eltype(X), size(X, 1))
#         centered_y = @alloc(eltype(Y), size(Y, 1))
#         covariance = @alloc(eltype(rotation), size(rotation)...)

#         for scale in scales
#             mul!(transformed_Y, rotation, Y)
#             transformed_Y .+= translation
#             for i in eachindex(weights)
#                 x = @view X[:, i]
#                 transformed_y = @view transformed_Y[:, i]
#                 weights[i] = geman_mcclure_weight(scale, x, transformed_y)
#             end
#             weights ./= sum(weights)

#             mul!(mean_x, X, weights)
#             mul!(mean_y, Y, weights)
#             fill!(covariance, zero(eltype(covariance)))
#             for (x, y, weight) in zip(eachcol(X), eachcol(Y), weights)
#                 centered_x .= x .- mean_x
#                 centered_y .= y .- mean_y
#                 mul!(covariance, centered_x, centered_y', weight, true)
#             end
#             dynamic_rot_from_cov!(rotation, covariance)
#             mul!(centered_y, rotation, mean_y)
#             translation .= mean_x .- centered_y
#         end
#     end

#     AffineMap(rotation, translation)
# end

function bbox(X::AbstractMatrix{T}) where {T}
    lo = SVector(ntuple(_ -> typemax(T), Val(NRows(X))))
    hi = SVector(ntuple(_ -> typemin(T), Val(NRows(X))))
    for i in axes(X, 2)
        x = X[:, i]
        lo = min.(lo, x)
        hi = max.(hi, x)
    end
    (lo, hi)
end

function p2p(X)
    lo, hi = bbox(X)
    maximum(hi .- lo)
end

guess_diameter(X, Y) = NRows(X) * max(p2p(X), p2p(Y))^2

statically_known_rows(X::AbstractMatrix) = statically_known_rows(Size(X), X)
function statically_known_rows(::Size{Sz}, X) where {Sz}
    N, M = Sz
    if N isa Int
        X
    else
        HybridMatrix{size(X, 1), M}(X)
    end
end

NRows(X::AbstractMatrix) = NRows(Size(X))
NRows(::Size{Sz}) where {Sz} = first(Sz)::Int

CommonType(X::AbstractMatrix...) = promote_type(eltype.(X)...)

function register_robustly(
    Y::AbstractMatrix,
    X::AbstractMatrix;
    annealing = 100,
    initialization = IdentityInitialization(),
    minscale,
)
    @assert size(X) == size(Y)

    _register_robustly(
        statically_known_rows(Y),
        statically_known_rows(X),
        annealing,
        initialization,
        minscale,
    )
end

function _register_robustly(Y, X, annealing, initialization, minscale)
    scales = logrange(guess_diameter(X, Y), minscale^2; length = annealing)
    buf = SlabBuffer()
    best_cost = typemax(CommonType(X, Y))
    best_transformation = identity_transformation(X, Y)
    for init_transformation in transformation_iterator(initialization, X, Y)
        problem = RobustProblem(X, Y, scales, init_transformation)
        transformation = optimize_transformation(problem, buf)
        cost = evaluate_geman_mcclure(last(scales), Y, X, transformation)
        if cost < best_cost
            best_cost = cost
            best_transformation = transformation
        end
    end

    best_transformation
end
