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

function correspondences(; wpcs::WeightedPointCloud...)
    
end

function optimize_transformation(problem::RobustProblem, buf)
    (; X, Y, scales, init_transformation) = problem
    transformation = init_transformation
    @no_escape buf begin
        weights = @alloc(promote_type(eltype(X), eltype(Y)), size(X, 2))

        for scale in scales
            for (i, src, trg) in
                zip(eachindex(weights), points(source), points(target))
                weights[i] =
                    geman_mcclure_weight(scale, transformation(src), trg)
            end
            weights ./= sum(weights)

            source_mean = _static_mean(source, weights)
            target_mean = _static_mean(target, weights)
            covariance = zero(rotation)
            for (src, trg, weight) in
                zip(eachcol(source), eachcol(target), weights)
                centered_x = x - mean_x
                centered_y = y - mean_y
                covariance += weight * centered_x * centered_y'
            end
            transformation = transformation_from_moments(
                covariance,
                source_mean,
                target_mean,
            )
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

function evaluate_geman_mcclure(sqscale, source, target, transformation)
    cost = zero(common_elype(source, target))
    for i in axes(source, 2)
        src = source[:, i]
        trg = target[:, i]
        sqdist = sqeuclidean(trg, transformation(src))
        cost += sqdist / (sqscale + sqdist)
    end
    cost *= sqscale
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

function register_robustly(
    source,
    target;
    scale::Real,
    annealing = nothing,
    initialization = SimpleInitialization(),
)
    _register_robustly(
        weighted(statically_known_rows(source)),
        weighted(statically_known_rows(target)),
        scale,
        annealing,
        initialization,
    )
end

function _register_robustly(source, target, scale, annealing, initialization)
    check_sizes(source, target)
    sq_scales = if annealing isa Nothing
        (scale^2,)
    elseif annealing isa Int
        logrange(guess_diameter(X, Y), scale^2; length = annealing)
    else
        throw(ArgumentError("annealing must either be integer or nothing"))
    end

    best = worst(transformation_type(source, target))
    for init_transformation in transformation_iterator(initialization, X, Y)
        problem = RobustProblem(X, Y, scales, init_transformation)
        transformation = optimize_transformation(problem, buf)
        best = better(
            best,
            TransformationWithCost(
                evaluate_geman_mcclure(scale, source, target, transformation),
                transformation,
            ),
        )
    end

    best
end
