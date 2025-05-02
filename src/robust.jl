struct GemanMcclure{T}
    sqscale::T
end

function mm_weight(gm::GemanMcclure, x, y)
    sqdist = sqeuclidean(x, y)
    (gm.sqscale / (gm.sqscale + sqdist))^2
end

function cost(gm::GemanMcclure, x, y)
    sqdist = sqeuclidean(x, y)
    gm.sqscale * sqdist / (gm.sqscale + sqdist)
end

function correspondences(; source::PointCloud{N}, target::PointCloud{N}) where N
    mappedarray(
        source.points,
        source.weights,
        target.points,
        target.weights,
    ) do src, w_src, trg, w_trg
        (source = src, target = trg, weight = w_src * w_trg)
    end
end

function evaluate_geman_mcclure(sqscale, source, target, transformation)
    total_cost = zero(common_eltype(source, target))
    gm = GemanMcclure(sqscale)
    for c in correspondences(; source, target)
        total_cost += c.weight * cost(gm, transformation(c.source), c.target)
    end
    TransformationWithCost(total_cost, transformation)
end

function register_robustly(source, target; scale::Real, kwargs...)
    config = (; default_config()..., kwargs...)
    pc_source = PointCloud(source)
    pc_target = PointCloud(target)
    sqscales = annealing_plan(pc_target, scale, config.annealing)
    _register_robustly(
        pc_source,
        pc_target,
        sqscales,
        config.restarts,
        config.iterations,
        config.rng,
    )
end

function _register_robustly(source, target, sqscales, restarts, iterations, rng)
    check_sizes(source, target)

    cs = correspondences(; source, target)
    mm_weights = zeros(common_eltype(source, target), size(source, 2))

    best = worst(transformation_type(source, target))
    init_transformation = simple_transformation(source, target)
    restart = 0
    while true
        transformation = init_transformation
        @logmsg LogLevel(-2000) "mm iteration" restart iter = -1 sqscale =
            -one(eltype(sqscales)) mm_weights = copy(mm_weights) rotation =
            transformation.linear translation = transformation.translation init_transformation _id =
            :mm

        for sqscale in sqscales
            gm = GemanMcclure(sqscale)
            for iter in 1:iterations
                @inbounds for i in eachindex(mm_weights, cs)
                    c = cs[i]
                    mm_weights[i] =
                        c.weight *
                        mm_weight(gm, transformation(c.source), c.target)
                end
                mm_weights ./= sum(mm_weights)

                source_mean = wsum(source.points, mm_weights)
                target_mean = wsum(target.points, mm_weights)
                covariance = zero(rotation_type(source, target))
                @inbounds for i in eachindex(mm_weights)
                    centered_trg = target.points[i] - target_mean
                    centered_src = source.points[i] - source_mean
                    covariance += mm_weights[i] * centered_trg * centered_src'
                end
                transformation = transformation_from_moments(
                    covariance,
                    source_mean,
                    target_mean,
                )
                @logmsg LogLevel(-2000) "mm iteration" restart iter sqscale mm_weights =
                    copy(mm_weights) rotation = transformation.linear translation =
                    transformation.translation init_transformation _id = :mm
            end
        end
        best = better(
            best,
            evaluate_geman_mcclure(
                last(sqscales),
                source,
                target,
                transformation,
            ),
        )
        if restart < restarts
            restart += 1
            init_transformation = rand_transformation(rng, source, target)
        else
            break
        end
    end

    best.transformation
end
