module KernelCorrelationSprings

using PointCloudRegistration
using Mooncake
using Distances
using NearestNeighbors
using StaticArrays

using ..Adam

struct KcSpringsObjective{N, PS <: PointCloud{N}, PT <: PointCloud{N}, SP, SD, T1, T2}
    source::PS
    target::PT
    spring_pairs::SP
    spring_dists::SD
    sqsigma::T1
    stiffness::T2
end

function (kcs::KcSpringsObjective{N})(params) where {N}
    (; source, target, spring_pairs, spring_dists, sqsigma, stiffness) = kcs
    displaced_source_points = interpret_params(params, Val(N))

    zero_dsrc = zero(eltype(displaced_source_points))
    zero_trg = zero(eltype(target.points))
    kc = sqeuclidean(zero_dsrc, zero_trg) / sqsigma

    totalw = zero(source.sum_of_weights) * zero(target.sum_of_weights)
    for (dsrc, src_w) in zip(displaced_source_points, source.weights)
        for (trg, trg_w) in zip(target.points, target.weights)
            w = src_w * trg_w
            totalw += w
            kc += w * exp(sqeuclidean(dsrc, trg) / (-2sqsigma))
        end
    end
    kc /= totalw

    spring_energy = zero(kc)

    for (original_dist, (j1, j2)) in zip(spring_dists, spring_pairs)
        dsrc1 = displaced_source_points[j1]
        dsrc2 = displaced_source_points[j2]
        dist = euclidean(dsrc1, dsrc2)
        # equivalent to ((original_dist - dist) / original_dist)^2
        # dist_deviation += (1 - dist / original_dist)^2
        spring_energy += (dist - original_dist)^2
    end
    spring_energy /= length(spring_dists)

    return -kc + stiffness * spring_energy 
end

@inline function interpret_params(params::Vector{T}, ::Val{N}) where {T, N}
    reinterpret(SVector{N, T}, params)
end

struct KcSpringsRegistration{D}
    displacements::D
end

function PointCloudRegistration.displacements(ksr::KcSpringsRegistration)
    return ksr.displacements
end

function PointCloudRegistration.register_kc_springs(
    source,
    target;
    scale = nothing,
    stiffness = 1,
    max_spring_length = nothing,
    iterations = 100_000,
    report_iteration::RI = PointCloudRegistration.no_report
) where {RI}
    source_pc = PointCloud(source)
    target_pc = PointCloud(target)
    if isnothing(scale)
        scale = PointCloudRegistration.avg_nn_dist(target_pc)
    end
    if isnothing(max_spring_length)
        max_spring_length = 2scale
    end

    _register_kc_springs(
        source_pc,
        target_pc,
        scale^2,
        stiffness,
        max_spring_length,
        iterations,
        report_iteration,
    )
end

function _register_kc_springs(
    source::PointCloud{N, T},
    target::PointCloud{N, T},
    sqscale,
    stiffness,
    max_spring_length,
    iterations,
    report_iteration::RI,
) where {N, T, RI}
    tree = KDTree(source.points)
    spring_pairs = inrange_pairs(tree, max_spring_length)
    spring_dists = [
        euclidean(source.points[j1], source.points[j2])
        for (j1, j2) in spring_pairs
    ]

    objective = KcSpringsObjective(source, target, spring_pairs, spring_dists, sqscale, stiffness)
    params = zeros(eltype(eltype(source.points)), N * length(source.points))
    interpret_params(params, Val(N)) .= source.points
    @info "does this work?" objective(params)

    config = Mooncake.Config(; friendly_tangents = false)
    mc_cache = prepare_gradient_cache(objective, params; config)
    adam = Adam.State(params)
    @info "setup done"

    for iter in 1:iterations
        obj_val, grad = value_and_gradient!!(mc_cache, objective, params)
        Adam.step!(adam, grad[2], params, iter)
        report_iteration(; iter, gradient = grad[2], obj_val, displaced_source_points = interpret_params(params, Val(N)))
        Adam.isdone(adam) && break
    end

    return KcSpringsRegistration(
        interpret_params(params, Val(N)) .- source.points
    )
end

end
