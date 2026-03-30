module KernelCorrelationSprings

using PointCloudRegistration
using Mooncake
using Distances
using NearestNeighbors
using StaticArrays

using ..Adam

struct KcSpringsRegistration{D}
    new_source_points::D
end

function PointCloudRegistration.nonrigid_kc_springs(
    source,
    target;
    scale = default_scale(),
    stiffness = 1,
    max_spring_length = nothing,
    iterations = 100_000,
    report_iteration::RI = PointCloudRegistration.no_report
) where {RI}
    source_pc = PointCloud(source)
    target_pc = PointCloud(target)
    prepared_target = prepare_target_kc(target_pc; scale)
    if isnothing(max_spring_length)
        max_spring_length = 2 / last(prepared_target.annealing_levels).grid.invΔ
    end

    _nonrigid_kc_springs(
        source_pc,
        prepared_target,
        scale^2,
        stiffness,
        max_spring_length,
        iterations,
        report_iteration,
    )
end

function PointCloudRegistration.nonrigid_kc_springs(
    source,
    prepared_target::PreparedTarget;
    stiffness = 1,
    max_spring_length = nothing,
    iterations = 100_000,
    report_iteration::RI = PointCloudRegistration.no_report
) where {RI}
    source_pc = PointCloud(source)
    if isnothing(max_spring_length)
        max_spring_length = 2 / last(prepared_target.annealing_levels).grid.invΔ
    end

    _nonrigid_kc_springs(
        source_pc,
        prepared_target,
        stiffness,
        max_spring_length,
        iterations,
        report_iteration,
    )
end

function _nonrigid_kc_springs(
    source::PointCloud{N, T},
    prepared_target::PreparedTarget{N},
    stiffness,
    max_spring_length,
    iterations,
    report_iteration::RI,
) where {N, T, RI}
    (; annealing_levels, target, axis_aligning_rotation) = prepared_target
    valid_idcs = CartesianIndices(size(grid))
    tree = KDTree(source.points)
    spring_pairs = inrange_pairs(tree, max_spring_length)
    spring_dists = [
        euclidean(source.points[j1], source.points[j2])
        for (j1, j2) in spring_pairs
    ]

    new_source_points = copy(source.points)
    gradient = similar(new_source_points)
    adam = Adam.State(to_matrix(new_source_points))

    for annealing_level in annealing_levels
        (; grid, convd_target, convd_weights_target) = annealing_level
        Adam.reset!(adam)
        for iter in 1:iterations
            for j in eachindex(source.weights, gradient, new_source_points)
                transformed_src = new_source_points[j]
                grid_idx = idx_on_grid(transformed_src, grid)
                if !(grid_idx in valid_idcs)
                    gradient[j] = zero(eltype(gradient))
                    continue
                end

                w_src = source.weights[j]
                convd_trg = convd_target[grid_idx]
                convd_w_trg = convd_weights_target[grid_idx]

                gradient[j] = w_src * (convd_w_trg * transformed_y - convd_trg)
            end

            for (original_dist, (j1, j2)) in zip(spring_dists, spring_pairs)
                nsrc1 = new_source_points[j1]
                nsrc2 = new_source_points[j2]
                dist = euclidean(nsrc1, nsrc2)
                diff = (original_dist - dist) / dist * (nsrc1 - nsrc2)
                gradient[j1] += stiffness * diff
                gradient[j2] -= stiffness * diff
            end

            Adam.step!(adam, to_matrix(gradient), to_matrix(new_source_points), iter)
            report_iteration(; iter, gradient, new_source_points)
            Adam.isdone(adam) && break
        end
    end

    return KcSpringsRegistration(new_source_points)
end

end
