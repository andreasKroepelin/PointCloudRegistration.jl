module ExactOptimalTransportExt

using PointCloudRegistration
using ExactOptimalTransport
using Distances

function PointCloudRegistration.nonrigid_registration(
    source, target, algorithm::EarthMover
)
    source_pc = PointCloud(source)
    target_pc = PointCloud(target)
    _nonrigid_emd(source_pc, target_pc, algorithm.optimizer)
end

function _nonrigid_emd(
    source::PointCloud{N},
    target::PointCloud{N},
    optimizer,
) where {N}
    costs = pairwise(euclidean, source.points, target.points)
    source_distr = source.weights ./ source.sum_of_weights
    target_distr = target.weights ./ target.sum_of_weights

    transport = emd(source_distr, target_distr, costs, optimizer)

    new_source_points = map(eachcol(transport)) do tcol
        PointCloudRegistration.wsum(target.points, tcol) / sum(tcol)
    end

    return PointCloudRegistration.Displacement(source.points, new_source_points)
end

end
