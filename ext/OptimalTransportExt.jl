module OptimalTransportExt

using PointCloudRegistration
using OptimalTransport
using Distances

struct SinkhornRegistration{N, C<:AbstractMatrix, PS <: PointCloud{N}, PT <: PointCloud{N}}
    transport::C
    source::PS
    target::PT
end

function PointCloudRegistration.register_sinkhorn(
    source, target; epsilon = 1.0,
)
    source_pc = PointCloud(source)
    target_pc = PointCloud(target)
    source_normalized = PointCloud(
        source_pc.points,
        source_pc.weights ./ source_pc.sum_of_weights,
    )
    target_normalized = PointCloud(
        target_pc.points,
        target_pc.weights ./ target_pc.sum_of_weights,
    )
    _register_sinkhorn(source_normalized, target_normalized, epsilon)
end

function _register_sinkhorn(
    source_normalized::PointCloud{N, TS},
    target_normalized::PointCloud{N, TT},
    epsilon
)
    sqdists = pairwise(
        euclidean,
        target_normalized.points,
        source_normalized.points,
    )

    transport = sinkhorn(
        target_normalized.weights,
        source_normalized.weights,
        sqdists,
        epsilon,
    )

    return SinkhornRegistration(transport, source_normalized, target_normalized)
end

function PointCloudRegistration.correspondences(sr::SinkhornRegistration)
    return sr.transport
end

function PointCloudRegistration.displacements(sr::SinkhornRegistration)
    map(eachcolumn(sr.transport), sr.source.points) do tcol, src
        totalt = zero(eltype(tcol))
        dsrc = zero(src)
        for (t, trg) in zip(tcol, sr.target.points)
            totalt += t
            dsrc += t * trg
        end
        inv(totalt) * dsrc - src
    end
end

end
