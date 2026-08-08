module ExactOptimalTransportExt

using PointCloudRegistration
using ExactOptimalTransport
using Distances

"""
    nonrigid_registration(source, target, algorithm::EarthMover)

Perform non-rigid registration via [`EarthMover`](@ref).
See [here](@ref nonrigid_registration(::Any, ::Any, ::Any)) for general info
about this function.

This method returns a
[`Displacement`](@ref PointCloudRegistration.Displacement)
that can only be applied to `source`.

This method is defined in a package extension that is only available when the
`ExactOptimalTransport.jl` package is loaded.

# Example

```julia
using PointCloudRegistration
using ExactOptimalTransport
using Tulip

nonrigid_registration(source, target, EarthMover(optimizer = Tulip.Optimizer()))
```
"""
function PointCloudRegistration.nonrigid_registration(
    source,
    target,
    algorithm::EarthMover,
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

    new_source_points = map(eachrow(transport)) do trow
        PointCloudRegistration.wsum(target.points, trow) / sum(trow)
    end

    return PointCloudRegistration.Displacement(source.points, new_source_points)
end

end
