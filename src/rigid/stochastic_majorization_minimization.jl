abstract type AbstractBatch end

"""
    FullBatch()

Always use all the points of the source in every iteration of the rigid
registration algorithm.
"""
struct FullBatch <: AbstractBatch end

point_cloud_iterator(::FullBatch, pc::PointCloud) =
    PointCloudIterator(0, nothing, pc, false)

"""
    StochasticBatch(count, [rng = Random.default_rng()])

Only use `count` points of the source in every iteration of the rigid
registration algorithm, chosen randomly by the random number generator `rng`.
After 90 % of iterations (unless convergence occurs earlier), the remaining
iterations will be performed with non-stochastic full batches.

In case the source has less than `count` points, `count` is set to the number
of points in the source.

You can make the batching fully deterministic by specifying a random number
generator with fixed seed, i.e.
```julia
using Random

StochasticBatch(50, Xoshiro(123))
```
"""
struct StochasticBatch{Rng <: AbstractRNG} <: AbstractBatch
    count::Int
    rng::Rng
end

StochasticBatch(count::Int) = StochasticBatch(count, Random.default_rng())

function point_cloud_iterator(sb::StochasticBatch, pc::PointCloud)
    count = min(sb.count, length(pc.points))
    PointCloudIterator(count, sb.rng, pc, true)
end

struct PointCloudIterator{PC <: PointCloud, Rng}
    count::Int
    rng::Rng
    pc::PC
    stochastic::Bool
end

function Base.iterate(pci::PointCloudIterator, i = 1)
    if pci.stochastic
        if i > pci.count
            return nothing
        end
        weight = one(eltype(pci.pc.weights))
        ((; sample_point(pci.rng, pci.pc)..., weight), i + 1)
    else
        if i > lastindex(pci.pc.points)
            return nothing
        end
        ((idx = i, point = pci.pc.points[i], weight = pci.pc.weights[i]), i + 1)
    end
end

# For type stability during iterations in the registration algorithms, it is
# crucial that this function returns something of the same type as its input.
non_stochastic(pci::PCI) where {PCI <: PointCloudIterator} =
    PointCloudIterator(pci.count, pci.rng, pci.pc, false)::PCI
