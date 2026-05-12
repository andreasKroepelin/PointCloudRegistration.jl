struct NoSmm end

smm_iterator(::NoSmm, pc::PointCloud) =
    PointCloudIterator(0, nothing, pc, false)

struct Smm{Rng <: AbstractRNG}
    count::Int
    rng::Rng
end

Smm(count::Int) = Smm(count, Random.default_rng())

function smm_iterator(sp::Smm, pc::PointCloud)
    count = min(sp.count, length(pc.points))
    PointCloudIterator(count, sp.rng, pc, true)
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
non_stochastic(pci::PCI)::PCI where {PCI <: PointCloudIterator} =
    PointCloudIterator(pci.count, pci.rng, pci.pc, false)
