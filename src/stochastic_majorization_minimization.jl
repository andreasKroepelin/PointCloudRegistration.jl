struct NoSmm end

struct NoSmmIterator{PC <: PointCloud}
    pc::PC
end

smm_iterator(::NoSmm, pc::PointCloud) = NoSmmIterator(pc)

function Base.iterate(nsi::NoSmmIterator, i = 1)
    if i > lastindex(nsi.pc.points)
        return nothing
    end
    ((idx = i, point = nsi.pc.points[i], weight = nsi.pc.weights[i]), i + 1)
end

struct Smm{Rng <: AbstractRNG}
    count::Int
    rng::Rng
end

Smm(count::Int) = Smm(count, Random.default_rng())

struct SmmIterator{PC <: PointCloud, Rng <: AbstractRNG}
    count::Int
    rng::Rng
    pc::PC
end

function smm_iterator(sp::Smm, pc::PointCloud)
    count = min(sp.count, length(pc.points))
    SmmIterator(count, sp.rng, pc)
end

function Base.iterate(spi::SmmIterator, i = 1)
    if i > spi.count
        return nothing
    end
    # we use `true` as a leightweight 1 here because the weighting is covered
    # by the sampling
    ((sample_point(spi.rng, spi.pc)..., weight = true), i + 1)
end
