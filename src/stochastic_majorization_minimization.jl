struct AllPoints end

struct AllPointsIterator{PC <: PointCloud}
    pc::PC
end

smm_iterator(::AllPoints, pc::PointCloud) = AllPointsIterator(pc)

function Base.iterate(api::AllPointsIterator, i = 1)
    if i > lastindex(api.pc.points)
        return nothing
    end
    ((idx = i, point = api.pc.points[i], weight = api.pc.weights[i]), i + 1)
end

struct SomePoints{Rng <: AbstractRNG}
    count::Int
    rng::Rng
end

SomePoints(count::Int) = SomePoints(count, Random.default_rng())

struct SomePointsIterator{PC <: PointCloud, Rng <: AbstractRNG}
    count::Int
    rng::Rng
    pc::PC
end

function smm_iterator(sp::SomePoints, pc::PointCloud)
    count = min(sp.count, length(pc.points))
    SomePointsIterator(count, sp.rng, pc)
end

function Base.iterate(spi::SomePointsIterator, i = 1)
    if i > spi.count
        return nothing
    end
    # we use `true` as a leightweight 1 here because the weighting is covered
    # by the sampling
    ((sample_point(spi.rng, spi.pc)..., weight = true), i + 1)
end
