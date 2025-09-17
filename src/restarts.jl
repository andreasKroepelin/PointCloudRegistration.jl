abstract type AbstractRestarts end

struct RandomRestarts{Rng} <: AbstractRestarts
    number::Int
    rng::Rng

    function RandomRestarts(number, rng)
        @argcheck number >= 0
        new{typeof(rng)}(number, rng)
    end
end

RandomRestarts(n::Int) = RandomRestarts(n, Random.default_rng())
RandomRestarts() = RandomRestarts(5)

struct RandomRestartIterator{PS <: PointCloud, PT <: PointCloud, RR <: RandomRestarts}
    source::PS
    target::PT
    random_restarts::RR
end

function Base.iterate(rri::RandomRestartIterator, i = 0)
    if i == 0
        return (simple_transformation(rri.source, rri.target), 1)
    elseif i > rri.random_restarts.number
        return nothing
    else
        T = rand_transformation(rri.random_restarts.rng, rri.source, rri.target)
        return (T, i + 1)
    end
end

struct FixedRestarts{Ts <: AbstractVector{<: AffineMap}} <: AbstractRestarts
    transformations::Ts
end

function restarts_iterator(source, target, rr::RandomRestarts)
    RandomRestartIterator(source, target, rr)
end

function restarts_iterator(_source, _target, fr::FixedRestarts)
    fr.transformations
end
