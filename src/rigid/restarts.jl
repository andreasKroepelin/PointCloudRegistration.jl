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

struct RandomRestartIterator{
    Flip <: FlipMarker,
    PS <: PointCloud,
    PT <: PointCloud,
    RR <: RandomRestarts,
}
    source::PS
    target::PT
    random_restarts::RR

    function RandomRestartsIterator{Flip}(source, target, random_restarts) where {Flip <: FlipMarker}
        new{Flip, typeof(source), typeof(target), typeof(random_restarts)}(source, target, random_restarts)
    end
end


function Base.iterate(rri::RandomRestartIterator{Flip}, i = 0) where {Flip}
    if i == 0
        return (simple_transformation(rri.source, rri.target, Flip()), 1)
    elseif i > rri.random_restarts.number
        return nothing
    else
        T = rand_transformation(rri.random_restarts.rng, rri.source, rri.target, Flip())
        return (T, i + 1)
    end
end

struct FixedRestarts{Ts <: AbstractVector{<: AffineMap}} <: AbstractRestarts
    transformations::Ts
end

function restarts_iterator(source, target, rr::RandomRestarts, ::Flip) where {Flip <: FlipMarker}
    RandomRestartIterator{Flip}(source, target, rr)
end

restarts_iterator(_source, _target, fr::FixedRestarts, _flip) = fr.transformations
