abstract type AbstractBatch end

"""
    FullBatch()

Always use all the points of the source in every iteration of the rigid
registration algorithm.
"""
struct FullBatch <: AbstractBatch end

weighted_iterator(::FullBatch, items, weights) =
    WeightedIterator(items, weights, nothing, 0, nothing, false)

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

function weighted_iterator(sb::StochasticBatch, items, weights)
    count = min(sb.count, length(weights))
    WeightedIterator(items, weights, cumsum(weights), sb.count, sb.rng, true)
end

struct WeightedIterator{I, W, WC, Rng}
    items::I
    weights::W
    weights_cumsum::WC
    count::Int
    rng::Rng
    stochastic::Bool
end

function _sample_categorical(rng, weights, weights_cumsum)
    r = rand(rng, float(eltype(weights_cumsum)))
    idx = searchsortedfirst(weights_cumsum, r * last(weights_cumsum))
    idx = clamp(idx, eachindex(weights))
    return idx
end

function _sample_categorical(rng, weights::FillArrays.AbstractFill, _weights_cumsum)
    idx = rand(rng, eachindex(weights))
    return idx
end


function Base.iterate(wi::WeightedIterator, i = 1)
    if wi.stochastic
        i > wi.count && return nothing
        idx = _sample_categorical(wi.rng, wi.weights, wi.weights_cumsum)
        weight = one(eltype(wi.weights))
        item = wi.items[idx]
        return ((; item, weight), i + 1)
    else
        i > length(wi.items) && return nothing
        item = wi.items[eachindex(wi.items)[i]]
        weight = wi.weights[eachindex(wi.weights)[i]]
        return ((; item, weight), i + 1)
    end
end

# For type stability during iterations in the registration algorithms, it is
# crucial that this function returns something of the same type as its input.
non_stochastic(wi::WI) where {WI <: WeightedIterator} =
    WeightedIterator(wi.items, wi.weights, wi.weights_cumsum, wi.count, wi.rng, false)::WI
