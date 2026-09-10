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

struct Batched{Rng, I <: AbstractVector}
    items::I
    rng::Rng
    batchsize::Int
    stochastic::Bool
end

is_stochastic(b::Batched) =
    b.rng !== nothing && b.stochastic && 2 * b.batchsize <= length(b.items)

non_stochastic(b::Batched) = @set b.stochastic = false

maybe_stochastic(b::Batched{Nothing}) = b
maybe_stochastic(b::Batched) = @set b.stochastic = true

function batched(sb::StochasticBatch, items)
    batchsize = min(sb.count, length(items))
    # we collect such that we later have a `Vector` for shuffling
    return Batched(collect(items), sb.rng, batchsize, true)
end

batched(::FullBatch, items) = Batched(items, nothing, length(items), false)

function batch(b::Batched, iteration::Int)
    if is_stochastic(b)
        number_of_batches = length(b.items) ÷ b.batchsize
        pos = mod1(iteration, number_of_batches)
        start = (pos - 1) * b.batchsize + 1
        stop = if pos == number_of_batches
            lastindex(b.items)
        else
            start + b.batchsize - 1
        end
        if pos == 1
            shuffle!(b.rng, b.items)
        end
    else
        start = firstindex(b.items)
        stop = lastindex(b.items)
    end
    return @view b.items[start:stop]
end

function batch_length_ratio(b::Batched)
    length(b.items) / b.batchsize
end


