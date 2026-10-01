"""
    Ordered()

Expresses correspondences between points of equal index.
That is, we assume the point clouds have the same size and are _ordered_
identically such that the `i`-th point in the source corresponds to the `i`-th
point in the target.
"""
struct Ordered end

"""
    Correspondences(idcs::AbstractVector{NTuple{2, Int}}, [weights])

Defines correspondences between certain points in the source and the target.
Every element `(j, i)` of `idcs` means that the `j`-th point in the source
corresponds to the `i`-th point to the target.
Optionally, you can provide weights for every correspondence.
"""
struct Correspondences
    idcs::Vector{NTuple{2, Int}}
    # weights::W

    # function Correspondences(idcs, weights)
    #     @argcheck length(idcs) == length(weights)
    #     new{typeof(weights)}(idcs, weights)
    # end
end

# Correspondences(idcs::AbstractVector{NTuple{2, Int}}) =
#     Correspondences(idcs, FillArrays.Trues(length(idcs)))

"""
    matching_labels(source_labels, target_labels)

Creates [`Correspondences`](@ref) between those 
"""
function matching_labels(source_labels, target_labels; by = identity)
    idcs = NTuple{2, Int}[]
    for (j, sl) in enumerate(source_labels)
        for (i, tl) in enumerate(target_labels)
            if sl == tl
                push!(idcs, (j, i))
            end
        end
    end
    return Correspondences(idcs)
end

struct Unknown end

function corresponding_indices(::Ordered, source, target)
    @argcheck length(source) == length(target) "ordered correspondences require point clouds of the same length"
    src_idcs = eachindex(source)
    trg_idcs = eachindex(target)
    return StructVector{NTuple{2, Int}}((src_idcs, trg_idcs))
end

# nzidcsvals(mat::AbstractMatrix, _source, _target) =
#     (Base.Broadcast.Broadcasted(Tuple, (CartesianIndices(mat),)), mat)

corresponding_indices(c::Correspondences, _source, _target) = c.idcs

function corresponding_indices(::Unknown, source, target)
    src_idcs = eachindex(source)
    trg_idcs = eachindex(target)
    return Tuple.(CartesianIndices((src_idcs, trg_idcs)))
end

function compatible_triangles(
    correspondences,
    source::PointCloud{N},
    target::PointCloud{N};
    deviation = 0.1,
    coverage = 100,
) where {N}
    ci = corresponding_indices(correspondences, source, target)
    threshold = log1p(deviation)
    compatible_correspondences = eltype(ci)[]
    ntrials = round(Int, coverage * length(ci))

    for _ in 1:ntrials
        a = rand(ci)
        b = rand(ci)
        c = rand(ci)
        iscompatible = true
        for (from, to) in ((a, b), (a, c), (b, c))
            (j1, i1) = from
            (j2, i2) = to
            src_side = euclidean(source[j1].coords, source[j2].coords)
            trg_side = euclidean(target[i1].coords, target[i2].coords)
            if abs(log(src_side / trg_side)) > threshold
                iscompatible = false
                break
            end
        end
        if iscompatible
            push!(compatible_correspondences, a)
            push!(compatible_correspondences, b)
            push!(compatible_correspondences, c)
        end
    end

    sort!(compatible_correspondences)
    unique!(compatible_correspondences)
    return Correspondences(compatible_correspondences)
end
