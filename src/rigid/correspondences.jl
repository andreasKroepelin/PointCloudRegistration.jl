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
"""
struct Correspondences
    idcs::Vector{NTuple{2, Int}}
end

"""
    matching_labels(source_labels, target_labels)

Creates [`Correspondences`](@ref) between equal labels.
This is useful if `source_labels` and `target_labels` are lists of labels for
the points in source and target point clouds you plan to rigidly register.

```julia
julia> matching_labels(['A', 'B', 'A', 'C'], ['C', 'B', 'A', 'B'])
Correspondences([(1, 3), (2, 2), (2, 4), (3, 3), (4, 1)])
```
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

"""
    Unknown()

Expresses no preference at all what points in the source correspond to what
points in the target.
A correspondence based registration method will thus have to consider the full
cartesian product.
Instead, it is advised to use a registration algorithm that does not rely on
correspondences.
"""
struct Unknown end

function corresponding_indices(::Ordered, source, target)
    @argcheck length(source) == length(target) "ordered correspondences require point clouds of the same length"
    src_idcs = eachindex(source)
    trg_idcs = eachindex(target)
    return StructVector{NTuple{2, Int}}((src_idcs, trg_idcs))
end

corresponding_indices(c::Correspondences, _source, _target) = c.idcs

function corresponding_indices(::Unknown, source, target)
    src_idcs = eachindex(source)
    trg_idcs = eachindex(target)
    return Tuple.(CartesianIndices((src_idcs, trg_idcs)))
end

"""
    compatible_triangles(correspondences, source, target; [deviation = 0.1, coverage = 100])

Finds [`Correspondences`](@ref) that are a subset of the given `correspondences`
where for each correspondence pair `(j, i)` the points `points(source)[j]` and
`points(target)[i]` have been identified to be part of _compatible triangles_.
Such compatible triangles consists of three source points ``A``, ``B``, ``C``
and three target points ``A'``, ``B'``, ``C'`` with matching side lengths, that
is the length of ``A B`` and the length of ``A' B'`` differ by not more than
10 % for `deviation = 0.1`, and so on for ``A C`` and ``B C``.

Each element of `correspondences` gets `coverage` chances to be part of a
triangle.
"""
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

    unique!(compatible_correspondences)
    sort!(compatible_correspondences)
    return Correspondences(compatible_correspondences)
end
