struct OrderedCorrespondences{SI, TI}
    src_idcs::SI
    trg_idcs::TI
end

function OrderedCorrespondences(source::PointCloud, target::PointCloud)
    @argcheck size(source) == size(target) "ordered correspondences require point clouds of the same size"
    OrderedCorrespondences(eachindex(source.points), eachindex(target.points))
end

struct MatchingLabels
    idcs::Vector{NTuple{2, Int}}
end

function MatchingLabels(source_labels, target_labels)
    idcs = NTuple{2, Int}[]
    for (j, sl) in enumerate(source_labels)
        for (i, tl) in enumerate(target_labels)
            if sl == tl
                push!(idcs, (j, i))
            end
        end
    end
    return MatchingLabels(idcs)
end

nzidcs(oc::OrderedCorrespondences) =
    Base.Broadcast.Broadcasted(tuple, (oc.src_idcs, oc.trg_idcs))
nzvals(oc::OrderedCorrespondences) = FillArrays.Trues(length(oc.src_idcs))

nzidcs(mat::AbstractMatrix) =
    Base.Broadcast.Broadcasted(Tuple, (vec(CartesianIndices(mat)),))
nzvals(mat::AbstractMatrix) = vec(mat)

nzidcs(ml::MatchingLabels) = ml.idcs
nzvals(ml::MatchingLabels) = FillArrays.Trues(length(ml.idcs))

