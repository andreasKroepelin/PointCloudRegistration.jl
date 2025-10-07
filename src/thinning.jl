@kwdef struct DpMeansState{Cs, Ws, Tree, T}
    centers::Cs
    weightsums::Ws
    indicators::Vector{Int}
    permutation::Vector{Int}
    tree::Ref{Tree}
    nn_idcs::Vector{Int}
    nn_dists::Vector{T}
end

function DpMeansState(pc::PointCloud)
    DpMeansState(;
        centers = [pc.mean],
        weightsums = [pc.sum_of_weights],
        indicators = ones(Int, length(pc.points)),
        permutation = [1],
        tree = Ref(KDTree([pc.mean])),
        nn_idcs = [1],
        nn_dists = [euclidean(pc.mean, pc.mean)],
    )
end

update_tree!(dp::DpMeansState) = dp.tree[] = KDTree(dp.centers)

function remove_empty!(dp::DpMeansState; relabel::Bool)
    (; centers, weightsums, indicators, permutation, tree) = dp
    n = length(centers)
    first_empty_idx = count(!iszero, weightsums)
    if first_empty_idx > 0
        resize!(permutation, n)
        sortperm!(permutation, weightsums; by = iszero)
        permute!(centers, permutation)
        permute!(weightsums, permutation)
        deleteat!(centers, first_empty_idx:n)
        deleteat!(weightsums, first_empty_idx:n)
        if relabel
            for i in eachindex(indicators)
                indicators[i] = permutation[indicators[i]]
            end
        end
    end
end

function closest_cluster(dp::DpMeansState, point)
    knn!(dp.nn_idcs, dp.nn_dists, dp.tree[], point, 1)
    dp.nn_idcs[1], dp.nn_dists[1]
end

function setlabel!(dp::DpMeansState, i, j)
    changed = false
    if dp.indicators[i] != j
        changed = true
        dp.indicators[i] = j
    end
    return changed
end

function newcluster!(dp::DpMeansState{Vector{C}}, i::Int, center::C) where {C}
    push!(dp.centers, center)
    newlen = length(dp.centers)
    resize!(dp.weightsums, newlen)
    dp.indicators[i] = newlen
    update_tree!(dp)
end

function recenter!(dp::DpMeansState, pc::PointCloud)
    fill!(dp.weightsums, zero(eltype(dp.weightsums)))
    fill!(dp.centers, zero(eltype(dp.centers)))
    for i in eachindex(dp.indicators, pc.points, pc.weights)
        w = pc.weights[i]
        iszero(w) && continue
        l = dp.indicators[i]
        dp.weightsums[l] += w
        dp.centers[l] += w * pc.points[i]
    end
    dp.centers ./= dp.weightsums
end

numclusters(dp::DpMeansState) = length(dp.centers)

function thin_dpmeans(
    pc::PointCloud,
    cutoffdist;
    iterations = 100,
    convergence = 1e-2,
    report_iteration::RI = no_report,
) where {RI}
    state = DpMeansState(pc)
    for iteration in 1:iterations
        change = 0.0
        additions = 0
        update_tree!(state)
        for i in eachindex(pc.points, pc.weights)
            w = pc.weights[i]
            iszero(w) && continue
            point = pc.points[i]
            j, dist = closest_cluster(state, point)

            if dist <= cutoffdist
                setlabel!(state, i, j) && (change += w)
            else
                newcluster!(state, i, point)
                additions += 1
                change += w
            end
        end
        relchange = change/pc.sum_of_weights
        report_iteration(;
            iteration,
            relchange,
            numclusters = numclusters(state),
            additions,
        )
        relchange < convergence && break

        recenter!(state, pc)
        remove_empty!(state; relabel = true)
    end

    remove_empty!(state; relabel = false)
    PointCloud(state.centers, state.weightsums)
end

function thin_kmeans(
    pc::PointCloud,
    numclusters;
    iterations = 100,
    convergence = 1e-2,
    report_iteration::RI = no_report,
) where {RI}
    # state = DpMeansState(pc)
    centers = sample(pc.points, numclusters; replace = false)
    weightsums = fill(zero(pc.sum_of_weights), numclusters)
    indicators = ones(Int, length(pc.points))
    for iteration in 1:iterations
        change = 0.0
        tree = KDTree(centers)
        for i in eachindex(pc.points, pc.weights, indicators)
            w = pc.weights[i]
            iszero(w) && continue
            point = pc.points[i]
            j, dist = nn(tree, point)

            if indicators[i] != j
                indicators[i] = j
                change += w
            end
        end
        relchange = change/pc.sum_of_weights
        report_iteration(; iteration, relchange)
        relchange < convergence && break

        fill!(centers, zero(eltype(centers)))
        fill!(weightsums, zero(eltype(weightsums)))
        for i in eachindex(pc.points, pc.weights, indicators)
            w = pc.weights[i]
            iszero(w) && continue
            l = indicators[i]
            centers[l] += w * pc.points[i]
            weightsums[l] += w
        end
        centers ./= weightsums
    end

    PointCloud(centers, weightsums)
end

function thin_droplowweight(pc::PointCloud, p)
    threshold = p * maximum(pc.weights)
    pc[pc.weights .>= threshold]
end
