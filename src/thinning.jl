function thin_dpmeans(pc::PointCloud, cutoffdist; iterations = 100)
    sqcutoffdist = cutoffdist ^ 2
    centers = [pc.mean]
    cluster_sumws = [zero(float(eltype(pc.weights)))]
    nearest_idcs  = [1]
    dists = [zero(eltype(pc))]
    indicators = ones(Int, length(pc.points))

    changed = 0.0
    for iter in 1:iterations
        @info "iteration" iter length(centers)
        changed = 0.0
        tree = KDTree(copy(centers))
        for (i, point) in enumerate(pc.points)
            w = pc.weights[i]
            iszero(w) && continue
            # sqdist, j = findmin(Base.Fix2(sqeuclidean, point), centers)
            # j = only(nearest_idcs)
            # sqdist = only(dists)^2
            j, dist = nn(tree, point)
            sqdist = dist^2
            if isnan(sqdist)
                @warn "sqdist is NaN" i length(centers)
                break
            end
            if sqdist <= sqcutoffdist
                # @info "nn" i j sqrt(sqdist) iter
                if indicators[i] != j
                    changed += w
                    indicators[i] = j
                end
            else
                # @info "nn" i j sqrt(sqdist) iter
                changed += w
                push!(centers, point)
                push!(cluster_sumws, zero(eltype(cluster_sumws)))
                tree = KDTree(centers)
                indicators[i] = length(centers)
            end
        end
        @info "proportion changed" changed/pc.sum_of_weights
        if changed/pc.sum_of_weights < 1e-2
            @info "converged" iter
            break
        end
        fill!(cluster_sumws, zero(eltype(cluster_sumws)))
        fill!(centers, zero(eltype(centers)))
        for i in eachindex(indicators, pc.points, pc.weights)
            w = pc.weights[i]
            iszero(w) && continue
            l = indicators[i]
            cluster_sumws[l] += w
            centers[l] += w * pc.points[i]
        end
        is_empty = iszero.(cluster_sumws)
        first_empty_idx = count(!, is_empty)
        if first_empty_idx > 0
            perm = sortperm(is_empty)
            permute!(centers, perm)
            permute!(cluster_sumws, perm)
            deleteat!(centers, first_empty_idx:lastindex(centers))
            deleteat!(cluster_sumws, first_empty_idx:lastindex(cluster_sumws))
            for i in eachindex(indicators)
                indicators[i] = perm[indicators[i]]
            end
        end
        centers ./= cluster_sumws
    end

    non_empty = cluster_sumws .> 0
    centers = centers[non_empty]
    cluster_sumws = cluster_sumws[non_empty]
    @info "result" centers cluster_sumws

    PointCloud(centers, cluster_sumws)
end

function thin_droplowweight(pc::PointCloud, p)
    threshold = p * maximum(pc.weights)
    pc[pc.weights .>= threshold]
end
