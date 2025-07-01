function nn_features(pc::PointCloud; nfeatures = 20)
    tree = KDTree(pc.points)
    _idcs, dists = knn(tree, pc.points, nfeatures + 1, #=sortres:=#true)
    @view stack(dists)[2:end, :]
end

function compatible_triangles(triangle_X, triangle_Y, deviation)
    @assert length(triangle_X) == length(triangle_Y) == 3
    hi = deviation
    lo = deviation / (deviation - 1)
    for from in 1:3
        for to in (from + 1):3
            distx = euclidean(triangle_X[from], triangle_X[to])
            disty = euclidean(triangle_Y[from], triangle_Y[to])
            lo <= 1 - distx / disty <= hi || return false
        end
    end
    return true
end

function guess_correspondences(
    X::PointCloud,
    Y::PointCloud,
    featuresX = nn_features(X),
    featuresY = nn_features(Y);
    compatibility_deviation = 0.1,
    compatibility_coverage = 100.0,
)
    treeX = KDTree(featuresX)
    treeY = KDTree(featuresY)

    nn_of_X, _ = nn(treeY, featuresX)
    nn_of_nn_of_X, _ = nn(treeX, featuresY[:, nn_of_X])
    idcsX = eachindex(X.points)
    mutual = nn_of_nn_of_X .== idcsX
    greedy_correspondences_X = idcsX[mutual]
    greedy_correspondences_Y = nn_of_X[mutual]

    # check compatibility
    compatible_set = Set{Int}()
    ntrials =
        round(Int, compatibility_coverage * length(greedy_correspondences_X))
    for _ in 1:ntrials
        idcs_to_check =
            StatsBase.sample(
                eachindex(greedy_correspondences_X),
                3;
                replace = false,
            ) |> SVector{3}
        triangle_X = @view X.points[greedy_correspondences_X[idcs_to_check]]
        triangle_Y = @view Y.points[greedy_correspondences_Y[idcs_to_check]]

        if compatible_triangles(triangle_X, triangle_Y, compatibility_deviation)
            union!(compatible_set, idcs_to_check)
        end
    end
    compatible = collect(compatible_set)
    greedy_correspondences_X[compatible], greedy_correspondences_Y[compatible]
end
