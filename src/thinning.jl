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
    mean, _ = mean_cov(pc)
    DpMeansState(;
        centers = [mean],
        weightsums = [sum_of_weights(pc)],
        indicators = ones(Int, length(pc)),
        permutation = [1],
        tree = Ref(KDTree([mean])),
        nn_idcs = [1],
        nn_dists = [euclidean(mean, mean)],
    )
end

update_tree!(dp::DpMeansState) = dp.tree[] = KDTree(dp.centers)

function remove_empty!(dp::DpMeansState; relabel::Bool)
    (; centers, weightsums, indicators, permutation, tree) = dp
    n = length(centers)
    first_empty_idx = count(!iszero, weightsums) + 1
    if first_empty_idx > n
        return nothing
    end
    if first_empty_idx > 1
        resize!(permutation, n)
        sortperm!(permutation, weightsums; by = iszero)
        permute!(centers, permutation)
        permute!(weightsums, permutation)
        deleteat!(centers, first_empty_idx:n)
        deleteat!(weightsums, first_empty_idx:n)
        if relabel
            for i in eachindex(indicators)
                old_label = indicators[i]
                if old_label <= n
                    indicators[i] = permutation[old_label]
                else
                    # the cluster this point was assigned to has been deleted
                    indicators[i] = 0
                end
            end
        end
    end
end

closest_cluster(dp::DpMeansState, point) = nn(dp.tree[], point)

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
    fillzeros!(dp.weightsums)
    fillzeros!(dp.centers)
    for i in eachindex(dp.indicators, points(pc), weights(pc))
        w = weights(pc)[i]
        # iszero(w) && continue
        l = dp.indicators[i]
        dp.weightsums[l] += w
        dp.centers[l] += w * points(pc)[i]
    end
    dp.centers ./= dp.weightsums
end

numclusters(dp::DpMeansState) = length(dp.centers)

"""
    thin_to_distance(pointcloud, distance)

Represents the shape of `pointcloud` by a new point cloud where nearest
neighbors have a distance of `distance`.

# Example
```julia
julia> pointcloud = PointCloud(randn(3, 1000))
3-dimensional point cloud with 1000 points of eltype Float64
  0.408793    0.627892  …  -1.01152   0.119291
 -0.0527832  -1.15918      -1.15897  -0.37836
 -2.94632     0.129725      1.69753  -0.439848
and unit weights

julia> thinned = thin_to_distance(pointcloud, 1.0)
3-dimensional point cloud with 78 points of eltype Float64
 -0.00862524   0.590869  …  -1.04324    1.0296
 -0.341446    -0.306642     -0.860819  -2.36446
 -0.234004    -2.19875       1.73244    1.442
and weights
 78-element Vector{Int64}
 40  9  33  22  2  20  …  2  1  17  1  16  5  1
```
"""
function thin_to_distance(
    pc::PointCloud{N},
    distance;
    iterations = 100,
    convergence = 1e-2,
    report_iteration::RI = no_report,
) where {N, RI}
    state = DpMeansState(pc)
    sow = sum_of_weights(pc)

    for iteration in 1:iterations
        change = 0.0
        additions = 0
        update_tree!(state)
        for (i, (point, w)) in enumerate(pc)
            # iszero(w) && continue
            j, dist = closest_cluster(state, point)

            if dist <= distance
                setlabel!(state, i, j) && (change += w)
            else
                newcluster!(state, i, point)
                additions += 1
                change += w
            end
        end
        recenter!(state, pc)
        relchange = change / sow
        report_iteration(;
            iteration,
            relchange,
            numclusters = numclusters(state),
            additions,
        )
        relchange < convergence && break

        remove_empty!(state; relabel = true)
    end

    remove_empty!(state; relabel = false)
    PointCloud(state.centers, state.weightsums)
end

"""
    thin_to_number(pointcloud, number)

Represents the shape of `pointcloud` by a new point cloud with exactly
`number` points.

# Example
```julia
julia> pointcloud = PointCloud(randn(3, 1000))
3-dimensional point cloud with 1000 points of eltype Float64
  0.408793    0.627892  …  -1.01152   0.119291
 -0.0527832  -1.15918      -1.15897  -0.37836
 -2.94632     0.129725      1.69753  -0.439848
and unit weights

julia> thinned = thin_to_number(pointcloud, 100)
3-dimensional point cloud with 100 points of eltype Float64
 -0.00643297  -1.89047   …  -0.18303    0.0797447
 -1.03833      0.51101       2.09415   -1.64585
 -0.30948     -0.157822      0.986703  -0.573778
and weights
 100-element Vector{Int64}
 12  10  14  13  14  11  …  10  12  10  5  10  8
```
"""
function thin_to_number(
    pc::PointCloud,
    numclusters;
    iterations = 100,
    convergence = 1e-2,
    report_iteration::RI = no_report,
) where {RI}
    sow = sum_of_weights(pc)
    centers =
        sample(points(pc), Weights(weights(pc)), numclusters; replace = false)
    weightsums = zeros(typeof(sow), numclusters)
    clustersizes = zeros(Int, numclusters)
    indicators = ones(Int, length(pc))
    for iteration in 1:iterations
        change = 0.0
        tree = KDTree(centers)
        for (i, (point, w)) in enumerate(pc)
            iszero(w) && continue
            j, dist = nn(tree, point)

            if indicators[i] != j
                indicators[i] = j
                change += w
            end
        end
        relchange = change / sow
        report_iteration(; iteration, relchange)
        relchange < convergence && break

        fillzeros!(centers)
        fillzeros!(weightsums)
        fillzeros!(clustersizes)
        for (i, (point, w)) in enumerate(pc)
            iszero(w) && continue
            l = indicators[i]
            centers[l] += w * point
            weightsums[l] += w
            clustersizes[l] += 1
        end
        centers ./= weightsums
        for l in eachindex(centers, weightsums, clustersizes)
            clustersizes[l] > 0 && continue
            point, weight = rand(pc)
            centers[l] = point
            weightsums[l] = weight
        end
    end

    PointCloud(centers, weightsums)
end

"""
    thin_to_grid(pointcloud, gridsize)

Places a grid with side length `gridsize` per cell over `pointcloud` and finds
the centroid of each grid cell.

# Example
```julia
julia> pointcloud = PointCloud(randn(3, 1000))
3-dimensional point cloud with 1000 points of eltype Float64
  0.408793    0.627892  …  -1.01152   0.119291
 -0.0527832  -1.15918      -1.15897  -0.37836
 -2.94632     0.129725      1.69753  -0.439848
and unit weights

julia> thinned = thin_to_grid(pointcloud, 1.0)
3-dimensional point cloud with 132 points of eltype Float64
 -0.646268   0.094386  …  0.703849  -0.90062
 -0.902819  -0.159374     0.267762   1.20793
 -2.81872   -2.78622      2.74575    2.61375
and weights
 132-element Vector{Int64}
 1  2  2  1  1  1  2  4  …  2  1  1  1  2  2  1
```
"""
function thin_to_grid(pc::PointCloud{N, T}, gridsize) where {N, T}
    lo, hi = bbox(pc)
    grid = Grid(lo, hi, gridsize)
    centers = zeros(SVector{N, T}, size(grid)...)
    weightsums = zeros(SumOfWeightsType(pc), size(grid)...)
    for (point, weight) in pc
        idx = idx_on_grid(point, grid)
        centers[idx] += weight * point
        weightsums[idx] += weight
    end
    new_points = eltype(centers)[]
    new_weights = eltype(weightsums)[]
    for (center, weightsum) in zip(centers, weightsums)
        iszero(weightsum) && continue
        push!(new_points, center / weightsum)
        push!(new_weights, weightsum)
    end
    return PointCloud(new_points, new_weights)
end

"""
    drop_threshold(pointcloud, threshold)

Returns a new point cloud with only the points that have a weight of at least
`threshold`.
"""
drop_threshold(pc::PointCloud, threshold::Number) = 
    filter(wp -> wp.weight >= threshold, pc)

"""
    drop_proportion(pointcloud, proportion)

Returns a new point cloud with only the points that have a weight of at least
`proportion` times the maximum weight in `pointcloud`.
"""
function drop_proportion(pc::PointCloud, proportion::Number)
    @argcheck zero(proportion) <= proportion <= oneunit(proportion)
    threshold = proportion * maximum(weights(pc))
    drop_threshold(pc, threshold)
end

"""
    drop_quantile(pointcloud, quantile)

Returns a new point cloud with only the points that have a weight of at least
the `quantile`-quantile of weights in `pointcloud`.
"""
function drop_quantile(pc::PointCloud, q::Number)
    @argcheck zero(q) <= q <= oneunit(q)
    threshold = quantile(weights(pc), q)
    drop_threshold(pc, threshold)
end
