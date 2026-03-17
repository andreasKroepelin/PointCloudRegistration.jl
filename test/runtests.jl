using Test
using InteractiveUtils
using PointCloudRegistration
import PointCloudRegistration as PCReg
using StaticArrays
using CoordinateTransformations
using Distances
using DelimitedFiles
using LinearAlgebra
using Random
using Base.Iterators

load_1ake() = PointCloud(readdlm("../assets/ake/1ake.csv", ','))
load_4ake() = PointCloud(readdlm("../assets/ake/4ake.csv", ','))

@testset "VecOfSVec" begin
    @test_throws ArgumentError PCReg.to_vec_of_svec(zeros(1, 10))

    for N in 2:4
        mat = zeros(N, 10)
        vecofsvec = PCReg.to_vec_of_svec(mat)
        @test vecofsvec isa AbstractVector
        T = eltype(vecofsvec)
        @test T <: SVector
        @test Size(T)[1] == N
    end
end

@testset "mean_cov_sumw" begin
    m = randn(SVector{3, Float64})
    sqrtcov = randn(SMatrix{3, 3, Float64})
    cov = sqrtcov * sqrtcov'
    n = 100_000
    points = [sqrtcov * randn(SVector{3, Float64}) + m for _ in 1:n]
    weights = ones(n)
    m_, cov_, sumw_ = PCReg.mean_cov_sumw(points, weights)
    @test isapprox(m, m_; atol = 1e-1)
    @test isapprox(cov, cov_; atol = 1e-1)
    @test sumw_ == n

    function alloc_counter(points, weights)
        @allocations PCReg.mean_cov_sumw(points, weights)
    end
    alloc_counter(points, weights) # precompile
    @test alloc_counter(points, weights) == 0
end

@testset "bbox" begin
    pc_1ake = load_1ake()
    lo, hi = PCReg.bbox(pc_1ake)

    # proper lower/upper bound?
    @test all(>=(lo), pc_1ake.points)
    @test all(<=(hi), pc_1ake.points)

    # tight bound?
    N = size(pc_1ake, 1)
    @test all(k -> any(p -> p[k] == lo[k], pc_1ake.points), 1:N)
    @test all(k -> any(p -> p[k] == hi[k], pc_1ake.points), 1:N)
end

@testset "PointCloud indexing" begin
    N = 2
    T = Float64
    pc = PointCloud(zeros(T, N, 5))
    i = 4
    x = pc[:, i]
    @test x == pc.points[i]
    @test x isa SVector{N, T}
end

@testset "mapping PointCloud" begin
    pc_1ake = load_1ake()
    rotation = qr(randn(SMatrix{3, 3, Float64})).Q
    translation = randn(SVector{3, Float64})
    transformation = AffineMap(rotation, translation)

    manual_mapping =
        PointCloud(transformation.(pc_1ake.points), pc_1ake.weights)
    clever_mapping = transformation(pc_1ake)
    @test manual_mapping == clever_mapping

    linear = LinearMap(rotation)

    manual_mapping = PointCloud(linear.(pc_1ake.points), pc_1ake.weights)
    clever_mapping = linear(pc_1ake)
    @test manual_mapping == clever_mapping
end

@testset "weighted PointCloud" begin
    points = load_1ake().points
    weights = rand(0:10, length(points))
    weighted_pc = PointCloud(points, weights)

    repeated_points = mapreduce(fill, vcat, points, weights)
    repeated_pc = PointCloud(repeated_points)

    @test isapprox(weighted_pc.sum_of_weights, repeated_pc.sum_of_weights)
    @test isapprox(weighted_pc.mean, repeated_pc.mean)
    @test isapprox(weighted_pc.coveigvecs, repeated_pc.coveigvecs)
    @test isapprox(weighted_pc.coveigvals, repeated_pc.coveigvals)
end

@testset "no_report" begin
    # test that `no_report` is properly compiled "away"
    target() = nothing
    function experiment(x, xs)
        y = 2x
        PCReg.no_report(; y, a = -x, b = xs)
    end

    target_code = (@code_typed target())[1].code
    experiment_code = (@code_typed experiment(42, rand(4)))[1].code
    @test target_code == experiment_code
end

@testset "transformation_type" begin
    for T in (Float32, Float64), N in 2:4
        mat = zeros(T, N, 10)
        pc = PointCloud(mat)
        @test PCReg.transformation_type(pc, pc) ==
              AffineMap{SMatrix{N, N, T, N * N}, SVector{N, T}}
    end
end

@testset "transformation_from_moments" begin
    for N in 2:3, _ in 1:10
        source_mean = randn(SVector{N, Float64})
        target_mean = randn(SVector{N, Float64})
        cov = randn(SMatrix{N, N, Float64}) |> (A -> A' * A)
        T = PCReg.transformation_from_moments(cov, source_mean, target_mean)

        @test det(T.linear) > 0.5 # should not only be slightly positive
        @test isapprox(T.linear' * T.linear, one(T.linear))
        @test isapprox(T(source_mean), target_mean)
    end
end

@testset "negative_last_column" begin
    A = SA[1 2 3; 4 5 6; 7 8 9]
    B = SA[1 2 -3; 4 5 -6; 7 8 -9]
    @test PCReg.negative_last_column(A) == B
end

@testset "rigid_rmsd" begin
    @test_throws ArgumentError rigid_rmsd(zeros(2, 5), zeros(2, 6))

    pc_1ake = load_1ake()
    rng = Random.Xoshiro(136)
    for _ in 1:10
        T_true = PCReg.rand_transformation(rng, pc_1ake, pc_1ake)
        T = rigid_rmsd(pc_1ake, T_true(pc_1ake))
        @test isapprox(T_true, T)
    end
end

@testset "TargetScales" begin
    pc_1ake = load_1ake()
    min_dist, max_dist = extrema(
        splat(sqeuclidean),
        Iterators.product(pc_1ake.points, pc_1ake.points),
    )
    sqscales = PCReg.annealing_plan(pc_1ake, TargetScales())
    @test all(sqscale -> min_dist <= sqscale <= max_dist, sqscales)
end

@testset "rigid_gmc" begin
    @test_throws ArgumentError rigid_gmc(zeros(2, 5), zeros(2, 6))

    pc_1ake = load_1ake()
    rng = Random.Xoshiro(136)
    for _ in 1:10
        T_true = PCReg.rand_transformation(rng, pc_1ake, pc_1ake)
        T = rigid_gmc(pc_1ake, T_true(pc_1ake); scale = 1.0)
        @test isapprox(T_true, T)
    end

    alloc_wrapper(pc) = @allocations rigid_gmc(pc, pc; scale = 1.0)
    alloc_wrapper(pc_1ake)
    @test alloc_wrapper(pc_1ake) == 0
end

@testset "compute_gaussians" begin
    # test that `compute_gaussians` only performs a look-up
    for T in (Float32, Float64)
        codeinfo, ret = @code_typed PCReg.compute_gaussians(T)
        code = codeinfo.code
        @test ret <: Vector{T}
        @test length(code) == 1
        @test code[1] isa Core.ReturnNode
        @test code[1].val isa Core.GlobalRef

        alloc_wrapper(T) = @allocations PCReg.compute_gaussians(T)
        alloc_wrapper(T)
        @test alloc_wrapper(T) == 0
    end
end

@testset "Grid" begin
    @test_throws ArgumentError PCReg.Grid(SA[1.0, 1.0], SA[0.0, 1.0], 0.5)

    for N in 2:3
        lo = rand(SVector{N, Float64})
        hi = rand(SVector{N, Float64}) .+ 3
        grid = PCReg.Grid(lo, hi, 0.1)

        # test that `grid` contains given bounding box
        @test lo >= grid.lo
        @test hi <= grid.hi

        ranges = PCReg.domains(grid)
        # test that grid centers are matched with themselves
        for ci in CartesianIndices(size(grid))
            grid_center = getindex.(ranges, Tuple(ci))
            @test ci == PCReg.idx_on_grid(grid_center, grid)
        end

        # test that we actually get the closest grid cell for any query
        for _ in 1:100
            # `ts` is elementwise between 0 and 1 so `query` is inside the grid
            ts = rand(SVector{N, Float64})
            query = lo .+ ts .* (hi .- lo)
            grid_idx = PCReg.idx_on_grid(query, grid)
            grid_center = getindex.(ranges, Tuple(grid_idx))
            dist = sqeuclidean(query, grid_center)
            @test all(Iterators.product(ranges...)) do center
                sqeuclidean(SVector(center), query) >= dist - 2eps(dist)
            end
        end
    end
end

@testset "_replace_tail_with_colons" begin
    @test PCReg._replace_tail_with_colons((1, 2, 3), Val(1)) == (1, 2, :)
    @test PCReg._replace_tail_with_colons((1, 2, 3), Val(2)) == (1, :, :)
    @test PCReg._replace_tail_with_colons((1, 2, 3), Val(3)) == (:, :, :)

    @test Core.Compiler.return_type(
        PCReg._replace_tail_with_colons,
        Tuple{Tuple{Int, Int, Int}, Val{2}},
    ) == Tuple{Int, Colon, Colon}
end

@testset "_make_sets" begin
    prototype = (Set([(1, 2, :)]), Set([(1, :, :)]), Set([(:, :, :)]))
    @test typeof(PCReg._make_sets(Val(3))) == typeof(prototype)
end

@testset "SetsOfSlices" begin
    idcs = [
        CartesianIndex(1, 2, 1),
        CartesianIndex(1, 2, 2),
        CartesianIndex(1, 2, 3),
        CartesianIndex(1, 5, 0),
        CartesianIndex(1, 5, 6),
        CartesianIndex(2, 4, 0),
        CartesianIndex(2, 7, 1),
    ]

    sos = PCReg.SetsOfSlices(idcs)
    @test Set(sos.sets[1]) == Set([(1, 2, :), (1, 5, :), (2, 4, :), (2, 7, :)])
    @test Set(sos.sets[2]) == Set([(1, :, :), (2, :, :)])
    @test Set(sos.sets[3]) == Set([(:, :, :)])
end

@testset "KDE" begin
    # test that our KDE computation is reasonably close to naive computation
    pc_1ake = load_1ake()
    sigma = 20.0
    grid = PCReg.kde_grid(PCReg.bbox(pc_1ake)...; sigma)
    kde! = PCReg.KdeComputation(pc_1ake.points, grid, sigma^2)
    buffer = zeros(size(grid))
    kde!(buffer, pc_1ake.weights)

    # this is the smallest number we consider > zero for KDE purposes
    almost_zero = last(PCReg.compute_gaussians(Float64))
    atol = length(pc_1ake.points) * almost_zero
    @info "KDE" size(grid) maximum(buffer) atol

    grid_centers = Iterators.product(PCReg.domains(grid)...)
    for (ci, center) in zip(CartesianIndices(buffer), grid_centers)
        rand() < 0.1 || continue # only test 10 % to save time
        s = 0.0
        for (x, w) in zip(pc_1ake.points, pc_1ake.weights)
            s += w * exp(sqeuclidean(SVector(center), x) / (-2 * sigma^2))
        end
        @test buffer[ci] >= 0
        @test isapprox(buffer[ci], s; atol)
    end
end

@testset "rigid_kc" begin
    pc_1ake = load_1ake()
    prep_1ake = prepare_target_kc(pc_1ake)
    rng = Random.Xoshiro(136)
    for _ in 1:10
        T_true = PCReg.rand_transformation(rng, pc_1ake, pc_1ake)
        T = rigid_kc(
            inv(T_true)(pc_1ake),
            prep_1ake;
            restarts = RandomRestarts(20, rng),
        )
        @test isapprox(T_true, T, rtol = 5e-2)
    end

    alloc_wrapper(pc, prep) = @allocations rigid_kc(pc, prep)
    alloc_wrapper(pc_1ake, prep_1ake)
    @test alloc_wrapper(pc_1ake, prep_1ake) == 0
end
