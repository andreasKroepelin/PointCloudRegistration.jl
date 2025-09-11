using Test
using InteractiveUtils
using PointCloudRegistration
import PointCloudRegistration as PCR
using PointCloudRegistration.StaticArrays
using PointCloudRegistration.CoordinateTransformations
using DelimitedFiles
using LinearAlgebra

@testset "VecOfSVec" begin
    @test_throws ArgumentError PCR.VecOfSVec(zeros(1, 10))

    for N in 2:4
        mat = zeros(N, 10)
        vecofsvec = PCR.VecOfSVec(mat)
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
    m_, cov_, sumw_ = PCR.mean_cov_sumw(points, weights)
    @test isapprox(m, m_; atol = 1e-1)
    @test isapprox(cov, cov_; atol = 1e-1)
    @test sumw_ == n

    function alloc_counter(points, weights)
        @allocations PCR.mean_cov_sumw(points, weights)
    end
    alloc_counter(points, weights) # precompile
    @test alloc_counter(points, weights) == 0
end

@testset "bbox" begin
    pc_1ake = PointCloud(readdlm("../assets/ake/1ake.csv", ','))
    lo, hi = PCR.bbox(pc_1ake)

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
    pc_1ake = PointCloud(readdlm("../assets/ake/1ake.csv", ','))
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

@testset "no_report" begin
    # test that `no_report` is properly compiled "away"
    target() = nothing
    function experiment(x, xs)
        y = 2x
        PCR.no_report(; y, a = -x, b = xs)
    end

    target_code = (@code_typed target())[1].code
    experiment_code = (@code_typed experiment(42, rand(4)))[1].code
    @test target_code == experiment_code
end

@testset "transformation_type" begin
    for T in (Float32, Float64), N in 2:4
        mat = zeros(T, N, 10)
        pc = PointCloud(mat)
        @test PCR.transformation_type(pc, pc) ==
              AffineMap{SMatrix{N, N, T, N * N}, SVector{N, T}}
    end
end
