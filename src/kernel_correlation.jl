struct Grid{N, T}
    lo::SVector{N, T}
    hi::SVector{N, T}
    invΔ::T
    size::NTuple{N, Int}

    function Grid(lo::SVector{N}, hi::SVector{N}, Δ) where {N}
        invΔ = inv(Δ)
        new{N, promote_type(eltype(lo), eltype(hi), typeof(invΔ))}(
            lo,
            hi,
            invΔ,
            Tuple(round.(Int, (hi - lo) * invΔ)),
        )
    end
end

idx_on_grid(x, grid::Grid{N}) where {N} = idx_on_grid(SVector{N}(x), grid)
idx_on_grid(x::SVector{N}, grid::Grid{N}) where {N} =
    CartesianIndex(round.(Int, (x - grid.lo) * grid.invΔ)...)

extent(grid::Grid) = grid.hi - grid.lo
Base.size(grid::Grid) = grid.size
domains(grid::Grid) = range.(grid.lo, grid.hi, grid.size)

struct FftGaussian{N, T} <: AbstractArray{T, N}
    size::NTuple{N, Int}
    characteristic_functions::NTuple{N, Vector{T}}

    function FftGaussian(grid::Grid{N, T}, sqsigma) where {N, T}
        freq_steps = -2pi ./ Tuple(extent(grid))
        cfs = map(freq_steps, size(grid)) do freq_step, s
            lo = 0
            hi = s ÷ 2
            exp.(-sqsigma ./ 2 .* (freq_step .* (lo:hi)) .^ 2)
        end
        new{N, T}(size(grid), cfs)
    end
end

Base.size(fg::FftGaussian) = fg.size

function Base.getindex(fg::FftGaussian{N}, idcs::Vararg{Int, N}) where {N}
    @boundscheck all(0 .< idcs .<= fg.size)

    pos = idcs .- 1
    pos = min.(pos, fg.size .- pos)
    factors = map((cf, p) -> cf[p + 1], fg.characteristic_functions, pos)
    prod(factors)
end

struct KdeComputation{
    N,
    T,
    B <: AbstractArray{Complex{T}, N},
    Pl <: FFTW.AbstractFFTs.Plan{Complex{T}},
    Pli <: FFTW.AbstractFFTs.Plan{Complex{T}},
}
    grid::Grid{N, T}
    grid_idcs::Vector{CartesianIndex{N}}
    sqsigma::T
    buffer_space::B
    buffer_freq::B
    fft_plan::Pl
    ifft_plan::Pli
    fft_gaussian::FftGaussian{N, T}

    function KdeComputation(
        points::VecOfSVec{N, T},
        grid::Grid{N, T},
        sqsigma::T,
    ) where {N, T}
        grid_idcs = idx_on_grid.(points, (grid,))
        buffer_space = Array{complex(T), N}(undef, size(grid))
        buffer_freq = Array{complex(T), N}(undef, size(grid))
        fft_plan = plan_fft(buffer_space)
        ifft_plan = plan_ifft(buffer_freq)
        fft_gaussian = FftGaussian(grid, sqsigma)
        new{N, T, typeof(buffer_space), typeof(fft_plan), typeof(ifft_plan)}(
            grid,
            grid_idcs,
            sqsigma,
            buffer_space,
            buffer_freq,
            fft_plan,
            ifft_plan,
            fft_gaussian,
        )
    end
end

function (kdecomp!::KdeComputation{N, T})(
    res::AbstractArray{T, N},
    weights::AbstractVector{<:Real},
) where {N, T}
    (;
        grid,
        grid_idcs,
        sqsigma,
        buffer_space,
        buffer_freq,
        fft_plan,
        ifft_plan,
        fft_gaussian,
    ) = kdecomp!
    fill!(buffer_space, zero(eltype(buffer_space)))
    for (idx, weight) in zip(grid_idcs, weights)
        buffer_space[idx] += weight
    end

    mul!(buffer_freq, fft_plan, buffer_space)
    buffer_freq .*= fft_gaussian
    mul!(buffer_space, ifft_plan, buffer_freq)

    # broadcasting leads to enourmous allocations for some reason, so let's use
    # map!...
    # res .= real.(buffer_space)
    # for i in eachindex(res, buffer_space)
    #     res[i] = real(buffer_space[i])
    # end
    map!(real, res, buffer_space)
end

struct AnnealingLevel{
    N,
    T,
    CT <: AbstractArray{<:SVector{N, T}, N},
    CWT <: AbstractArray{T, N},
}
    grid::Grid{N, T}
    convd_target::CT
    convd_weights_target::CWT

    function AnnealingLevel(
        grid::Grid{N, T},
        convd_target,
        convd_weights_target,
    ) where {N, T}
        new{N, T, typeof(convd_target), typeof(convd_weights_target)}(
            grid,
            convd_target,
            convd_weights_target,
        )
    end
end

function AnnealingLevel(
    target::PointCloud{N},
    target_bbox::NTuple{2},
    weighted_target_points,
    sqscale,
) where {N}
    target_lo, target_hi = target_bbox
    sigma = sqrt(sqscale)
    grid = Grid(target_lo .- 3sigma, target_hi .+ 3sigma, sigma / 4)
    convd_target = reinterpret(
        reshape,
        SVector{N, eltype(target)},
        zeros(eltype(target), N, size(grid)...),
    )
    convd_weights_target = zeros(eltype(target), size(grid)...)
    kde! = KdeComputation(target.points, grid, sqscale)
    kde!(convd_weights_target, target.weights)
    kde!.(
        eachslice(parent(convd_target); dims = 1),
        eachrow(
            reinterpret(
                reshape,
                eltype(eltype(weighted_target_points)),
                weighted_target_points,
            ),
        ),
    )
    # for (a, b) in zip(each1slice(convd_target), eachrow(weighted_target_points))
    #     kde!(a, b)
    # end
    # display(@code_typed kde!.(each1slice(convd_target), eachrow(weighted_target_points)))
    # println("\n"^5)
    AnnealingLevel(grid, convd_target, convd_weights_target)
end

function compute_annealing_levels(target, sqscales)
    target_bbox = bbox(target)
    weighted_target_points = target.points .* target.weights

    [
        AnnealingLevel(target, target_bbox, weighted_target_points, sqscale) for
        sqscale in sqscales
    ]
end

function eval_kernel_correlation(
    al::AnnealingLevel,
    source::PointCloud,
    transformation,
)
    valid_idcs = CartesianIndices(size(al.grid))
    kc = zero(eltype(source))
    for j in eachindex(source.points, source.weights)
        src = source.points[j]
        w_src = source.weights[j]
        transformed_src = transformation(src)
        grid_idx = idx_on_grid(transformed_src, al.grid)
        grid_idx in valid_idcs || continue
        convd_w_trg = al.convd_weights_target[grid_idx]
        kc += w_src * convd_w_trg
    end
    kc
end

function axisalign_target(target)
    axisaligner = target.coveigvecs'
    if det(axisaligner) < 0
        axisaligner = negative_last_column(axisaligner)
    end
    axisaligner, LinearMap(axisaligner)(target)
end

struct PreparedTarget{
    N,
    T,
    R <: SMatrix{N, N, T},
    Al <: AnnealingLevel,
    PC <: PointCloud{N, T},
}
    axis_aligning_rotation::R
    annealing_levels::Vector{Al}
    target::PC
end

Base.eltype(::PreparedTarget{N, T}) where {N, T} = T

"""
    prepare_target_kc(target[; scale, axisalign])

Perform all the necessary precomputation for `register_kc`.
This function is especially useful if you plan to register multiple sources to
the same target.

```julia
X = PointCloud(rand(3, 100))
Y1 = PointCloud(rand(3, 150))
Y2 = PointCloud(rand(3, 130))

prepd_X = prepare_target_kc(X) # this takes some time

T1 = register_kc(prepd_X, Y1) # this is fast
T2 = register_kc(prepd_X, Y2) # this is fast
```
"""
function prepare_target_kc(
    target::AbstractMatrix;
    scale::ScaleType = default_config().scale,
    axisalign::Bool = default_config().axisalign,
)
    _prepare_target(PointCloud(target), scale, axisalign)
end

function _prepare_target(target_original, scale, axisalign)
    if axisalign
        axis_aligning_rotation, target = axisalign_target(target_original)
    else
        target = target_original
        axis_aligning_rotation = one(rotation_type(target, target))
    end
    sqscales = annealing_plan(target, scale)
    annealing_levels = compute_annealing_levels(target, sqscales)
    PreparedTarget(axis_aligning_rotation, annealing_levels, target)
end

"""
    register_kc(source, target[; scale, axisalign, restarts, iterations, rng, accumulator])

$REGISTER_DOCS_START
maximizes the Kernel Correlation to `target`,
i.e.
```math
\\sum_{i = 1}^n \\sum_{j = 1}^m \\exp\\left(- \\frac{\\Vert R y_i + t - x_i \\Vert}{2 \\sigma^2}\\right)
```
$REGISTER_DOCS_SYMBOLS

$REGISTER_DOCS_TYPES

The bandwidth parameter ``\\sigma`` corresponds to the `scale` keyword argument
and you can learn about how to use the keyword arguments in
[this section](#Common-keyword-arguments).

Use this function if you do not know correspondences.
"""
function register_kc(
    source,
    target::AbstractMatrix;
    scale::ScaleType = default_config().scale,
    axisalign::Bool = default_config().axisalign,
    restarts::Int = default_config().restarts,
    iterations::Int = default_config().iterations,
    rng = default_config().rng,
    accumulator::Type{<: AbstractTransformationAccumulator} = default_config().accumulator,
)
    prepared_target = prepare_target_kc(target; scale, axisalign)

    _register_kc(
        PointCloud(source),
        prepared_target,
        restarts,
        iterations,
        rng,
        accumulator,
    )
end

"""
    register_kc(source, prepared_target[; restarts, iterations, rng, accumulator])

$REGISTER_DOCS_START
maximizes the Kernel Correlation to `prepared_target` which was computed by
[`prepare_target_kc`](@ref).

Use this function if you plan to register multiple sources to the same target.
"""
function register_kc(
    source,
    prepared_target::PreparedTarget;
    restarts::Int = default_config().restarts,
    iterations::Int = default_config().iterations,
    rng = default_config().rng,
    accumulator::Type{<: AbstractTransformationAccumulator} = default_config().accumulator,
)
    _register_kc(
        PointCloud(source),
        prepared_target,
        restarts,
        iterations,
        rng,
        accumulator,
    )
end

function _register_kc(
    source::PointCloud{N, TS},
    prepared_target::PreparedTarget{N, TT},
    restarts,
    iterations,
    rng,
    accumulator_type,
) where {N, TS, TT}
    T = promote_type(TS, TT)
    (; annealing_levels, target, axis_aligning_rotation) = prepared_target
    source_grid_idcs = zeros(CartesianIndex{N}, length(source.points))

    accumulator = accumulator_type(transformation_type(Val(N), T))
    init_transformation = simple_transformation(source, target)
    restart = 0
    kc = zero(T)
    @withprogress name="performing restarts" while true
        transformation = init_transformation

        for annealing_level in annealing_levels
            (; grid, convd_target, convd_weights_target) = annealing_level
            valid_idcs = CartesianIndices(size(grid))

            @logmsg LogLevel(-2000) "mm iteration" restart iter = -1 kc =
                eval_kernel_correlation(
                    last(annealing_levels),
                    source,
                    transformation,
                ) rotation = transformation.linear translation =
                transformation.translation init_transformation target_kde =
                copy(convd_weights_target) grid id = :mm

            for iter in 1:iterations
                changed = false
                target_mean = zero(SVector{N, TT})
                source_mean = zero(SVector{N, TS})
                kc = zero(T)
                for j in eachindex(source.points)
                    src = source.points[j]
                    transformed_src = transformation(src)
                    grid_idx = idx_on_grid(transformed_src, grid)
                    if source_grid_idcs[j] != grid_idx
                        changed = true
                    end
                    source_grid_idcs[j] = grid_idx
                    grid_idx in valid_idcs || continue

                    w_src = source.weights[j]
                    convd_trg = convd_target[grid_idx]
                    convd_w_trg = convd_weights_target[grid_idx]
                    target_mean += w_src * convd_trg
                    source_mean += w_src * convd_w_trg * src
                    kc += w_src * convd_w_trg
                end
                if !changed && iter > 1
                    break
                end
                target_mean /= kc
                source_mean /= kc
                covariance = zero(rotation_type(Val(N), T))
                for j in eachindex(source.points)
                    grid_idx = source_grid_idcs[j]
                    grid_idx in valid_idcs || continue

                    src = source.points[j]
                    w_src = source.weights[j]
                    convd_trg = convd_target[grid_idx]
                    convd_w_trg = convd_weights_target[grid_idx]
                    covariance +=
                        w_src *
                        (convd_trg - convd_w_trg * target_mean) *
                        (src - source_mean)'
                end
                transformation = transformation_from_moments(
                    covariance,
                    source_mean,
                    target_mean,
                )
                @logmsg LogLevel(-2000) "mm iteration" restart iter kc =
                    eval_kernel_correlation(
                        last(annealing_levels),
                        source,
                        transformation,
                    ) rotation = transformation.linear translation =
                    transformation.translation init_transformation target_kde =
                    copy(convd_weights_target) grid id = :mm
            end
        end
        accumulator =
            update(accumulator, TransformationWithCost(-kc, transformation))
        if restart < restarts
            restart += 1
            @logprogress (restart / restarts)
            init_transformation = rand_transformation(rng, source, target)
        else
            @logprogress "done"
            break
        end
    end

    result(accumulator, LinearMap(axis_aligning_rotation'))
end

struct Rff{N, T}
    params::Vector{@NamedTuple{w::SVector{N, T}, b::T}}
    # ws::Vector{SVector{N, T}}
    # bs::Vector{T}
    # sigma::T
end

function Rff{N, T}(num_features::Int) where {N, T}
    # T = float(typeof(sigma))
    params =
        [(w = randn(SVector{N, T}), b = 2pi * rand(T)) for _ in 1:num_features]
    Rff{N, T}(params)
    # ws = randn(SVector{N, T}, num_features)
    # ws .*= inv(sigma)
    # bs = rand(T, num_features)
    # bs .*= 2pi
    # Rff{N, T}(ws, bs, sigma)
end

num_features(rff::Rff) = length(rff.params)

struct RffPointCloud{
    N,
    T,
    PC <: PointCloud{N, T},
    TP <: AbstractVector{SVector{N, T}},
    TW <: AbstractVector{T},
}
    pointcloud::PC
    rff::Rff{N, T}
    points::TP
    weights::TW
    feature_buffer::Vector{T}
end

function RffPointCloud(pc::PointCloud{N, T}, rff::Rff{N, T}) where {N, T}
    points = similar(pc.points, num_features(rff))
    weights = similar(pc.weights, T, num_features(rff))
    feature_buffer = zeros(T, num_features(rff))
    RffPointCloud(pc, rff, points, weights, feature_buffer)
end

function compute!(
    rffpc::RffPointCloud{N, T},
    scale,
    transformation = identity,
) where {N, T}
    (; feature_buffer, rff, pointcloud) = rffpc
    factor = T(sqrt(2 / num_features(rff)))
    inv_sigma = T(inv(scale))
    rffpc.weights .= zero(T)
    rffpc.points .= (zero(eltype(rffpc.points)),) # wrap in tuple for broadcasting
    for i in eachindex(pointcloud.points)
        x = pointcloud.points[i]
        w = pointcloud.weights[i]
        transformed_x = transformation(x)
        map!(feature_buffer, rff.params) do (; w, b)
            factor * cos(inv_sigma * dot(w, transformed_x) + b)
        end
        rffpc.weights .+= w .* feature_buffer
        for r in eachindex(rffpc.points)
            rffpc.points[r] += w * x * feature_buffer[r]
        end
    end
end

function register_kc_rff(
    source,
    target;
    scale::ScaleType = default_config().scale,
    features::Int = default_config().features,
    restarts::Int = default_config().restarts,
    iterations::Int = default_config().iterations,
    rng = default_config().rng,
    accumulator::Type{<: AbstractTransformationAccumulator} = default_config().accumulator,
)
    pc_source = PointCloud(source)
    pc_target = PointCloud(target)
    sqscales = annealing_plan(pc_target, scale)
    _register_kc_rff(
        pc_source,
        pc_target,
        sqscales,
        features,
        restarts,
        iterations,
        rng,
        accumulator,
    )
end

function _register_kc_rff(
    source::PointCloud{N, TS},
    target::PointCloud{N, TT},
    sqscales,
    num_features,
    restarts,
    iterations,
    rng,
    accumulator_type,
) where {N, TS, TT}
    T = promote_type(TS, TT)
    rff = Rff{N, T}(num_features)
    rff_target = RffPointCloud(target, rff)
    rff_source = RffPointCloud(source, rff)
    # for debugging:
    kc_weights = zeros(T, length(target.points), length(source.points))
    target_kc_weights = similar(kc_weights, length(target.points), 1)
    source_kc_weights = similar(kc_weights, 1, length(source.points))

    accumulator = accumulator_type(transformation_type(Val(N), T))
    init_transformation = simple_transformation(source, target)
    restart = 0
    kc = zero(T)
    @withprogress name="performing restarts" while true
        transformation = init_transformation

        for sqscale in sqscales
            scale = sqrt(sqscale)
            # for debugging:
            kernel = GaussKernel(-inv(2sqscale))
            compute!(rff_target, scale)
            for iter in 1:iterations
                compute!(rff_source, scale, transformation)

                kc = zero(T)
                target_mean = zero(SVector{N, TT})
                source_mean = zero(SVector{N, TS})
                covariance = zero(rotation_type(Val(N), T))
                for r in 1:num_features
                    trg_f = rff_target.points[r]
                    src_f = rff_source.points[r]
                    trg_f_w = rff_target.weights[r]
                    src_f_w = rff_source.weights[r]

                    kc += trg_f_w * src_f_w
                    target_mean += src_f_w * trg_f
                    source_mean += trg_f_w * src_f
                    covariance += trg_f * src_f'
                end
                target_mean /= kc
                source_mean /= kc
                covariance -= kc * target_mean * source_mean'

                # for debugging
                compute_kc_weights!(
                    kc_weights,
                    source,
                    target,
                    kernel,
                    transformation,
                )
                kc_exact = sum(kc_weights)
                sum!(target_kc_weights, kc_weights)
                sum!(source_kc_weights, kc_weights)
                target_mean_exact =
                    wsum(target.points, target_kc_weights) / kc_exact
                source_mean_exact =
                    wsum(source.points, source_kc_weights) / kc_exact

                @info "iteration" iter kc kc_exact string(target_mean) string(
                    target_mean_exact,
                ) string(source_mean) string(source_mean_exact)

                transformation = transformation_from_moments(
                    covariance,
                    source_mean,
                    target_mean,
                )
            end
        end
        accumulator =
            update(accumulator, TransformationWithCost(-kc, transformation))
        if restart < restarts
            restart += 1
            @logprogress (restart / restarts)
            init_transformation = rand_transformation(rng, source, target)
        else
            @logprogress "done"
            break
        end
    end

    result(accumulator, LinearMap(I))
end

struct GaussKernel{T}
    neghalfinvsqbandwidth::T
end

(kernel::GaussKernel)(sqr::Real) = exp(kernel.neghalfinvsqbandwidth * sqr)

function (kernel::GaussKernel)(x::AbstractVector, y::AbstractVector)
    kernel(sqeuclidean(x, y))
end

function compute_kc_weights!(
    kc_weights,
    source::PointCloud,
    target::PointCloud,
    kernel,
    transformation,
)
    @inbounds for j in axes(kc_weights, 2)
        src = source.points[j]
        srcw = source.weights[j]
        transformed_src = transformation(src)
        for i in axes(kc_weights, 1)
            trg = target.points[i]
            trgw = target.weights[i]
            kc_weights[i, j] = trgw * srcw * kernel(transformed_src, trg)
        end
    end
end

function register_kc_naive(
    source,
    target;
    scale::ScaleType = default_config().scale,
    restarts::Int = default_config().restarts,
    iterations::Int = default_config().iterations,
    rng = default_config().rng,
    accumulator::Type{<: AbstractTransformationAccumulator} = default_config().accumulator,
)
    pc_target = PointCloud(target)
    sqscales = annealing_plan(pc_target, scale)

    _register_kc_naive(
        PointCloud(source),
        pc_target,
        sqscales,
        restarts,
        iterations,
        rng,
        accumulator,
    )
end

function _register_kc_naive(
    source::PointCloud{N, TS},
    target::PointCloud{N, TT},
    sqscales,
    restarts,
    iterations,
    rng,
    accumulator_type,
) where {N, TS, TT}
    T = promote_type(TS, TT)
    kc_weights = zeros(T, length(target.points), length(source.points))
    target_kc_weights = similar(kc_weights, length(target.points), 1)
    source_kc_weights = similar(kc_weights, 1, length(source.points))

    accumulator = accumulator_type(transformation_type(Val(N), T))
    init_transformation = simple_transformation(source, target)
    restart = 0
    kc = zero(T)
    while true
        transformation = init_transformation

        for sqscale in sqscales
            kernel = GaussKernel(-inv(2sqscale))
            for iter in 1:iterations
                changed = false

                compute_kc_weights!(
                    kc_weights,
                    source,
                    target,
                    kernel,
                    transformation,
                )
                kc = sum(kc_weights)
                sum!(target_kc_weights, kc_weights)
                sum!(source_kc_weights, kc_weights)

                target_mean = wsum(target.points, target_kc_weights) / kc
                source_mean = wsum(source.points, source_kc_weights) / kc

                covariance = zero(rotation_type(Val(N), T))
                for j in axes(kc_weights, 2)
                    src = source.points[j]
                    src_centered = src - source_mean
                    for i in axes(kc_weights, 1)
                        trg = target.points[i]
                        trg_centered = trg - target_mean
                        w = kc_weights[i, j]
                        covariance += w * trg_centered * src_centered'
                    end
                end

                transformation = transformation_from_moments(
                    covariance,
                    source_mean,
                    target_mean,
                )
            end
        end
        accumulator =
            update(accumulator, TransformationWithCost(-kc, transformation))
        if restart < restarts
            restart += 1
            init_transformation = rand_transformation(rng, source, target)
        else
            break
        end
    end

    result(accumulator, LinearMap(I))
end
