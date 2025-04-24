using InteractiveUtils

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

    function KdeComputation(
        points::AbstractVector{<:StaticVector{N, T}},
        grid::Grid{N, T},
        sqsigma::T,
    ) where {N, T}
        grid_idcs = idx_on_grid.(points, (grid,))
        buffer_space = Array{complex(T), N}(undef, size(grid))
        buffer_freq = Array{complex(T), N}(undef, size(grid))
        fft_plan = plan_fft(buffer_space)
        ifft_plan = plan_ifft(buffer_freq)
        new{N, T, typeof(buffer_space), typeof(fft_plan), typeof(ifft_plan)}(
            grid,
            grid_idcs,
            sqsigma,
            buffer_space,
            buffer_freq,
            fft_plan,
            ifft_plan,
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
    ) = kdecomp!
    fill!(buffer_space, zero(eltype(buffer_space)))
    for (idx, weight) in zip(grid_idcs, weights)
        buffer_space[idx] += weight
    end

    mul!(buffer_freq, fft_plan, buffer_space)
    freq_steps = -2pi ./ extent(grid)
    for idx in CartesianIndices(buffer_freq)
        pos = Tuple(idx) .- 1
        pos = min.(pos, size(buffer_freq) .- pos)
        cf = exp(-sqsigma / 2 * sum((freq_steps .* pos) .^ 2))
        buffer_freq[idx] *= cf
    end
    mul!(buffer_space, ifft_plan, buffer_freq)

    # broadcasting leads to enourmous allocations for some reason, so let's use
    # a loop...
    # res .= real.(buffer_space)
    for i in eachindex(res, buffer_space)
        res[i] = real(buffer_space[i])
    end
end

struct AnnealingLevel{
    N,
    M,
    T,
    CT <: AbstractArray{T, M},
    CWT <: AbstractArray{T, N},
}
    grid::Grid{N, T}
    convd_target::CT
    convd_weights_target::CWT

    function AnnealingLevel(grid, convd_target, convd_weights_target)
        N = ndims(convd_weights_target)
        M = ndims(convd_target)
        @assert M == N + 1
        new{
            N,
            M,
            eltype(convd_target),
            typeof(convd_target),
            typeof(convd_weights_target),
        }(
            grid,
            convd_target,
            convd_weights_target,
        )
    end
end

function AnnealingLevel(
    target::PointCloud,
    target_bbox::NTuple{2},
    weighted_target_points,
    sqscale,
)
    target_lo, target_hi = target_bbox
    sigma = sqrt(sqscale)
    grid = Grid(target_lo .- 3sigma, target_hi .+ 3sigma, sigma / 4)
    convd_target = convd_target_type(target)(
        zeros(eltype(target), nrows(target), size(grid)...),
    )
    convd_weights_target = zeros(eltype(target), size(grid)...)
    kde! = KdeComputation(points(target), grid, sqscale)
    kde!(convd_weights_target, weights(target))
    kde!.(each1slice(convd_target), eachrow(weighted_target_points))
    # for (a, b) in zip(each1slice(convd_target), eachrow(weighted_target_points))
    #     kde!(a, b)
    # end
    # display(@code_typed kde!.(each1slice(convd_target), eachrow(weighted_target_points)))
    # println("\n"^5)
    AnnealingLevel(grid, convd_target, convd_weights_target)
end

function compute_annealing_levels(target, sqscales)
    target_bbox = bbox(target)
    weighted_target_points = similar(target.points)
    mul!(weighted_target_points, target.points, Diagonal(target.weights))

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
    for j in eachindex(points(source), weights(source))
        src = points(source)[j]
        w_src = weights(source)[j]
        transformed_src = transformation(src)
        grid_idx = idx_on_grid(transformed_src, al.grid)
        grid_idx in valid_idcs || continue
        convd_w_trg = al.convd_weights_target[grid_idx]
        kc += w_src * convd_w_trg
    end
    kc
end

function convd_target_type(target)
    N = nrows(target)
    HybridArray{Tuple{N, ntuple(_ -> StaticArrays.Dynamic(), Val(N))...}}
end

function each1slice(
    X::HybridArray{Tuple{N, Vararg{StaticArrays.Dynamic(), M}}},
) where {N, M}
    colons = ntuple(_ -> :, Val(M))
    ntuple(i -> view(X, i, colons...), Val(N))
end

function swap_columns(A::SMatrix{N, N}) where {N}
    idcs = ntuple(identity, Val(N)) |> SVector
    @reset idcs[1] = 2
    @reset idcs[2] = 1
    A[:, idcs]
end

function axisalign_target(target, eig)
    axisaligner = eig.vectors'
    if det(axisaligner) < 0
        axisaligner = swap_columns(axisaligner)
    end
    axisaligner, LinearMap(axisaligner)(target)
    # aligned_target_points = similar(target.points)
    # mul!(aligned_target_points, axisaligner, target.points)
    # axisaligner, PointCloud(aligned_target_points, target.weights)
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

function prepare_target(
    target::AbstractMatrix;
    scale,
    axisalign::Bool = true,
    annealing::Int = 5,
)
    _prepare_target(PointCloud(target), scale, axisalign, annealing)
end

prepare_target(prepared_target::PreparedTarget; _kwargs...) = prepared_target

function _prepare_target(target_original, scale, axisalign, annealing)
    if axisalign
        eig = eigen_cov(target_original)
        axis_aligning_rotation, target = axisalign_target(target_original, eig)
        sqscales = annealing_plan(eig, scale, annealing)
    else
        axis_aligning_rotation = one(rotation_type(target, target))
        target = target_original
        sqscales = annealing_plan(target, scale, annealing)
    end
    annealing_levels = compute_annealing_levels(target, sqscales)
    PreparedTarget(axis_aligning_rotation, annealing_levels, target)
end

function register_no_correspondences(
    source,
    target::AbstractMatrix;
    scale::Real,
    axisalign::Bool = true,
    annealing::Int = default_config().annealing,
    restarts::Int = default_config().restarts,
    iterations::Int = default_config().iterations,
    rng = default_config().rng,
)
    prepared_target = prepare_target(target; scale, axisalign, annealing)

    _register_no_correspondences(
        PointCloud(source),
        prepared_target,
        restarts,
        iterations,
        rng,
    )
end

function register_no_correspondences(
    source,
    prepared_target::PreparedTarget;
    restarts::Int = default_config().restarts,
    iterations::Int = default_config().iterations,
    rng = default_config().rng,
)
    _register_no_correspondences(
        PointCloud(source),
        prepared_target,
        restarts,
        iterations,
        rng,
    )
end

function _register_no_correspondences(
    source::PointCloud{N, TS},
    prepared_target::PreparedTarget{N, TT},
    restarts,
    iterations,
    rng,
) where {N, TS, TT}
    T = promote_type(TS, TT)
    (; annealing_levels, target, axis_aligning_rotation) = prepared_target
    source_grid_idcs = zeros(CartesianIndex{N}, size(source, 2))

    best = worst(transformation_type(Val(N), T))
    init_transformation = simple_transformation(source, target)
    restart = 0
    kc = zero(T)
    while true
        transformation = init_transformation

        for annealing_level in annealing_levels
            (; grid, convd_target, convd_weights_target) = annealing_level
            valid_idcs = CartesianIndices(size(grid))

            @logmsg LogLevel(-2000) "mm iteration" restart iter = -1 kc = eval_kernel_correlation(last(annealing_levels), source, transformation) rotation = transformation.linear translation = transformation.translation init_transformation target_kde = copy(convd_weights_target) grid id = :mm

            for iter in 1:iterations
                changed = false
                target_mean = zero(SVector{N, TT})
                source_mean = zero(SVector{N, TS})
                kc = zero(T)
                for j in axes(source, 2)
                    src = points(source)[j]
                    transformed_src = transformation(src)
                    grid_idx = idx_on_grid(transformed_src, grid)
                    if source_grid_idcs[j] != grid_idx
                        changed = true
                    end
                    source_grid_idcs[j] = grid_idx
                    grid_idx in valid_idcs || continue

                    w_src = weights(source)[j]
                    convd_trg = convd_target[:, grid_idx]
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
                for j in axes(source, 2)
                    grid_idx = source_grid_idcs[j]
                    grid_idx in valid_idcs || continue

                    src = points(source)[j]
                    w_src = weights(source)[j]
                    convd_trg = convd_target[:, grid_idx]
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
        best = better(best, TransformationWithCost(-kc, transformation))
        if restart < restarts
            restart += 1
            init_transformation = rand_transformation(rng, source, target)
        else
            break
        end
    end

    LinearMap(axis_aligning_rotation') ∘ best.transformation
end
