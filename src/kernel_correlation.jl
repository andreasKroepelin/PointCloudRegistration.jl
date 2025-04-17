struct Grid{N, T}
    lo::SVector{N, T}
    hi::SVector{N, T}
    Δ::T
    size::NTuple{N, Int}

    Grid(lo::SVector{N}, hi::SVector{N}, Δ) where {N} =
        new{N, promote_type(eltype(lo), eltype(hi), typeof(Δ))}(
            lo,
            hi,
            Δ,
            Tuple(round.(Int, (hi - lo) / Δ)),
        )
end

idx_on_grid(x, grid::Grid{N}) where {N} = idx_on_grid(SVector{N}(x), grid)
idx_on_grid(x::SVector{N}, grid::Grid{N}) where {N} =
    CartesianIndex(round.(Int, (x - grid.lo) / grid.Δ)...)

extent(grid::Grid) = grid.hi - grid.lo
Base.size(grid::Grid) = grid.size
domains(grid::Grid) = range.(grid.lo, grid.hi, grid.size)

struct KdeComputation{
    N,
    T,
    B <: AbstractArray{Complex{T}, N},
    Pl <: FFTW.AbstractFFTs.Plan{Complex{T}},
}
    grid::Grid{N, T}
    grid_idcs::Vector{CartesianIndex{N}}
    sqsigma::T
    buffer_space::B
    buffer_freq::B
    fft_plan::Pl

    function KdeComputation(
        points::AbstractVector{<:StaticVector{N, T}},
        grid::Grid{N, T},
        sqsigma::T,
    ) where {N, T}
        grid_idcs = idx_on_grid.(points, (grid,))
        buffer_space = Array{complex(T), N}(undef, size(grid))
        buffer_freq = Array{complex(T), N}(undef, size(grid))
        fft_plan = plan_fft(buffer_space)
        new{N, T, typeof(buffer_space), typeof(fft_plan)}(
            grid,
            grid_idcs,
            sqsigma,
            buffer_space,
            buffer_freq,
            fft_plan,
        )
    end
end

function (kdecomp!::KdeComputation{N, T})(
    res::AbstractArray{T, N},
    weights::AbstractVector{<:Real},
) where {N, T}
    (; grid, grid_idcs, sqsigma, buffer_space, buffer_freq, fft_plan) = kdecomp!
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
    ldiv!(buffer_space, fft_plan, buffer_freq)

    res .= real.(buffer_space)
end

# function kde!(convd_target, convd_weights_target, target, grid, sqscale, buf)
#     @no_escape buf begin
#         fft_buffer = @alloc(Complex(eltype(target)), size(grid)..., nrows(target) + 1)
#         fill!(fft_buffer, zero(eltype(fft_buffer)))
#         for (trg, trg_w) in zip(points(target), weights(target))
#             idx = idx_on_grid(trg, grid)
#             @view(fft_buffer[idx..., SOneTo(nrows(target))]) .+= trg_w * trg
#             fft_buffer[idx..., end] += trg_w
#         end

#         fft!(fft_buffer; dims = 1:nrows(target))
#         freq_steps = -2pi ./ extent(grid)
#         for idx in CartesianIndices(size(fft_buffer)[SOneTo(nrows(target))])
#             pos = Tuple(idx) .- 1
#             pos = min.(pos, size(fft_buffer) .- pos)
#             cf = exp(-sqscale / 2 * sum((freq_steps .* pos) .^ 2))
#             @view(fft_buffer[idx, :]) .*= cf
#         end
#         ifft!(fft_buffer; dims = 1:nrows(target))

#         N_colons = ntuple(_ -> :, Val(nrows(target)))
#         for l in SOneTo(nrows(target))
#             @view(convd_target[l, N_colons...]) .=
#                 real.(@view(fft_buffer[N_colons..., l]))
#         end
#         convd_weights_target .= real.(@view(fft_buffer[N_colons..., end]))
#     end
# end

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

function swap_columns(A::SMatrix{N, N}) where N
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
    aligned_target_points = similar(target.points)
    mul!(aligned_target_points, axisaligner, target.points)
    axisaligner, WeightedPointCloud(aligned_target_points, target.weights)
end

function register_no_correspondences(
    source,
    target;
    scale::Real,
    axisalign::Bool = true,
    kwargs...,
)
    config = (; default_config()..., kwargs...)
    ws_source = weighted(statically_known_rows(source))
    ws_target = weighted(statically_known_rows(target))

    if axisalign
        eig = eigen_cov(ws_target)
        axisaligner, ws_target = axisalign_target(ws_target, eig)
        sqscales = annealing_plan(eig, scale, config.annealing)
        @logmsg LogLevel(-2000) "align" id=:align rotation=axisaligner
    else
        sqscales = annealing_plan(ws_target, scale, config.annealing)
    end

    transformation = _register_no_correspondences(
        ws_source,
        ws_target,
        sqscales,
        config.restarts,
        config.iterations,
        config.rng,
    )

    if axisalign
        LinearMap(axisaligner') ∘ transformation
    else
        transformation
    end
end

function _register_no_correspondences(
    source,
    target,
    sqscales,
    restarts,
    iterations,
    rng,
)
    target_lo, target_hi = bbox(target)
    source_grid_idcs = zeros(CartesianIndex{nrows(source)}, size(source, 2))
    weighted_target_points = similar(target.points)
    mul!(weighted_target_points, target.points, Diagonal(target.weights))

    best = worst(transformation_type(source, target))
    init_transformation = simple_transformation(source, target)
    restart = 0
    kc = zero(common_eltype(source, target))
    while true
        transformation = init_transformation

        for sqscale in sqscales
            sigma = sqrt(sqscale)
            grid = Grid(target_lo .- 3sigma, target_hi .+ 3sigma, sigma / 4)
            convd_target = convd_target_type(target)(
                zeros(eltype(target), nrows(target), size(grid)...),
            )
            convd_weights_target = zeros(eltype(target), size(grid)...)
            valid_idcs = CartesianIndices(size(grid))
            kde! = KdeComputation(points(target), grid, sqscale)
            kde!(convd_weights_target, weights(target))
            kde!.(each1slice(convd_target), eachrow(weighted_target_points))

            for iter in 1:iterations
                changed = false
                target_mean = zero(SVector{nrows(target), eltype(target)})
                source_mean = zero(SVector{nrows(source), eltype(source)})
                kc = zero(common_eltype(source, target))
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
                covariance = zero(rotation_type(source, target))
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
                @logmsg LogLevel(-2000) "mm iteration" restart iter sqscale rotation = transformation.linear translation = transformation.translation init_transformation target_kde=copy(convd_weights_target) grid id = :mm
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

    best.transformation
end
