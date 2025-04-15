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
            round.(Int, (hi - lo) / Δ),
        )
end

idx_on_grid(x, grid::Grid{N}) where {N} = idx_on_grid(SVector{N}(x), grid)
idx_on_grid(x::SVector{N}, grid::Grid{N}) where {N} =
    round.(Int, (x - grid.lo) / grid.Δ)

extent(grid::Grid) = grid.hi - grid.lo
Base.size(grid::Grid) = grid.size

function kde!(convd_X, convd_weights_X, X, weights_X, grid, scale, buf)
    @no_escape buf begin
        fft_buffer = @alloc(Complex(eltype(X)), size(grid)..., NRows(X) + 1)
        fill!(fft_buffer, zero(eltype(fft_buffer)))
        for (x, wx) in zip(eachcol(X), weights_X)
            idx = idx_on_grid(x, grid)
            @view(fft_buffer[idx..., SOneTo(NRows(X))]) .+= wx * SVector(x)
            fft_buffer[idx..., end] += wx
        end

        fft!(fft_buffer; dims = 1:NRows(X))
        freq_steps = -2pi ./ extent(grid)
        for idx in CartesianIndices(size(fft_buffer)[SOneTo(Nrows(X))])
            pos = Tuple(idx) .- 1
            pos = min.(pos, size(fft_buffer) .- pos)
            cf = exp(-scale / 2 * sum((freq_steps .* pos) .^ 2))
            @view(fft_buffer[idx, :]) .*= cf
        end
        ifft!(fft_buffer; dims = 1:NRows(X))

        N_colons = ntuple(_ -> :, Val(NRows(X)))
        for l in SOneTo(NRows(X))
            @view(convd_X[l, N_colons...]) .=
                real.(@view(fft_buffer[N_colons..., l]))
        end
        convd_weights_X .= real.(@view(fft_buffer[N_colons..., end]))
    end
end

function optimize_kernel_correlation(;
    Y,
    X,
    weights_Y,
    weights_X,
    scales,
    init_transformation,
    buf,
)
    X_lo, X_hi = bbox(X)
    N_Dynamics = ntuple(_ -> StaticArrays.Dynamic(), Val(NRows(X)))
    @no_escape buf begin
        y_grid_idcs = @alloc(CartesianIndex{NRows(X)}, size(Y, 2))
        for scale in scales
            sigma = sqrt(scale)
            grid = Grid(X_lo - 3sigma, X_hi + 3sigma, sigma / 4)
            @no_escape buf begin
                convd_X = HybridArray{Tuple{NRows(X), N_Dynamics...}}(
                    @alloc(eltype(X), NRows(X), size(grid)...)
                )
                convd_weights_X = @alloc(eltype(X), size(grid)...)
                kde!(convd_X, convd_weights_X, X, weights_X, grid, scale)

                for iter in 1:niterations
                    changed = false
                    x_mean = zero(SVector{NRows(X), eltype(X)})
                    y_mean = zero(SVector{NRows(Y), eltype(Y)})
                    kc = zero(eltype(x_mean))
                    for j in axes(Y, 2)
                        y = Y[:, j]
                        transformed_y = rotation * y + translation
                        grid_idx =
                            CartesianIndex(idx_on_grid(transformed_y, grid)...)
                        if y_grid_idcs[j] != grid_idx
                            changed = true
                        end
                        y_grid_idcs[j] = grid_idx
                        if !(grid_idx in CartesianIndices(convd_weights_X))
                            continue
                        end
                        wy = weights_Y[j]
                        convd_x = convd_X[:, grid_idx]
                        convd_wx = convd_weights_X[grid_idx]
                        x_mean += wy * convd_x
                        y_mean += wy * convd_wx * y
                        kc += wy * convd_wx
                    end
                    if !changed && iter > 1
                        break
                    end
                    x_mean /= kc
                    y_mean /= kc
                    cov = zero(rotation)
                    for j in axes(Y, 2)
                        y = Y[:, j]
                        wy = weights_Y[j]
                        grid_idx = y_grid_idcs[j]
                        convd_x = convd_X[:, grid_idx]
                        convd_wx = convd_weights_X[grid_idx]
                        cov +=
                            wy * (convd_x - convd_wx * x_mean) * (y - y_mean)'
                    end
                    rotation = rot_from_cov(cov)
                    translation = x_mean - rotation * y_mean
                end
            end
        end
    end
end
