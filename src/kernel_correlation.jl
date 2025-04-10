function kde!(res, grid, X, ranges, scale)
    fill!(grid, zero(eltype(grid)))
    for x in eachcol(X)
        idx = searchsortedfirst.(ranges, SVector(x), (Base.ForwardOrdering(),))
        grid[idx...] += 1
    end
    fft!(grid)
    freq_steps = -2pi ./ (maximum.(ranges) .- minimum.(ranges))
    for idx in CartesianIndices(grid)
        pos = Tuple(idx) .- 1
        pos = min.(pos, size(grid) .- pos)
        grid[idx] *= exp(-scale / 2 * sum((freq_steps .* pos) .^ 2))
    end
    ifft!(grid)
    res .= real.(grid)
end

function optimize_kernel_correlation(Y, X, scales, init_transformation, buf)
    X_lo, X_hi = bbox(X)
    @no_escape buf begin
        sigma = sqrt(scale)
        ranges = range.(X_lo, X_hi, step = sigma / 4)
        X_kde = @alloc(eltype(X), length.(ranges)...)
        X_grid = @alloc(Complex(eltype(X)), length.(ranges)...)
        kde!(X_kde, X_grid, X, ranges, scale)
    end
end
