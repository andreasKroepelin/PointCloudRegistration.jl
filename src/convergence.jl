struct ConvergenceChecker{T}
    min_recent::T
    max_recent::T
    period::Int
    threshold::T
end

ConvergenceChecker(period::Int, threshold::T) where {T <: Number} = ConvergenceChecker(typemax(T), typemin(T), period, threshold)

function update_and_check(cc::ConvergenceChecker{T}, value::T, iteration::Int) where {T}
    (; min_recent, max_recent, period, threshold) = cc
    if iteration % period == 0
        converged = max_recent - min_recent < threshold
        @reset cc.min_recent = typemax(T)
        @reset cc.max_recent = typemin(T)
    else
        converged = false
        @reset cc.min_recent = min(min_recent, value)
        @reset cc.max_recent = max(max_recent, value)
    end
    return cc, converged
end

struct PointsConvergenceChecker{N, T, B <: AbstractVector{NTuple{2, SVector{N, T}}}}
    bboxes::B
    period::Int
    threshold::T

    function PointsConvergenceChecker(points::VecOfSVec{N, T}, period::Int, threshold) where {N, T}
        lo = @SVector fill(typemax(T), N)
        hi = @SVector fill(typemin(T), N)
        bboxes = fill((lo, hi), length(points))
        return new{N, T, typeof(bboxes)}(bboxes, period, threshold)
    end
end

function update_and_check!(pcc::PointsConvergenceChecker{N, T}, points::VecOfSVec{N, T}, iteration::Int) where {N, T}
    (; bboxes, period, threshold) = pcc
    if iteration % period == 0
        movement = maximum(splat(euclidean), bboxes)
        # @info "convergence check" iteration movement threshold
        converged = movement < threshold
        lo = @SVector fill(typemax(T), N)
        hi = @SVector fill(typemin(T), N)
        fill!(bboxes, (lo, hi))
    else
        converged = false
        for i in eachindex(bboxes, points)
            lo, hi = bboxes[i]
            point = points[i]
            lo = min.(lo, point)
            hi = max.(hi, point)
            bboxes[i] = (lo, hi)
        end
    end
    return converged
end
