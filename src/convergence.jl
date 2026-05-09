struct ConvergenceChecker{T}
    min_recent::T
    max_recent::T
    period::Int
end

ConvergenceChecker(T::Type{<: Number}, period::Int) = ConvergenceChecker(typemax(T), typemin(T), period)
ConvergenceChecker(::T, period::Int) where {T <: Number} = ConvergenceChecker(T, period)

function update_and_check(cc::ConvergenceChecker{T}, value::T, iteration::Int) where {T}
    if iteration % cc.period == 0
        new_cc = ConvergenceChecker(T, cc.period)
        converged = (cc.max_recent - cc.min_recent) / cc.min_recent < 1e-3
        return new_cc, converged
    else
        min_recent = min(cc.min_recent, value)
        max_recent = max(cc.max_recent, value)
        return ConvergenceChecker(min_recent, max_recent, cc.period), false
    end
end
