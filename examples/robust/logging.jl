using Logging

struct AnalysisLogger <: AbstractLogger
    logs::Vector
end

Logging.min_enabled_level(::AnalysisLogger) = Logging.BelowMinLevel
Logging.shouldlog(::AnalysisLogger, lvl, mdl, grp, id) = true
function Logging.handle_message(al::AnalysisLogger, lvl, msg, mdl, grp, id, fl, ln; kwargs...)
    push!(al.logs, values(kwargs))
end
