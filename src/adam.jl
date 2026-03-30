module Adam
using PointCloudRegistration: fillzeros!

struct State{L, Q}
    v::L
    s::Q
    vhat::L
    shat::Q

    function State(x::AbstractArray)
        v = similar(x)
        s = v .* v
        vhat = similar(v)
        shat = similar(s)
        new{typeof(v), typeof(s)}(v, s, vhat, shat) |> reset!
    end
end

function step!(adam::State, g::AbstractArray, x::AbstractArray, k::Int)
    (;v, s, vhat, shat) = adam
    T = eltype(x)
    alpha = T(0.01)
    gammav = T(0.9)
    gammas = T(0.999)
    epsilon = T(1e-8)
    @. v = gammav * v + (1 - gammav) * g
    @. s = gammas * s + (1 - gammas) * g * g
    @. vhat = v / (1 - gammav^k)
    @. shat = s / (1 - gammas^k)
    @. x -= alpha * vhat / (sqrt(shat) + epsilon)

    return adam
end

function reset!(adam::State)
    fillzeros!(adam.v)
    fillzeros!(adam.s)

    return adam
end

function isdone(adam::State, tolerance)
    T = eltype(adam.vhat)
    alpha = T(0.01)
    epsilon = T(1e-8)
    for (vhat, shat) in zip(adam.vhat, adam.shat)
        val = alpha * vhat / (sqrt(shat) + epsilon)
        # @show (val, tolerance)
        abs(val) > tolerance && return false
    end
    return true
end

end
