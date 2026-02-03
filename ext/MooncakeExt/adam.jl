module Adam

struct State{M}
    v::M
    s::M
    vhat::M
    shat::M

    State(x::M) where {M} = new{M}(similar(x), similar(x), similar(x), similar(x)) |> adam_reset!
end

function step!(adam::State, g::AbstractVector, x::AbstractVector, k::Int)
    (;v, s, vhat, shat) = adam
    T = eltype(x)
    alpha = T(0.001)
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
    fill!(adam.v, zero(eltype(adam.v)))
    fill!(adam.s, zero(eltype(adam.s)))

    return adam
end

end
