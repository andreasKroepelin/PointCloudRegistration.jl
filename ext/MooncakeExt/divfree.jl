module DivFree

using PointCloudRegistration
using StaticArrays
using Distances
using CoordinateTransformations
using Mooncake
using LinearAlgebra
using Statistics

using PointCloudRegistration.Adam


struct Velocity{N, C, Cs <: AbstractVector{C}}
    coefficients::Cs

    Velocity{N}(coefficients) where {N} = new{N, eltype(coefficients), typeof(coefficients)}(coefficients)
end

@generated function velocity_for_frequency(f::SVector{N}, x::SVector{N, T}, c::SVector{M}) where {N, M, T}
    if M != binomial(N, 2)
        error("`c` should have length (N choose 2) where N is length of `x` and `f`.")
    end
    setup = quote
        fpi = f .* T(pi)
        sincos_fpix = sincos.(fpi .* x) .|> NamedTuple{(:sin, :cos)}
        # We manually "unroll" the result vector because we cannot mutate single
        # entries of an SVector later.
        $([:($(Symbol(:v, i)) = zero(T)) for i in 1:N]...)
    end
    compute_scalars = map(1:N) do i
        quote
            $(Symbol(:s, i)) = *(
                fpi[$i],
                sincos_fpix[$i].cos,
                $([:(sincos_fpix[$j].sin) for j in 1:N if j != i]...)
            )
        end
    end
    additions = []
    k = 1
    for i in 1:N
        for j in (i + 1):N
            addition = quote
                $(Symbol(:v, i)) += c[$k] * $(Symbol(:s, j)) 
                $(Symbol(:v, j)) -= c[$k] * $(Symbol(:s, i)) 
            end
            push!(additions, addition)
            k += 1
        end
    end

    return quote
        $setup
        $(compute_scalars...)
        $(additions...)
        SVector($([Symbol(:v, i) for i in 1:N]...))
    end
end

function (velocity::Velocity{N, C})(x) where {N, C}
    (; coefficients) = velocity
    degree = floor(Int, (length(coefficients) / binomial(N, 2))^(1/N))
    frequencies = CartesianIndices(ntuple(_ -> degree, Val(N)))
    coefficients_chunks = reinterpret(SVector{binomial(N, 2), C}, coefficients)
    total = zero(x)
    for (c, f_ci) in zip(coefficients_chunks, frequencies)
        f = SVector(Tuple(f_ci))
        total += velocity_for_frequency(f, x, c)
    end
    return total ./ 2^N
end

function displace(x, velocity)
    timesteps = 20
    h = inv(timesteps)
    for _ in 1:timesteps
        x += h * velocity(x + h/2 * velocity(x))
    end
    return x
end

function huber(r)
    r0 = typeof(r)(0.01)
    absr = abs(r)
    if absr < r0
        r^2 / 2
    else
        r0 * absr - r0^2 / 2
    end
end

function compute_invlambdas(::Type{T}, ::Val{N}, degree::Int) where {T, N}
    single = map(vec(CartesianIndices(ntuple(_ -> degree, Val(N))))) do ci
        T(sqrt(pi^2 * sum(abs2, Tuple(ci)))^N)
    end
    return repeat(single; inner = binomial(N, 2))
end

struct Energy{N, T, PS<:PointCloud{N, T}, PT<:PointCloud{N, T}}
    sigma::T
    invlambdas::Vector{T}
    source::PS
    target::PT
    correspondences::Matrix{T}
end

function Energy(source::PointCloud{N, T}, target::PointCloud{N, T}, sigma, degree) where {N, T}
    invlambdas = compute_invlambdas(T, Val(N), degree)
    correspondences = zeros(T, length(target.points), length(source.points))

    Energy(sigma, invlambdas, source, target, correspondences)
end

function (energy::Energy{N, T})(coefficients::AbstractVector) where {N, T}
    (; sigma, invlambdas, correspondences, source, target) = energy
    velocity = Velocity{N}(coefficients)

    likelihood = zero(T)
    for j in eachindex(source.points)
        dsrc = displace(source.points[j], velocity)
        for i in eachindex(target.points)
            trg = target.points[i]
            likelihood += correspondences[i, j] * huber(euclidean(dsrc, trg))
            # likelihood += correspondences[i, j] * sqeuclidean(dsrc, trg)
        end
    end

    prior = zero(T)
    for (coefficient, invlambda) in zip(coefficients, invlambdas)
        prior += invlambda * coefficient^2
    end

    return sigma^2 / 2 * prior + likelihood
end

struct VelocityOptimizer{E, C, A, G}
    energy::E
    cache::C
    adam::A
    grad::G

    function VelocityOptimizer(energy, coefficients)
        cache = prepare_gradient_cache(energy, coefficients)
        adam = Adam.State(coefficients)
        grad = similar(coefficients)
        new{
            typeof(energy),
            typeof(cache),
            typeof(adam),
            typeof(grad),
        }(
            energy,
            cache,
            adam,
            grad,
        )
    end
end

function run!(vo::VelocityOptimizer, coefficients; iterations = 100)
    Adam.reset!(vo.adam)
    for iter in 1:iterations
        val, grad = value_and_gradient!!(vo.energy, vo.cache, coefficients)
        # norm(vo.grad) / length(vo.grad) < 1e-3 && break
        Adam.step!(vo.adam, grad, coefficients, iter)
    end
end

function unit_box_squisher(pointclouds::PointCloud{N, T}...) where {N, T}
    bboxes = map(pointclouds) do pc
        lo, hi = PointCloudRegistration.bbox(pc)
        SVector(lo, hi)
    end
    all_lo, all_hi = PointCloudRegistration.bbox(reduce(vcat, bboxes))
    center_to_zero = Translation(-(all_lo + all_hi) ./ 2)
    extent = all_hi - all_lo
    scalefactor = 7 / (10 * maximum(extent))
    scaling = LinearMap(scalefactor)
    center_to_half = Translation(@SVector fill(one(T) / 2, N))

    return center_to_half ∘ scaling ∘ center_to_zero
end

struct DivFreeRegistration{S, V, C, PS}
    ubs::S
    invubs::S
    velocity::V
    correspondences::C
    source::PS

    function DivFreeRegistration(ubs, velocity, correspondences, source)
    new{
        typeof(ubs),
        typeof(velocity),
        typeof(correspondences),
        typeof(source),
    }(
        ubs,
        inv(ubs),
        velocity,
        correspondences,
        source,
    )
    end
end

function PointCloudRegistration.correspondences(dfr::DivFreeRegistration)
    return dfr.correspondences
end

function PointCloudRegistration.displacements(dfr::DivFreeRegistration)
    return map(dfr.source.points) do src
        invubs(displace(ubs(src), dfr.velocity))
    end
end

function PointCloudRegistration.nonrigid_divfree(
    source,
    target;
    scale = nothing,
    degree::Int = 3,
)
    source_pc = PointCloud(source)
    target_pc = PointCloud(target)
    if isnothing(scale)
        scale = PointCloudRegistration.avg_nn_dist(target_pc)
    end
    _nonrigid_divfree(source_pc, target_pc, scale, degree)
end

function _nonrigid_divfree(
    source::PointCloud{N},
    target::PointCloud{N},
    scale,
    degree,
) where {N}
    ubs = unit_box_squisher(source, target)
    source_box = ubs(source)
    target_box = ubs(target)
    sigma = eltype(source_box)(LinearMap(ubs.linear)(scale))
    outlier_term = sqrt(2pi * sigma^2) ^ N
    energy = Energy(source_box, target_box, sigma, degree)
    corr_row_sums = similar(energy.correspondences, size(energy.correspondences, 1), 1)
    coefficients = zeros(eltype(source_box), N * degree^N)
    velocity_optimizer = VelocityOptimizer(energy, coefficients)

    for iter in 1:100
        for j in eachindex(source_box.points)
            src = source_box.points[j]
            dsrc = displace(src, Velocity{N}(coefficients)) 
            src_w = source_box.weights[j]
            for i in eachindex(target_box.points)
                trg = target_box.points[i]
                trg_w = target_box.weights[i]
                c = exp(-inv(2sigma^2) * sqeuclidean(dsrc, trg))
                energy.correspondences[i, j] = src_w * trg_w * c
            end
        end
        sum!(corr_row_sums, energy.correspondences)
        energy.correspondences ./= corr_row_sums .+ outlier_term

        run!(velocity_optimizer, coefficients)
    end

    return DivFreeRegistration(
        ubs,
        Velocity{N}(coefficients),
        energy.correspondences,
        source,
    )
end

end
