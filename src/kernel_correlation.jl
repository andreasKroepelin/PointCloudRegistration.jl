# For a kernel with standard deviation σ, we want a grid cell width of σ/4 and
# a margin around the point cloud's bounding box of 3σ.
const GRID_CELLS_PER_STD = 4
const GRID_MARGIN_PER_STD = 3

struct Grid{N, T, InvT}
    lo::SVector{N, T}
    hi::SVector{N, T}
    invΔ::InvT
    size::NTuple{N, Int}

    function Grid(lo::SVector{N}, hi::SVector{N}, Δ) where {N}
        @argcheck all(hi .>= lo)
        invΔ = inv(Δ)
        sz = ceil.(Int, (hi - lo) * invΔ)
        # make sure that each grid edge length is actually a multiple of Δ
        adjusted_hi = lo .+ sz .* Δ
        T = promote_type(eltype(lo), eltype(adjusted_hi))
        InvT = typeof(invΔ)
        new{N, T, InvT}(lo, adjusted_hi, invΔ, Tuple(sz) .+ 1)
    end
end

idx_on_grid(x, grid::Grid{N}) where {N} = idx_on_grid(SVector{N}(x), grid)
idx_on_grid(x::SVector{N}, grid::Grid{N}) where {N} =
    CartesianIndex((round.(Int, (x - grid.lo) * grid.invΔ) .+ 1)...)

extent(grid::Grid) = grid.hi - grid.lo
Base.size(grid::Grid) = grid.size
domains(grid::Grid) = range.(grid.lo, grid.hi, grid.size)

function _compute_gaussians(::Type{T}) where {T <: Real}
    steps = 1:(GRID_CELLS_PER_STD * GRID_MARGIN_PER_STD)
    half = exp.(steps .^ 2 ./ (-2 * GRID_CELLS_PER_STD ^ 2)) .|> T
    [reverse(half); one(T); half]
end

const GAUSSIANS_F64 = _compute_gaussians(Float64)
const GAUSSIANS_F32 = _compute_gaussians(Float32)

compute_gaussians(::Type{T}) where {T <: Real} = _compute_gaussians(T)
compute_gaussians(::Type{Float64}) = GAUSSIANS_F64
compute_gaussians(::Type{Float32}) = GAUSSIANS_F32

struct SetsOfSlices{N, Tpl <: Tuple}
    sets::Tpl

    function SetsOfSlices(sets::Tuple)
        Tpl = typeof(sets)
        N = length(Tpl.parameters)
        new{N, Tpl}(sets)
    end
end

function _replace_tail_with_colons(tpl::Tuple, ::Val{M}) where {M}
    N = length(tpl) - M
    ntuple(Val(length(tpl))) do i
        if i <= N
            tpl[i]
        else
            (:)
        end
    end
end

function _make_sets(::Val{N}) where {N}
    ntuple(Val(N)) do M
        repr_tpl = ntuple(_ -> 1, Val(N))
        T = typeof(_replace_tail_with_colons(repr_tpl, Val(M)))
        Set{T}()
    end
end

function SetsOfSlices(idcs::Vector{CartesianIndex{N}}) where {N}
    sets = _make_sets(Val(N))
    Ms = ntuple(Val, Val(N))

    for idx in idcs
        map(sets, Ms) do set, M
            push!(set, _replace_tail_with_colons(Tuple(idx), M))
        end
    end

    SetsOfSlices(map(collect, sets))
end

struct KdeComputation{
    N,
    T <: Number,
    G <: Grid{N},
    SOS <: SetsOfSlices{N},
    Gau <: Vector{<: Number},
}
    grid::G
    grid_idcs::Vector{CartesianIndex{N}}
    sets_of_slices::SOS
    buffer1::Array{T, N}
    buffer2::Array{T, N}
    gaussian::Gau

    function KdeComputation(
        points::VecOfSVec{N, TU},
        grid::Grid{N, TU},
        T::Type{<: Number},
    ) where {N, TU}
        grid_idcs = idx_on_grid.(points, (grid,))
        sets_of_slices = SetsOfSlices(grid_idcs)
        gaussian = compute_gaussians(typeof(one(T)))
        new{N, T, typeof(grid), typeof(sets_of_slices), typeof(gaussian)}(
            grid,
            grid_idcs,
            sets_of_slices,
            Array{T, N}(undef, size(grid)),
            Array{T, N}(undef, size(grid)),
            gaussian,
        )
    end
end

function convolve!(convd, data, filter)
    idcs = eachindex(convd, data)
    step(idcs) == 1 || error("convolve! assumes 1-step indices")
    center = (firstindex(filter) + lastindex(filter)) ÷ 2
    below = center - firstindex(filter)
    above = lastindex(filter) - center
    for i in idcs
        lo_data = max(first(idcs), i - below)
        hi_data = min(last(idcs), i + above)
        lo_filter = center - (i - lo_data)
        hi_filter = center + (hi_data - i)

        c = zero(eltype(convd))
        for (j, k) in zip(lo_data:hi_data, lo_filter:hi_filter)
            @inbounds c += data[j] * filter[k]
        end
        @inbounds convd[i] = c
    end
end

function convolve_all_along_first_dim!(convd, data, filter)
    CartesianIndices(convd) == CartesianIndices(data) ||
        error("buffers have different indices")
    for tail_idx in tail_idcs(CartesianIndices(data))
        idx = (:, Tuple(tail_idx)...)
        convd_1dim = view(convd, idx...)
        data_1dim = view(data, idx...)
        convolve!(convd_1dim, data_1dim, filter)
    end
end

tail_idcs(ci::CartesianIndices) = CartesianIndices(Base.tail(ci.indices))

@generated function (kde!::KdeComputation{N, T})(
    res::AbstractArray{T, N},
    weights::AbstractVector{<:Number},
) where {N, T}
    init = quote
        (; grid, grid_idcs, sets_of_slices, buffer1, buffer2, gaussian) = kde!
        fillzeros!(buffer1)
        fillzeros!(buffer2)
        for (idx, weight) in zip(grid_idcs, weights)
            buffer1[idx] += weight
        end
    end

    directions = [
        quote
            for slice_idcs in sets_of_slices.sets[$i]
                convolve_all_along_first_dim!(
                    view(buffer2, slice_idcs...),
                    view(buffer1, slice_idcs...),
                    gaussian,
                )
            end
            buffer1, buffer2 = buffer2, buffer1
        end for i in 1:N
    ]

    quote
        $init
        $(directions...)
        copyto!(res, buffer1)
    end
end

struct AnnealingLevel{
    N,
    G <: Grid{N},
    CT <: AbstractArray{<:SVector{N, <: Number}, N},
    CWT <: AbstractArray{<: Number, N},
}
    grid::G
    convd_target::CT
    convd_weights_target::CWT

    function AnnealingLevel(
        grid::Grid{N},
        convd_target,
        convd_weights_target,
    ) where {N}
        new{N, typeof(grid), typeof(convd_target), typeof(convd_weights_target)}(
            grid,
            convd_target,
            convd_weights_target,
        )
    end
end

function kde_grid(lo, hi; sigma)
    grid_lo = lo .- GRID_MARGIN_PER_STD * sigma
    grid_hi = hi .+ GRID_MARGIN_PER_STD * sigma
    Grid(grid_lo, grid_hi, sigma / GRID_CELLS_PER_STD)
end

function AnnealingLevel(
    target::PointCloud{N},
    target_bbox::NTuple{2},
    weighted_target_points,
    sqscale,
) where {N}
    grid = kde_grid(target_bbox...; sigma = sqrt(sqscale))
    convd_target = reinterpret(
        reshape,
        SVector{N, eltype(target)},
        zeros(eltype(target), N, size(grid)...),
    )
    convd_weights_target = zeros(typeof(one(eltype(target))), size(grid)...)
    kde_convd_weights_target! =
        KdeComputation(target.points, grid, eltype(convd_weights_target))
    kde_convd_target! =
        KdeComputation(target.points, grid, eltype(parent(convd_target)))

    kde_convd_weights_target!(convd_weights_target, target.weights)
    kde_convd_target!.(
        eachslice(parent(convd_target); dims = 1),
        eachrow(
            reinterpret(
                reshape,
                eltype(eltype(weighted_target_points)),
                weighted_target_points,
            ),
        ),
    )
    AnnealingLevel(grid, convd_target, convd_weights_target)
end

function compute_annealing_levels(target, sqscales)
    target_bbox = bbox(target)
    weighted_target_points = target.points .* target.weights

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
    kc = zero(eltype(source.weights)) * zero(eltype(al.convd_weights_target))
    for j in eachindex(source.points, source.weights)
        src = source.points[j]
        w_src = source.weights[j]
        transformed_src = transformation(src)
        grid_idx = idx_on_grid(transformed_src, al.grid)
        grid_idx in valid_idcs || continue
        convd_w_trg = al.convd_weights_target[grid_idx]
        kc += w_src * convd_w_trg
    end
    kc
end

default_axisalign() = false

function axisalign_target(target)
    axisaligner = nearest_rotation(target.coveigvecs')
    axisaligner, LinearMap(axisaligner)(target)
end

struct PreparedTarget{
    N,
    T,
    R <: RotMatrix{N},
    Al <: AnnealingLevel,
    PC <: PointCloud{N, T},
}
    axis_aligning_rotation::R
    annealing_levels::Vector{Al}
    target::PC
end

Base.eltype(::PreparedTarget{N, T}) where {N, T} = T

dimension(::PreparedTarget{N}) where {N} = N

"""
    prepare_target_kc(target[; scale, axisalign])

Perform all the necessary precomputation for `rigid_kc`.
This function is especially useful if you plan to register multiple sources to
the same target.

```julia
X = PointCloud(rand(3, 100))
Y1 = PointCloud(rand(3, 150))
Y2 = PointCloud(rand(3, 130))

prepd_X = prepare_target_kc(X) # this takes some time

T1 = rigid_kc(prepd_X, Y1) # this is fast
T2 = rigid_kc(prepd_X, Y2) # this is fast
```
"""
function prepare_target_kc(
    target;
    scale::ScaleType = default_scale(),
    axisalign::Bool = default_axisalign(),
)
    _prepare_target(PointCloud(target), scale, axisalign)
end

function _prepare_target(target_original, scale, axisalign)
    if axisalign
        axis_aligning_rotation, target = axisalign_target(target_original)
    else
        target = target_original
        axis_aligning_rotation = one(rotation_type(target, target))
    end
    sqscales = annealing_plan(target, scale)
    annealing_levels = compute_annealing_levels(target, sqscales)
    PreparedTarget(axis_aligning_rotation, annealing_levels, target)
end

"""
    rigid_kc(source, target[; scale, axisalign, restarts, iterations, rng, accumulator])

$REGISTER_DOCS_START
maximizes the Kernel Correlation to `target`,
i.e.
```math
\\sum_{i = 1}^n \\sum_{j = 1}^m \\exp\\left(- \\frac{\\Vert R y_i + t - x_i \\Vert}{2 \\sigma^2}\\right)
```
$REGISTER_DOCS_SYMBOLS

$REGISTER_DOCS_TYPES

The bandwidth parameter ``\\sigma`` corresponds to the `scale` keyword argument
and you can learn about how to use the keyword arguments in
[this section](#Common-keyword-arguments).

Use this function if you do not know correspondences.
"""
function rigid_kc(
    source,
    target;
    scale::ScaleType = default_scale(),
    axisalign::Bool = default_axisalign(),
    restarts::AbstractRestarts = RandomRestarts(50),
    iterations::Int = 100,
    smm = Smm(50),
    report_iteration::RI = no_report,
    report_restart::RR = no_report,
) where {RI, RR}
    @argcheck iterations >= 1

    pc_source = PointCloud(source)
    pc_target = PointCloud(target)
    @argcheck dimension(pc_source) == dimension(pc_target)

    prepared_target = prepare_target_kc(pc_target; scale, axisalign)

    _rigid_kc(
        pc_source,
        prepared_target,
        restarts,
        iterations,
        smm,
        report_iteration,
        report_restart,
    )
end

"""
    rigid_kc(source, prepared_target[; restarts, iterations, rng, accumulator])

$REGISTER_DOCS_START
maximizes the Kernel Correlation to `prepared_target` which was computed by
[`prepare_target_kc`](@ref).

Use this function if you plan to register multiple sources to the same target.
"""
function rigid_kc(
    source,
    prepared_target::PreparedTarget;
    restarts::AbstractRestarts = RandomRestarts(50),
    iterations::Int = 100,
    smm = Smm(50),
    report_iteration::RI = no_report,
    report_restart::RR = no_report,
) where {RI, RR}
    @argcheck iterations >= 1

    pc_source = PointCloud(source)
    @argcheck dimension(pc_source) == dimension(prepared_target)

    _rigid_kc(
        pc_source,
        prepared_target,
        restarts,
        iterations,
        smm,
        report_iteration,
        report_restart,
    )
end

function _rigid_kc(
    source::PointCloud{N},
    prepared_target::PreparedTarget{N},
    restarts,
    iterations,
    smm,
    report_iteration,
    report_restart,
) where {N}
    (; annealing_levels, target, axis_aligning_rotation) = prepared_target

    SrcT = eltype(source.points)
    TrgT = eltype(target.points)
    SrcWT = eltype(source.weights)
    TrgWT = eltype(target.weights)
    CostT = typeof(
        zero(SrcWT) *
        zero(eltype(first(annealing_levels).convd_weights_target)),
    )
    best = worst(CostT, transformation_type(source, target))
    kc = zero(CostT)
    restarts_iter = restarts_iterator(source, target, restarts)
    source_iter = smm_iterator(smm, source)
    for (restart, transformation) in enumerate(restarts_iter)
        for annealing_level in annealing_levels
            (; grid, convd_target, convd_weights_target) = annealing_level
            valid_idcs = CartesianIndices(size(grid))
            prev_transformation = identity_transformation(transformation)

            for this_source_iter in (source_iter, non_stochastic(source_iter))
                for iter in 1:iterations
                    target_mean = zero(eltype(target.points))
                    source_mean = zero(eltype(source.points))
                    covariance = target_mean * source_mean'
                    kc = zero(CostT)
                    for source_element in this_source_iter
                        src = source_element.point
                        transformed_src = transformation(src)
                        grid_idx = idx_on_grid(transformed_src, grid)
                        grid_idx in valid_idcs || continue

                        w_src = source_element.weight
                        convd_trg = convd_target[grid_idx]
                        convd_w_trg = convd_weights_target[grid_idx]
                        target_mean += w_src * convd_trg
                        source_mean += w_src * convd_w_trg * src
                        covariance += w_src * convd_trg * src'
                        kc += w_src * convd_w_trg
                    end

                    if iszero(kc)
                        # As far as we can tell, the two point clouds do not overlap
                        # at all with the current transformation so there is nothing
                        # we can do here.
                        # It is best to simply try another restart.
                        @goto did_my_best
                    end

                    target_mean /= kc
                    source_mean /= kc
                    covariance /= kc
                    covariance -= target_mean * source_mean'

                    transformation = transformation_from_moments(
                        covariance,
                        source_mean,
                        target_mean,
                    )

                    report_iteration(;
                        iter,
                        annealing_level,
                        cost = -kc,
                        transformation,
                    )

                    if iter > 1 && isapprox(transformation, prev_transformation)
                        break
                    end
                    prev_transformation = transformation
                end
            end
        end
        best = better(best, TransformationWithCost(-kc, transformation))
        report_restart(; restart, cost = -kc, transformation)
        @label did_my_best
    end

    return LinearMap(axis_aligning_rotation') ∘ best.transformation
end
