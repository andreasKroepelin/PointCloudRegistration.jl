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

function axisalign_target(target)
    axisaligner = nearest_rotation(target.coveigvecs')
    axisaligner, LinearMap(axisaligner)(target)
end

struct PreparedTargetKernelCorrelation{
    N,
    R <: RotMatrix{N},
    Al <: AnnealingLevel,
}
    axis_aligning_rotation::R
    annealing_levels::Vector{Al}
end

Base.eltype(::PreparedTargetKernelCorrelation{N, T}) where {N, T} = T

dimension(::PreparedTargetKernelCorrelation{N}) where {N} = N

"""
    prepare_target_kernelcorrelation(target[; scale, axisalign])

Perform all the source independent precomputation for the target that is used in
`rigid_registration(source, target, ::KernelCorrelationMM)`.
This function is especially useful if you plan to register multiple sources to
the same target.

For the meaning of the keyword arguments, see [`KernelCorrelationMM`](@ref).

# Example
```julia
X = rand(3, 100)
Y1 = rand(3, 150)
Y2 = rand(3, 130)

X_prep = prepare_target_kernelcorrelation(X) # this takes some time

T1 = rigid_registration(X, Y1, KernelCorrelationMM(); target_preparation = X_prep) # this is fast
T2 = rigid_registration(X, Y2, KernelCorrelationMM(); target_preparation = X_prep) # this is fast
```
"""
function prepare_target_kernelcorrelation(
    target;
    scale::ScaleType = TargetScales(),
    axisalign::Bool = false,
)
    _prepare_target_kernelcorrelation(PointCloud(target), scale, axisalign)
end

function _prepare_target_kernelcorrelation(target_original, scale, axisalign)
    if axisalign
        axis_aligning_rotation, target = axisalign_target(target_original)
    else
        target = target_original
        axis_aligning_rotation = one(rotation_type(target, target))
    end
    sqscales = annealing_plan(target, scale)
    annealing_levels = compute_annealing_levels(target, sqscales)
    PreparedTargetKernelCorrelation(axis_aligning_rotation, annealing_levels)
end

# We use sigma^2 and not 2sigma^2 in the exponent in the docstring because only
# this way the sigma corresponds to the `scale` parameter.
# The reason is that by muliplying the kernel denisities for the inner product
# we obtain 2sigma^2 in the kernel correlation formula as it is implemented
# here.

"""
    KernelCorrelation([; scale, axisalign, restarts, iterations, batching, report_iteration, report_restart])

Rigid registration of point clouds that does not need any correspondence
information while still having a runtime that depends *linearly* on the point
cloud sizes.

This algorithm maximizes the *kernel correlation* of the target and the
transformed source via majorization minimization.
For a weighted point cloud
``x_1, \\dots, x_I \\in \\mathbb{R}^D`` with weights ``p_1, \\dots, p_I``,
the (Gaussian) kernel density is
```math
\\mu(z) = \\sum_{i = 1}^I p_i \\, \\exp(- \\Vert x_i - z \\Vert^2 / \\sigma^2)
```
The kernel correlation of two weighted point clouds with kernel densities
``\\mu`` and ``\\nu`` is then simply the inner product
``\\langle \\mu, \\nu \\rangle =
\\int_{\\mathbb{R}^D} \\mu(z) \\nu(z) \\mathrm{d} z``.
The scale parameter ``\\sigma`` determines how small/large details are resolved
in the kernel densities.

# Parameters
- `scale`: Determines the value of ``\\sigma`` (see above).
  Can be set to a specific number/collection of numbers or chosen heuristically,
  see Section [Scale parameter](@ref).
  Default: [`TargetScales()`](@ref)
- `axisalign`: Whether or not to temporarily rotate the target to be more axis
  aligned, which can improve performance for very "long" shapes.
  Default: `false`
- `restarts`: Determines how to restart the optimization to avoid local optima,
  see Section [Restarts](@ref).
  Default: `RandomRestarts(50)`
- `iterations`: How many iterations to perform at most, might stop earlier if
  convergence is detected.
  Default: `100`
- `batching`: How to select points from the source in each iteration, see
  Section [Batching](@ref).
  Default: `StochasticBatch(50)`
- `report_iteration`: Callback to run on every iteration.
  Must accept the following keyword arguments:
  - `iter`: Number of the current iteration.
  - `annealing_level`: Current value of ``\\sigma^2``.
  - `cost`: Cost of the current solution candidate.
  - `transformation`: Currently best found transformation.
  Default: `(; kwargs...) -> nothing`
- `report_restart`: Callback to run on every restart.
  Must accept the following keyword arguments:
  - `restart`: Number of the current restart.
  - `cost`: Cost of the optimum found in this restart.
  - `transformation`: Optimal transformation found in this restart.
  Default: `(; kwargs...) -> nothing`
"""
@kwdef struct KernelCorrelationMM{S <: ScaleType, R <: AbstractRestarts, B <: AbstractBatch, RI, RR}
    scale::S = TargetScales()
    axisalign::Bool = false
    restarts::R = RandomRestarts(50)
    iterations::Int = 100
    batching::B = StochasticBatch(50)
    report_iteration::RI = no_report
    report_restart::RR = no_report
end

"""
    rigid_registration(source, target, algorithm::KernelCorrelationMM[; target_preparation])

Perform rigid registration via [`KernelCorrelationMM`](@ref).
See [here](@ref rigid_registration(::Any, ::Any, ::Any)) for general info about
this function.

# Performance
The runtime depends only linearly on the size of `source` and `target`, even
though, conceptually, all pairs of points in both point clouds have to be
considered in every iteration.
To achieve this speed up, certain quantities are precomputed for the target,
independently of the source.
If you are planning to register multiple sources to the same target, it is
advantageous to only do this precomputation once.
This is possible via [`prepare_target_kernelcorrelation`](@ref) and its result
can be given to this function as the `target_preparation` argument.

# Example
For this example, we generate a source point cloud with a sufficiently distinct
shape (somewhat triangluar) and let the target be a more coarsly sampled and
additionally randomly shuffled version of the same shape.
Since the kernel correlation is oblivious to these properties, we still obtain
an identity matrix and a zero vector as rotation and translation, respectively.
```julia
julia> source = stack([t, t * sin(10t)] for t in 0:0.01:10)
2×1001 Matrix{Float64}:
 0.0  0.01         0.02        0.03        0.04       0.05       …   9.94      9.95      9.96      9.97      9.98      9.99     10.0
 0.0  0.000998334  0.00397339  0.00886561  0.0155767  0.0239713     -8.99395  -8.53506  -7.98988  -7.36366  -6.66253  -5.89334  -5.06366

julia> shape_coarser = stack([t, t * sin(10t)] for t in 0:0.02:10)
2×501 Matrix{Float64}:
 0.0  0.02        0.04       0.06       0.08       0.1        …   9.88      9.9       9.92      9.94      9.96      9.98     10.0
 0.0  0.00397339  0.0155767  0.0338785  0.0573885  0.0841471     -9.75354  -9.89215  -9.63607  -8.99395  -7.98988  -6.66253  -5.06366

julia> target = shape_coarser[:, shuffle(axes(shape_coarser, 2))]
2×501 Matrix{Float64}:
 1.3       3.8     2.78      1.84       5.54      9.38     2.1      …   3.6      2.86       6.06     0.02        7.62     5.1       9.98
 0.546217  1.1262  1.26975  -0.799601  -5.05369  -4.06121  1.75698     -3.5704  -0.915028  -4.78342  0.00397339  5.47568  3.41817  -6.66253

julia> transformation = rigid_registration(source, target, KernelCorrelationMM());

julia> transformation.linear
2×2 RotMatrix2{Float64} with indices SOneTo(2)×SOneTo(2):
 1.0         -0.00011193
 0.00011193   1.0

julia> transformation.translation
2-element StaticArraysCore.SVector{2, Float64} with indices SOneTo(2):
 -4.513610792056255e-5
 -0.0042339181606050325
```
"""
function rigid_registration(
    source,
    target,
    alg::KernelCorrelationMM;
    target_preparation::Union{Nothing, PreparedTargetKernelCorrelation} = nothing
)
    @argcheck alg.iterations >= 1

    pc_source = PointCloud(source)
    pc_target = PointCloud(target)
    @argcheck dimension(pc_source) == dimension(pc_target)

    if isnothing(target_preparation)
        target_preparation = prepare_target_kernelcorrelation(pc_target; alg.scale, alg.axisalign)
    end

    _rigid_kc(
        pc_source,
        pc_target,
        target_preparation,
        alg.restarts,
        alg.iterations,
        alg.batching,
        alg.report_iteration,
        alg.report_restart,
    )
end

function _rigid_kc(
    source::PointCloud{N},
    target::PointCloud{N},
    prepared_target::PreparedTargetKernelCorrelation{N},
    restarts,
    iterations,
    batching,
    report_iteration,
    report_restart,
) where {N}
    (; annealing_levels, axis_aligning_rotation) = prepared_target

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
    source_iter = point_cloud_iterator(batching, source)
    for (restart, transformation) in enumerate(restarts_iter)
        for annealing_level in annealing_levels
            (; grid, convd_target, convd_weights_target) = annealing_level
            valid_idcs = CartesianIndices(size(grid))
            prev_transformation = identity_transformation(transformation)

            for iter in 1:iterations
                if 10iter > 9iterations
                    source_iter = non_stochastic(source_iter)
                end
                target_mean = zero(eltype(target.points))
                source_mean = zero(eltype(source.points))
                covariance = target_mean * source_mean'
                kc = zero(CostT)
                for source_element in source_iter
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
        best = better(best, TransformationWithCost(-kc, transformation))
        report_restart(; restart, cost = -kc, transformation)
        @label did_my_best
    end

    return LinearMap(axis_aligning_rotation') ∘ best.transformation
end
