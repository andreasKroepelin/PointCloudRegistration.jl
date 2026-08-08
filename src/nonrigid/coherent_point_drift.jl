struct CpdPreparedSource{T, T2, T3}
    invgram::Matrix{T}
    sqsigma_displacements::T2
    corr_factor::T3
end

"""
    prepare_source_coherentpointdrift(source; corr_length, expected_displacement)

Perform all the target independent precomputation for the source that is used in
[`nonrigid_registration(source, target, ::CoherentPointDrift)`](@ref).
This function is especially useful if you plan to register the same source to
multiple targets.

For the meaning of the keyword arguments, see [`CoherentPointDrift`](@ref).

# Example
Say, you have the three point clouds `source`, `target1`, and `target2` where
you want to register `source` to `target1` and `target2`:
```julia
prep = prepare_source_coherentpointdrift(source; corr_length = 13.0, expected_displacement = 42.0)
transformation1 = nonrigid_registration(source, target1, CoherentPointDrift(); source_preparation = prep)
transformation2 = nonrigid_registration(source, target2, CoherentPointDrift(); source_preparation = prep)
```
"""
function prepare_source_coherentpointdrift(
    source;
    corr_length,
    expected_displacement,
)
    source_pc = PointCloud(source)
    _prepare_source_cpd(source_pc, expected_displacement, corr_length)
end

function _prepare_source_cpd(
    source::PointCloud,
    expected_displacement,
    corr_length,
)
    corr_factor = -2 / corr_length^2
    gram = [
        exp(corr_factor * sqeuclidean(src1, src2)) for
        src1 in source.points, src2 in source.points
    ]
    gram_cholesky = cholesky!(Symmetric(gram))
    invgram = LinearAlgebra.inv!(gram_cholesky)
    sqsigma_displacements = expected_displacement^2 / dimension(source)
    CpdPreparedSource(invgram, sqsigma_displacements, corr_factor)
end

struct CpdDisplacement{N, D <: AbstractMatrix, P <: VecOfSVec{N}, F <: Number}
    displacements_invgram::D
    source_points::P
    corr_factor::F
end

function CpdDisplacement(
    source::PointCloud,
    displacements,
    source_prepd::CpdPreparedSource,
)
    displacements_invgram = to_matrix(displacements) * source_prepd.invgram
    CpdDisplacement(
        displacements_invgram,
        source.points,
        source_prepd.corr_factor,
    )
end

function (cpd::CpdDisplacement{N})(pc::PointCloud{N}) where {N}
    connecting_gram = [
        exp(cpd.corr_factor * sqeuclidean(p1, p2)) for
        p1 in cpd.source_points, p2 in pc.points
    ]
    new_points = copy(pc.points)
    # The following call to `mul!` is equivalent to
    # `to_matrix(new_points) += cpd.displacements_invgram * connecting_gram`
    mul!(
        to_matrix(new_points),
        cpd.displacements_invgram,
        connecting_gram,
        true,
        true,
    )

    return PointCloud(new_points, pc.weights)
end

dimension(::CpdDisplacement{N}) where {N} = N

function Base.show(
    io::IO,
    ::MIME"text/plain",
    displacement::CpdDisplacement{N},
) where {N}
    print(io, N, "-dimensional coherent point drift displacement")
end

"""
    CoherentPointDrift(; corr_length, expected_displacement[, outlier_proportion, iterations])

Non-rigid registration of point clouds that assumes that close-by points should
be displaced coherently.

This algorithm alternates between estimating a displacement that would bring the
source close to the target and then smoothing this displacement via Gaussian
Process regression.
Involving Gaussian Processes has the side effect that the resulting
displacements can be applied to other point clouds than the source as well.

# Parameters

* `corr_length`: Determines the range of coherence.
  The smaller this value, the more independently parts of the source can be
  displaced.
* `expected_displacement`: How large (in the sense of vector magnitude) the
  displacement is expected to be on average.
  This acts as a regularization parameter.
  Smaller values mean more regularization.
* `outlier_proportion`: How much of the target point cloud is assumed to be
  outliers, i.e. not produced by displacing the source.
  Must be a number between zero and one.
  Default: zero.
* `iterations`: How many iterations to perform at most, might stop earlier if
  convergence is detected.
  Default: `1000`
"""
@kwdef struct CoherentPointDrift{
    C <: Union{Number, Nothing},
    E <: Union{Number, Nothing},
    O <: Real,
}
    corr_length::C = nothing
    expected_displacement::E = nothing
    outlier_proportion::O = false
    iterations::Int = 1000
end

"""
    nonrigid_registration(source, target, algorithm::CoherentPointDrift[; source_preparation])

Perform non-rigid registration via [`CoherentPointDrift`](@ref).
See [here](@ref nonrigid_registration(::Any, ::Any, ::Any)) for general info
about this function.

This method returns an object of type `CpdDisplacement` (public but not
exported).
It can be applied to arbitrary point clouds (of the same dimension as source and
target).

# Performance
Some of the necessary computation depends only on the source and can thus be
reused for different targets.
To exploit this, use [`prepare_source_coherentpointdrift`](@ref) and provide its
result to the `source_preparation` keyword argument.
In this case, the `corr_length` and `expected_displacement` parameters of
[`CoherentPointDrift`](@ref) do not have to be provided (and are ignored if
provided).

# Example
```julia
julia> # TODO: add example with thinned source
```
"""
function nonrigid_registration(
    source,
    target,
    alg::CoherentPointDrift;
    source_preparation = nothing,
)
    source_pc = PointCloud(source)
    target_pc = PointCloud(target)
    if isnothing(source_preparation)
        @argcheck !isnothing(alg.corr_length) "without `source_preparation`, `corr_length` must be specified"
        @argcheck !isnothing(alg.expected_displacement) "without `source_preparation`, `expected_displacement` must be specified"
        source_preparation = prepare_source_coherentpointdrift(
            source_pc;
            alg.corr_length,
            alg.expected_displacement,
        )
    end
    _nonrigid_cpd(
        source_pc,
        target_pc,
        source_preparation,
        alg.outlier_proportion,
        alg.iterations,
    )
end

function _nonrigid_cpd(
    source::PointCloud{N, TS},
    target::PointCloud{N, TT},
    source_preparation::CpdPreparedSource,
    outlier_p,
    iterations,
) where {N, TS, TT}
    (; invgram, sqsigma_displacements) = source_preparation
    displacements = similar(source.points)
    fillzeros!(displacements)
    I = length(target.points)
    J = length(source.points)
    R = [sqeuclidean(src, trg) for trg in target.points, src in source.points]
    sqsigma = sum(R) / (I * J * N)
    smoothing = similar(invgram)
    outlier_preterm = outlier_p / (1 - outlier_p) / bbox_hypervolume(target)
    C = float.(target.weights .* source.weights')
    c_per_src = sum(C; dims = 1)
    c_per_trg = sum(C; dims = 2)
    Z = sum(c_per_trg)

    convergence_checker = PointsConvergenceChecker(
        displacements,
        10,
        promote_type(TT, TS) |> eps |> sqrt,
    )
    for iter in 1:iterations
        for j in 1:J
            dsrc = source.points[j] + displacements[j]
            for i in 1:I
                trg = target.points[i]
                R[i, j] = sqeuclidean(dsrc, trg)
            end
        end
        sqsigma = dot(vec(R), vec(C)) / (Z * N)

        converged = update_and_check!(convergence_checker, displacements, iter)
        converged && break

        expfactor = inv(-2 * sqsigma)
        @. C = target.weights * source.weights' * exp(expfactor * R)
        sum!(c_per_trg, C)
        outlier_term = outlier_preterm * (2pi * sqsigma)^(N // 2)
        C ./= outlier_term .+ c_per_trg
        c_per_trg ./= outlier_term .+ c_per_trg
        sum!(c_per_src, C)
        Z = sum(c_per_src)

        mul!(to_matrix(displacements), to_matrix(target.points), C)
        displacements .-= vec(c_per_src) .* source.points
        ratio_vars = sqsigma_displacements / sqsigma
        copyto!(smoothing, invgram)
        diagview(smoothing) .+= ratio_vars .* vec(c_per_src)
        cholesky_smoothing = cholesky!(Symmetric(smoothing))
        rdiv!(to_matrix(displacements), cholesky_smoothing)
        lmul!(ratio_vars, to_matrix(displacements))
    end

    return CpdDisplacement(source, displacements, source_preparation)
end
