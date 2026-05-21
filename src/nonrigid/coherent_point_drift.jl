struct CpdPreparedSource{T, T2, T3}
    invgram::Matrix{T}
    sqsigma_displacements::T2
    corr_factor::T3
end

function prepare_source_coherentpointdrift(source; corr_length, expected_displacement)
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
        exp(corr_factor * sqeuclidean(src1, src2))
        for src1 in source.points, src2 in source.points
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

function CpdDisplacement(source::PointCloud, displacements, source_prepd::CpdPreparedSource)
    displacements_invgram = to_matrix(displacements) * source_prepd.invgram
    CpdDisplacement(
        displacements_invgram,
        source.points,
        source_prepd.corr_factor,
    )
end

function (cpd::CpdDisplacement{N})(pc::PointCloud{N}) where {N}
    connecting_gram = [
        exp(cpd.corr_factor * sqeuclidean(p1, p2))
        for p1 in cpd.source_points, p2 in pc.points
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

@kwdef struct CoherentPointDrift{C <: Number, E <: Number, O <: Real}
    corr_length::C
    expected_displacement::E
    outlier_proportion::O = 0
    iterations::Int = 1000
end

function nonrigid_registration(
    source,
    target,
    alg::CoherentPointDrift;
    source_preparation = nothing,
)
    source_pc = PointCloud(source)
    target_pc = PointCloud(target)
    if isnothing(source_preparation)
        source_preparation = prepare_source_coherentpointdrift(
            source_pc;
            alg.corr_length,
            alg.expected_displacement
        )
    end
    _nonrigid_cpd(source_pc, target_pc, source_preparation, alg.outlier_proportion, alg.iterations)
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

    convergence_checker = ConvergenceChecker(sqsigma, 100)
    for iter in 1:iterations
        for j in 1:J
            dsrc = source.points[j] + displacements[j]
            for i in 1:I
                trg = target.points[i]
                R[i, j] = sqeuclidean(dsrc, trg)
            end
        end
        sqsigma = dot(vec(R), vec(C)) / (Z * N)

        convergence_checker, converged = update_and_check(convergence_checker, sqsigma, iter)
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

    return CpdRegistration(source, displacements, source_preparation)
end
