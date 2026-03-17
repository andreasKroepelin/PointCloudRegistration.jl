struct CpdPreparedSource{T, T2, PC <: PointCloud}
    source::PC
    invgram::Matrix{T}
    sqsigma_displacements::T2
end

function prepare_source_cpd(source; corr_length, expected_displacement)
    source_pc = PointCloud(source)
    _prepare_source_bcpd(source_pc, expected_displacement, corr_length)
end

function _prepare_source_bcpd(
    source::PointCloud,
    expected_displacement,
    corr_length,
)
    factor = -2 / corr_length^2
    gram = [
        exp(factor * sqeuclidean(src1, src2))
        for src1 in source.points, src2 in source.points
    ]
    gram_cholesky = cholesky!(Symmetric(gram))
    invgram = LinearAlgebra.inv!(gram_cholesky)
    sqsigma_displacements = expected_displacement^2 / dimension(source)
    CpdPreparedSource(source, invgram, sqsigma_displacements)
end

struct BcpdRegistration{C, D}
    correspondences::C
    displacements::D
end

displacements(bcpd::BcpdRegistration) = bcpd.displacements

correspondences(bcpd::BcpdRegistration) = bcpd.correspondences

function (bcpd::BcpdRegistration)(pc::PointCloud)
end

function register_cpd(
    source,
    target;
    corr_length,
    expected_displacement,
    outlier_proportion = 0
)
    source_prepd = prepare_source_cpd(source; corr_length, expected_displacement)
    target_pc = PointCloud(target)
    _register_cpd(source_prepd, target, outlier_proportion)
end

function register_cpd(
    source_prepd::CpdPreparedSource,
    target;
    outlier_proportion = 0
)
    target_pc = PointCloud(target)
    _register_cpd(source_prepd, target, outlier_proportion)
end

function _register_cpd(
    prepd_source::CpdPreparedSource{T, <: PointCloud{N, TS}},
    target::PointCloud{N, TT},
    outlier_p,
) where {T, N, TS, TT}
    (; source, invgram, sqsigma_displacements) = prepd_source
    displacements = similar(source.points)
    fillzeros!(displacements)
    I = length(target.points)
    J = length(source.points)
    R = [sqeuclidean(src, trg) for trg in target.points, src in source.points]
    sqsigma = sum(R) / (I * J * N)
    prev_sqsigma = typemax(sqsigma)
    smoothing = similar(invgram)
    outlier_preterm = outlier_p / (1 - outlier_p) / bbox_hypervolume(target)
    C = float.(target.weights .* source.weights')
    c_per_src = sum(C; dims = 1)
    c_per_trg = sum(C; dims = 2)
    Z = sum(c_per_trg)

    convergence_counter = 0
    for iter in 1:1000
        @info "iteration" iter sqrt(sqsigma) convergence_counter maximum(norm, displacements)
        for j in 1:J
            dsrc = source.points[j] + displacements[j]
            for i in 1:I
                trg = target.points[i]
                R[i, j] = sqeuclidean(dsrc, trg)
            end
        end
        sqsigma = dot(vec(R), vec(C)) / (Z * N)
        if abs(1 - sqrt(sqsigma / prev_sqsigma)) < 1e-5
            convergence_counter += 1
        else
            convergence_counter = 0
        end
        if convergence_counter > 10
            break
        end
        prev_sqsigma = sqsigma
        expfactor = inv(-2 * sqsigma)
        @. C = target.weights * source.weights' * exp(expfactor * R)
        sum!(c_per_trg, C)
        outlier_term = outlier_preterm * (2pi * sqsigma)^(N // 2)
        C ./= outlier_term .+ c_per_trg
        c_per_trg ./= outlier_term .+ c_per_trg
        sum!(c_per_src, C)
        vec_e = vec(e)
        Z = sum(c_per_src)

        mul!(to_matrix(displacements), to_matrix(target.points), C)
        displacements .-= vec(c_per_src) .* source.points
        ratio_vars = sqsigma_displacement / sqsigma
        copyto!(smoothing, invgram)
        diagview(smoothing) .+= ratio_vars .* c_per_src
        cholesky_smoothing = cholesky!(Symmetric(smoothing))
        rdiv!(to_matrix(displacements), cholesky_smoothing)
        lmul!(ratio_vars, to_matrix(displacements))
    end

    return CpdRegistration(C, displacements)
end
