using Revise
using PointCloudRegistration
import PointCloudRegistration as PCReg
using LinearAlgebra
# using BioStructures
using GLMakie
using DelimitedFiles

# v = normalize(randn(2))
# u = [-v[2], v[1]]
# coords = v .* range(1, 200; step = 5)'
# m = 5
ake_dir = pkgdir(PointCloudRegistration, "assets", "ake")
X = PointCloud(readdlm(joinpath(ake_dir, "1ake.csv"), ','))
Y = PointCloud(readdlm(joinpath(ake_dir, "4ake.csv"), ','))
resolution = PCReg.avg_nn_dist(X)
T = register_gmc(Y, X; scale = DownTo(resolution))
TY = T(Y)

let
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    meshscatter!(ax, X.points; markersize = resolution)
    meshscatter!(ax, TY.points; markersize = resolution)
    fig
end

cpd = register_cpd(
    TY,
    X;
    outlier_proportion = 0.,
    scale = resolution,
    regularizer_lengthscale = 2resolution,
    regularizer_strength = 1e2,
);

my_blues = range(colorant"#0074d900", colorant"#0074d9ff");
let
    # prepd_source = PCReg.prepare_source_cpd(TY; regularizer_lengthscale = 2resolution)
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    ax_c = Axis(fig[1, 2]; aspect = DataAspect())
    slg = SliderGrid(fig[2, 1:2],
        (label = "log10lambda", range = -5:.1:3, startvalue = 0),
        (label = "scalefactor", range = .2:.05:3, startvalue = 1),
        (label = "regscalefactor", range = 1:.05:50, startvalue = 1),
    )
    params_obs = map(NamedTuple{(:log10lambda, :scalefactor, :regscalefactor)} ∘ tuple, [s.value for s in slg.sliders]...)
    # meshscatter!(ax, X.points; markersize = .4resolution)
    # meshscatter!(ax, TY.points; markersize = .4resolution)
    plt_ttr = linesegments!(ax, vec(permutedims(hcat(X.points, cpd.target_representatives))))
    # plt_trep = meshscatter!(ax, cpd.target_representatives; markersize = .4resolution, color = eachindex(X.points))
    # plt_disp = arrows3d!(ax, TY.points, cpd.displacement)
    # arrows3d!(ax, X.points, cpd.target_representatives .- X.points)
    plt_crsp = heatmap!(ax_c, cpd.correspondences; colormap = my_blues)
    on(params_obs) do params
        (; log10lambda, scalefactor, regscalefactor) = params
        cpd = register_cpd(
            TY,
            X;
            outlier_proportion = 0.,
            scale = scalefactor * resolution,
            regularizer_strength = exp10(log10lambda),
            regularizer_lengthscale = regscalefactor * resolution,
        )
        Makie.update!(plt_ttr, arg1 = vec(permutedims(hcat(X.points, cpd.target_representatives))))
        # Makie.update!(plt_trep, arg1 = cpd.target_representatives)
        # Makie.update!(plt_disp, arg2 = cpd.displacement)
        Makie.update!(plt_crsp, arg1 = cpd.correspondences)
    end
    fig
end

# X = PointCloud(
#     coordarray(retrievepdb("1ake"; dir = tempdir())["A"], calphaselector),
# )
# Y = PointCloud(
#     coordarray(retrievepdb("4ake"; dir = tempdir())["A"], calphaselector),
# )

T = register_kc(Y, X; restarts = 100).best.transformation
YT = T(Y)
V, P = register_cpd(YT, X, 2.0^2, 0.0, 2.0, 1.0)

P_colsums = sum(P; dims = 1)
corr_points = similar(YT.points)
for j in eachindex(YT.points)
    src = YT.points[j]
    coeffs = @view P[:, j]
    s = P_colsums[j]
    corr_points[j] = PointCloudRegistration.wsum(X.points, coeffs) / s
end
YC = PointCloud(corr_points, YT.weights)

heatmap(P)

fig = Figure()
ax = Axis3(fig[1, 1]; aspect = :data)
sl = Slider(fig[2, 1]; range = 0:0.01:1)
# meshscatter!(ax, X.points, markersize = 2.)
# meshscatter!(ax, YT.points, markersize = 2.)
# arrows3d!(ax, YT.points, V)
YTV = @lift YT.points .+ $(sl.value) .* V
meshscatter!(ax, YTV; markersize = 2.0)
# meshscatter!(ax, YC, markersize = 2.)
