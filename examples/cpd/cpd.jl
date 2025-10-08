using Revise
using PointCloudRegistration
using LinearAlgebra
# using BioStructures
using GLMakie

v = normalize(randn(2))
u = [-v[2], v[1]]
coords = v .* range(1, 200; step = 5)'
X = PointCloud(coords)
m = 5
Y = PointCloud(coords .+ m * u)

let
    fig = Figure()
    ax = Axis(fig[1, 1]; autolimitaspect = 1)
    scatter!(ax, X.points)
    scatter!(ax, Y.points)
    fig
end

cpd = register_cpd(
    Y,
    X;
    outlier_proportion = 0.,
    scale = m,
    regularizer_lengthscale = m,
    regularizer_strength = 1e2,
);

my_blues = range(colorant"#0074d900", colorant"#0074d9ff");
let
    fig = Figure()
    ax = Axis(fig[1, 1]; autolimitaspect = 1)
    ax_c = Axis(fig[1, 2]; autolimitaspect = 1)
    scatter!(ax, X.points)
    scatter!(ax, Y.points)
    scatter!(ax, cpd.target_representatives)
    # arrows2d!(ax, Y.points, cpd.displacement)
    arrows2d!(ax, Y.points, cpd.target_representatives .- Y.points)
    heatmap!(ax_c, cpd.correspondences; colormap = my_blues)
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
