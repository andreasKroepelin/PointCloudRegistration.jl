using Revise
using PointCloudRegistration
using BioStructures
using Makie, GLMakie

X = PointCloud(coordarray(retrievepdb("1ake"; dir = tempdir())["A"], calphaselector))
Y = PointCloud(coordarray(retrievepdb("4ake"; dir = tempdir())["A"], calphaselector))

T = register_kc(Y, X, restarts = 100).best.transformation
YT = T(Y)
V, P = register_cpd(YT, X, 2.0^2, 0., 20., 5.0)

fig = Figure()
ax = Axis3(fig[1, 1], aspect = :data)
sl = Slider(fig[2, 1], range = 0:.01:1)
# meshscatter!(ax, X.points, markersize = 2.)
# meshscatter!(ax, YT.points, markersize = 2.)
# arrows3d!(ax, YT.points, V)
YTV = @lift YT.points .+ $(sl.value) .* V
meshscatter!(ax, YTV, markersize = 2.)
