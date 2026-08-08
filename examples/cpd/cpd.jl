using Revise
using PointCloudRegistration
using GLMakie

# X, Y = PointCloudRegistration.Assets.load_1ake_A_4ake_A()
# X, Y = PointCloudRegistration.Assets.load_1su4_A_1iwo_A()
X, Y = PointCloudRegistration.Assets.load_1ih7_A_1ig9_A()

Y_thin = thin_to_grid(Y, 5)

T_rigid = rigid_kc(Y_thin, X)
Y_thin_rr = T_rigid(Y_thin)

T_rigid = rigid_kc(Y, X)
Y_rr = T_rigid(Y)

cpd = nonrigid_cpd(Y_thin_rr, X; corr_length = 20, expected_displacement = 10)

cpd = nonrigid_cpd(Y_rr, X; corr_length = 15, expected_displacement = 10)

TY = T_rigid(Y)
dY = cpd(TY)

let
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    # plot!(ax, X)
    # plot!(ax, TY)
    arrows3d!(ax, TY.points, dY.points .- TY.points)
    # plot!(ax, Y_thin_rr)
    # arrows3d!(ax, Y_thin_rr.points, cpd(Y_thin_rr).points .- Y_thin_rr.points)
    fig
end
