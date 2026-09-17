using Revise
using PointCloudRegistration
using FileIO
using MeshIO
using GeometryBasics
using StaticArrays
using GLMakie
using LinearAlgebra

# 3D model from https://free3d.com/3d-model/hand-v1--945174.html

hand_mesh = load("./hand.obj")
mfaces = map(faces(hand_mesh)) do f
    NgonFace(reverse(f.data))
end;

mesh(hand_mesh)

mirror(p) = SA[30 - p.x, p.y, p.z]

hand_pc = PointCloud(coordinates(hand_mesh))
hand_pc_thinned = thin_to_grid(hand_pc, 0.5)
hand_pc_mirrored = PointCloud(mirror.(points(hand_pc)), weights(hand_pc))
hand_pc_mirrored_thinned = thin_to_grid(hand_pc_mirrored, 0.5)

let
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    mesh!(ax, hand_mesh)
    hand_mesh_mirrored = GeometryBasics.Mesh(
        GeometryBasics.Point.(points(hand_pc_mirrored)),
        mfaces,
    )
    mesh!(ax, hand_mesh_mirrored)
    fig
end

prep = prepare_target_kernelcorrelation(hand_pc_thinned);

m = rigid_registration(hand_pc_mirrored_thinned, hand_pc_thinned, KernelCorrelationMM(restarts = RandomRestarts(500)), NoFlip(); target_preparation = prep)

let
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    mesh!(ax, hand_mesh)
    hand_mesh_mirrored = GeometryBasics.Mesh(
        GeometryBasics.Point.(points(m(hand_pc_mirrored))),
        mfaces,
    )
    mesh!(ax, hand_mesh_mirrored)
    fig
end

mf = rigid_registration(hand_pc_mirrored_thinned, hand_pc_thinned, KernelCorrelationMM(restarts = RandomRestarts(500)), WithFlip(); target_preparation = prep)

let
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    mesh!(ax, hand_mesh)
    hand_mesh_mirrored = GeometryBasics.Mesh(
        GeometryBasics.Point.(points(mf(hand_pc_mirrored))),
        det(mf.linear) < 0 ? faces(hand_mesh) : mfaces,
    )
    mesh!(ax, hand_mesh_mirrored)
    fig
end
