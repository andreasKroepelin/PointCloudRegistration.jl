using Revise
using GLMakie
using PointCloudRegistration
using PointCloudRegistration.CoordinateTransformations
using DelimitedFiles

X = PointCloud(readdlm("../../assets/ake/1ake.csv", ','))
Y = PointCloud(readdlm("../../assets/ake/4ake.csv", ','))

T = register_gmc(Y, X).transformation

function partial_transformation(transformation::AffineMap{R, T}, t) where {R, T}
    rotation = transformation.linear ^ t |> real |> R
    translation = t * transformation.translation
    AffineMap(rotation, translation)
end

let
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    sl = Slider(fig[2, 1]; range = 0:.01:1)
    Y_tr = @lift (partial_transformation(T, $(sl.value)))(Y)
    meshscatter!(ax, X, markersize = 2)
    meshscatter!(ax, Y_tr, markersize = 2)
    fig
end
