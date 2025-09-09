using Revise
using GLMakie
using PointCloudRegistration
using PointCloudRegistration.CoordinateTransformations
using DelimitedFiles
using The2DShapeStructureDataset

X = PointCloud(readdlm("../../assets/ake/1ake.csv", ','))
Y = PointCloud(readdlm("../../assets/ake/4ake.csv", ','))

T = register_gmc(Y, X)

function partial_transformation(transformation::AffineMap{R, T}, t) where {R, T}
    rotation = transformation.linear ^ t |> real |> R
    translation = t * transformation.translation
    AffineMap(rotation, translation)
end

let
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    sl = Slider(fig[2, 1]; range = 0:0.01:1)
    Y_tr = @lift (partial_transformation(T, $(sl.value)))(Y)
    meshscatter!(ax, X; markersize = 2)
    meshscatter!(ax, Y_tr; markersize = 2)
    fig
end

TY = T(Y)
cpd = register_cpd(TY, X, scale = 2., outlier_proportion = 0., regularizer_strength = .1, regularizer_lengthscale = 10.);

heatmap(cpd.correspondences, axis = (; autolimitaspect = 1))

arrows3d(TY.points, cpd.displacement)

let
    fig = Figure()
    ax = Axis3(fig[1, 1]; aspect = :data)
    sl = Slider(fig[2, 1]; range = 0:0.01:1)
    Y_tr = @lift let t = $(sl.value)
        TY.points .+ t .* cpd.displacement
    end
    meshscatter!(ax, X; markersize = 2)
    meshscatter!(ax, Y_tr; markersize = 2)
    fig
end

X = PointCloud(shape_coords("apple-9"))
Y = PointCloud(rand(PointCloudRegistration.Rotations.RotMatrix2) * shape_coords("apple-15") .+ 1)

scatter(X, axis = (; autolimitaspect = 1))

T = register_kc(Y, X)

let
    fig, ax, _ = scatter(X; axis = (; autolimitaspect = 1))
    scatter!(ax, T(Y))
    fig
end

let
    fig = Figure()
    ax = Axis(fig[1, 1]; autolimitaspect = 1)
    sl = Slider(fig[2, 1]; range = 0:0.01:1)
    Y_tr = @lift (partial_transformation(T, $(sl.value)))(Y)
    scatter!(ax, X)
    scatter!(ax, Y_tr)
    fig
end

TY = T(Y)

cpd = register_cpd(TY, X, scale = .01, outlier_proportion = .1, regularizer_strength = .1, regularizer_lengthscale = .1);

heatmap(cpd.correspondences, axis = (; autolimitaspect = 1))

arrows2d(TY.points, cpd.displacement)

let
    fig = Figure()
    ax = Axis(fig[1, 1]; autolimitaspect = 1)
    sl = Slider(fig[2, 1]; range = 0:0.01:1)
    Y_tr = @lift let t = $(sl.value)
        TY.points .+ t .* cpd.displacement
    end
    scatter!(ax, X)
    scatter!(ax, Y_tr)
    arrows2d!(ax, TY.points, cpd.displacement; color = (:gray, .1))
    fig
end
