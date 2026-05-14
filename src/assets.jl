module Assets
using StaticArrays
using PointCloudRegistration

for (X_name, Y_name) in (
    ("1ake_A", "4ake_A"),
    ("1ysy_A", "2ahm_D"),
    ("1su4_A", "1iwo_A"),
    ("1q9x_B", "1q9y_A"),
    ("1ih7_A", "1ig9_A"),
)
    @eval function $(Symbol("load_$(X_name)_$(Y_name)"))()
        map($((X_name, Y_name))) do name
            path = pkgdir(PointCloudRegistration, "assets", "pdbs", "$name.bin")
            points = reinterpret(SVector{3, Float32}, read(path))
            PointCloud(points)
        end
    end
end

function load_cats()
    map((1, 2)) do id
        path = pkgdir(PointCloudRegistration, "assets", "cat-tail")
        points = reinterpret(SVector{2, Float64}, read(joinpath(path, "cat$id-points.bin")))
        weights = reinterpret(Int64, read(joinpath(path, "cat$id-weights.bin")))
        PointCloud(points, weights)
    end
end

end
