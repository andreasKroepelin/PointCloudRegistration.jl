using Revise
using PointCloudRegistration
using Profile, PProf
using BenchmarkTools
using Cthulhu

X_raw = rand(3, 2_000_000)
@btime PointCloud($X_raw);
@btime PointCloud{3}($X_raw);
X = PointCloud(X_raw)
@btime prepare_target_kc($X, scale = 0.1);
Profile.clear()
@pprof prepare_target_kc(X, scale = 0.1)

X_prepd =
    PointCloudRegistration.prepare_source_cpd(X; regularizer_lengthscale = 0.1)
nonrigid_cpd(
    X_prepd,
    X;
    scale = 0.3,
    outlier_proportion = 0.01,
    regularizer_strength = 1.0,
)

Profile.clear()
@pprof PointCloud(X_raw)

Profile.clear()
@pprof nonrigid_cpd(
    X_prepd,
    X;
    scale = 0.3,
    outlier_proportion = 0.01,
    regularizer_strength = 1.0,
)

@btime rigid_gmc($X, $X)

Profile.clear()
@pprof rigid_gmc(
    X,
    X;
    restarts = 100,
    iterations = 100,
    scale = logrange(1.0, 0.1, length = 5),
)

X, Y = PointCloudRegistration.Assets.load_1ake_A_4ake_A()
pX = prepare_target_kc(X);
@btime rigid_kc($Y, $pX; restarts = RandomRestarts(30), smm = Smm(100));

X, Y = PointCloudRegistration.Assets.load_cats()
Y = rigid_registration(Y, X)(Y)

nonrigid_registration(
    Y,
    X,
    DistancePreserving(;
        max_edge_length = 45,
        sensitivity = 1.4,
        rel_deviation = 1e-4,
        iterations = 2_000,
    ),
)
Profile.clear()
@pprof nonrigid_registration(
    Y,
    X,
    DistancePreserving(
        max_edge_length = 45,
        sensitivity = 1.4,
        rel_deviation = 1e-4,
        iterations = 200_000,
    ),
)
