using Revise
using PointCloudRegistration
using Profile, PProf
using BenchmarkTools
using Cthulhu

X_raw = rand(3, 2000)
@btime PointCloud($X_raw);
@btime PointCloud{3}($X_raw);
X = PointCloud(X_raw)
@btime prepare_target_kc($X, scale = 0.1);
Profile.clear()
@pprof prepare_target_kc(X, scale = 0.1)

X_prepd =
    PointCloudRegistration.prepare_source_cpd(X; regularizer_lengthscale = 0.1)
register_cpd(
    X_prepd,
    X;
    scale = 0.3,
    outlier_proportion = 0.01,
    regularizer_strength = 1.0,
)

Profile.clear()
@pprof register_cpd(
    X_prepd,
    X;
    scale = 0.3,
    outlier_proportion = 0.01,
    regularizer_strength = 1.0,
)

@btime register_gmc($X, $X)

Profile.clear()
@pprof register_gmc(
    X,
    X;
    restarts = 100,
    iterations = 100,
    scale = logrange(1.0, 0.1, length = 5),
)
