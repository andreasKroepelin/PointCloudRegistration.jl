using Revise
using PointCloudRegistration
using Profile, PProf
using BenchmarkTools

X_raw = rand(3, 2000)
@btime PointCloud($X_raw);
@btime PointCloud{3}($X_raw);
X = PointCloud(X_raw)
@btime prepare_target_kc($X_raw);
Profile.clear()
@pprof prepare_target_kc(X_raw)

X_prepd = PointCloudRegistration.prepare_source_cpd(X; regularizer_lengthscale = .1)
register_cpd(X_prepd, X; scale = .3, outlier_proportion = .01, regularizer_strength = 1.)

Profile.clear()
@pprof register_cpd(X_prepd, X; scale = .3, outlier_proportion = .01, regularizer_strength = 1.)
