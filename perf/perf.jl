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

register_cpd(X, X, .1, .01, 1., .1)

Profile.clear()
@pprof register_cpd(X, X, .1, .01, 1., .1)
