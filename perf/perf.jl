using Revise
using PointCloudRegistration
using Profile, PProf
using BenchmarkTools

X_raw = rand(3, 100_000)
@btime PointCloud($X_raw);
@btime PointCloud{3}($X_raw);
X = PointCloud(X_raw)
@btime prepare_target_kc($X_raw);
Profile.clear()
@pprof prepare_target_kc(X_raw)
