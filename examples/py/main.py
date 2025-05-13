from juliacall import Main as jl
import cloudy as cl
import numpy as np


def main():
    print("Hello from py!")
    X = np.cumsum(np.random.randn(500, 3) + .1, axis=0)
    Y = X + 1
    target = cl.PointCloud(X)
    source = cl.PointCloud(Y)
    sigma = 1.
    with cl.take_time('cloudy prep'):
        kc = cl.GriddedKernelCorrelation(target, source, sigma)
        mm = cl.GriddedMM(kc, cl.Pose(np.eye(3), -np.random.rand(3)))
    with cl.take_time('cloudy run'):
        transformation = mm.run(100)
    print(transformation.R)
    print(transformation.t)
    jl.seval("using PointCloudRegistration")
    # for precompilation:
    jl.register_kc(Y.T, X.T, scale=sigma, annealing=0)
    with cl.take_time('PCR prep'):
        prepd_X = jl.prepare_target_kc(X.T, scale=sigma, annealing=0, axisalign=False)
    with cl.take_time('PCR run'):
        transformation = jl.register_kc(Y.T, prepd_X, restarts=0)
    transformation.linear._jl_display()
    transformation.translation._jl_display()
    with cl.take_time('PCR kabsch'):
        transformation = jl.register_rmsd(Y.T, X.T)
    transformation.linear._jl_display()
    transformation.translation._jl_display()


if __name__ == "__main__":
    main()
