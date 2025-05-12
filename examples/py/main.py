from juliacall import Main as jl
import cloudy as cl
import numpy as np


def main():
    print("Hello from py!")
    X = np.cumsum(np.random.randn(1000, 3) + .1, axis=0)
    Y = X + 1
    target = cl.PointCloud(X)
    source = cl.PointCloud(Y)
    sigma = 1.
    with cl.take_time('cloudy impl'):
        kc = cl.GriddedKernelCorrelation(target, source, sigma)
        mm = cl.GriddedMM(kc, cl.Pose(np.eye(3), -np.random.rand(3)))
        transformation = mm.run(100)
    print(transformation.R)
    print(transformation.t)
    jl.seval("using PointCloudRegistration")
    transformation = jl.register_kc(Y.T, X.T, scale=sigma, annealing=5, restarts=100)
    with cl.take_time('PCR impl'):
        transformation = jl.register_kc(Y.T, X.T, scale=sigma, annealing=5, restarts=100)
    transformation.linear._jl_display()
    transformation.translation._jl_display()


if __name__ == "__main__":
    main()
