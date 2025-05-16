print("loading juliacall...")
from juliacall import Main as jl
print("loaded juliacall")
import cloudy as cl
import numpy as np


def main():
    print("loading data...")
    X = np.load("/home/andreas/morphs/1su4_A-1iwo_A_morph/initial.npy")
    Y = np.load("/home/andreas/morphs/1su4_A-1iwo_A_morph/target.npy")
    target = cl.PointCloud(X)
    source = cl.PointCloud(Y)
    sigma = 2.
    with cl.take_time('cloudy prep'):
        kc = cl.GriddedKernelCorrelation(target, source, sigma)
        mm = cl.GriddedMM(kc, cl.Pose(np.eye(3), -np.random.rand(3)))
    with cl.take_time('cloudy run'):
        transformation = mm.run(100)
    print(transformation.R)
    print(transformation.t)
    print("loading julia package...")
    jl.seval("using PointCloudRegistration")
    print("loaded julia package")
    # for precompilation:
    jl.register_kc(Y.T, X.T, scale=sigma, annealing=5, axisalign=True)
    with cl.take_time('PCR prep'):
        prepd_X = jl.prepare_target_kc(X.T, scale=sigma, annealing=5, axisalign=True)
    with cl.take_time('PCR run'):
        transformation = jl.register_kc(Y.T, prepd_X, restarts=100)
    transformation.linear._jl_display()
    transformation.translation._jl_display()
    with cl.take_time('PCR kabsch'):
        transformation = jl.register_rmsd(Y.T, X.T)
    transformation.linear._jl_display()
    transformation.translation._jl_display()


if __name__ == "__main__":
    main()
