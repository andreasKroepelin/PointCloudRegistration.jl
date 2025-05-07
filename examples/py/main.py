from juliacall import Main as jl
import numpy as np


def main():
    print("Hello from py!")
    jl.seval("using PointCloudRegistration")
    X_np = np.random.rand(2, 5)
    transformation = jl.register(X_np, X_np, correspondences=jl.Symbol("known"))
    print(transformation.linear)


if __name__ == "__main__":
    main()
