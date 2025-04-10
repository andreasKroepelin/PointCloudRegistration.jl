# Point Cloud Registration

This package provides three main functions to find an `AffineMap` that maps a
point cloud `Y` on a point cloud `X`.

## You know the correspondences with certainty

```julia
register(Y, X)
```

## You know the correspondences but some might be wrong

```julia
register_robustly(Y, X)
```

## You don't know correspondences

```julia
register_densities(Y, X)
```

