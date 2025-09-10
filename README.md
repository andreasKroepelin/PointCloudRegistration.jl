# `PointCloudRegistration.jl`

<video src="PointCloudRegistration/raw/branch/main/examples/animations/register-julia-logo.mp4" type="video/mp4" width="600" height="400" controls>
</video>

**PointCloudRegistration** is a Julia package for registering two point clouds.
This is also known as aligning or superimposing.

Given two point clouds `source` and `target, this package can perform
* **rigid registration**, i.e. rotate and translate `source` such that it
  matches `target`, or
* **nonrigid registration**, i.e. shift the points in `source` individually but
  coherently to match `target`.

Currently, the following four methods are implemented:
* **`register_rmsd`:**
  Rigid registration that minimizes the *root mean square deviation* between
  `source` and `target`; assumes that equal-index points correspond to each
  other and that all pairs of points are equally relevant.
* **`register_gmc`:**
  Rigid registration that minimizes the *Geman-McClure loss* between `source`
  and `target`; assumes that equal-index points correspond to each
  other but caps the influence of far apart pairs of points (outliers).
* **`register_kc`:**
  Rigid registration that maximizes the *kernel correlation* between `source`
  and `target`; assumes no correspondences and is robust against outliers.
* **`register_cpd`:**
  Nonrigid registration that estimates a *coherent point drift* from `source`
  to `target`; internally estimates correspondences and is configurable to
  account for outliers.
