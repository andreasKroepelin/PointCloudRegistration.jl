#set page(columns: 2)
#set text(font: "New Computer Modern")
#set par(justify: true, spacing: .6em, first-line-indent: 1em)


= Introduction
Point clouds, i.e. sets of points in some $D$-dimensional space, are a versatile
model for physical objects in many disciplines, from laser scans in computer
vision to particle models of molecules in structural biology.
Typical operations on pairs of point clouds, such as their union, intersection,
or more complex comparisons, rely on a common reference frame.
However, the observation process generating the point coordinates often cannot
easily provide it.
Hence, _registration_ of point clouds (also known as _alignment_ or
_superimposing_) is a key tool at the beginning of many analysis workflows.
This can consist of either _rigid_ registration (one point cloud can be rotated
and translated to match the other) or _non-rigid_ registration (more intricate
transformations are allowed).

For performance critical computations involving point clouds, researchers and
practitioners might turn to the fast and research-appropriate language Julia.
While different packages in its ecosystem provide implementations of basic
registration algorithms, there is no comprehensive library covering the diverse
use cases of rigid and non-rigid registration, to our knowledge.
The presented package aims to fill this gap.

= Scope of the package
PointCloudRegistration.jl addresses applications within structural biology and
beyond concerning point cloud data.
Point clouds representing biological macromolecules were a prime consideration
during the development but no biological features are specifically assumed.
This also means that no surface or normal information is required or used.

The package is intended for the following general setting:
Given two weighted $D$-dimensional point clouds, a _target_ $X$ with points
$𝒙_1, ..., 𝒙_I in RR^D$ and weights $p_1, ..., p_I$ as well as a _source_ with
points $𝒚_1, ..., 𝒚_J in RR^D$ and weights $q_1, ..., q_J$, we want to find a
rigid or non-rigid transformation that maps the source onto the target in some
optimal way.

In summary, these features are provided by the package:
Rigid registration with known point-to-point correspondences via minimizing the
_root mean square displacement_, the _mean absolute displacement_, and the
_Geman-McClure loss_.
Rigid registration with unknown correspondences via maximizing the _kernel
correlation_ and the _iterative closest point_ method.
Non-rigid registration via _Bayesian coherent point drift_, _divergence free
shape interpolation_, _optimal transport_, and _kernel correlation maximization
with a Gaussian network model_.
Thinning of point clouds via _$k$-means_ and _DP-means_.

= Rigid registration



