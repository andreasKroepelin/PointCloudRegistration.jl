#set page(margin: 1mm, width: 210mm - 25mm, height: 297mm - 25mm)
#set text(
  size: 13pt,
  font: "Gentium",
  number-type: "old-style",
)
#show math.equation: set text(font: "TeX Gyre Pagella Math")
#show raw: set text(font: "Atkinson Hyperlegible Mono", size: 1.1em)

#set heading(numbering: "1.1")
#set par(justify: true)

#let idea = text.with(fill: gray, style: "italic")

#[
  #set align(center)
  #smallcaps[*Supplementary Information*]

  #text(size: 1.5em)[
    `PointCloudRegistration.jl`: \
    Rigid and Non-Rigid Registration of Point Clouds in Julia
  ]

  Andreas Kröpelin, Michael Habeck
]

= Description of algorithms

This package implements the following algorithms:
- rigid registration:
  - Kabsch
  - Geman-McClure Majorization Minimization
  - Kernel Correlation Majorization Minimization
  - Iterative Closest Point
- non-rigid registration:
  - Coherent Point Drift
- thinning:
  - DP-means
  - $k$-means

= Experiments

== Robustness against wrong correspondences

#idea[
  when start with two point clouds with perfect correspondence (e.g. from sequence
  alingment) and then increasingly shuffle around the indices, observe how well
  the rigid registration still performs
]

One of the key challenges of rigid registration is that often the iid Gaussian
residue assumption behind minimizing the RMSD is violated.
This generally has two possible reasons:
Either, the $i$-th point in the source does correspond to the $i$-th point in
the target but their difference cannot be explained by a rigid transformation.
Or, the assumed correspondence is wrong in the first place.
In this section, we will explore the effect of these violations on the
performance of the different rigid registration algorithms implemented.

== Stanford Bunny

=== Assembly from raw scans

#idea[
  pairwise registration of all raw scans of bunny, analyse how well this works.
  explain why it does not work and that we use less data than in the original
  assembly, namely surface information
]

=== Reassembly from artificial cuts

#idea[
  cut the assembled bunny point cloud in two with varying amount of overlap,
  analyse performance of rigid registration

  works better with more overlap

  success depends on initial scale of annealing (less success when scale is to
  large)
]

== Coherent Point Drift hyperparameters

#idea[
  systematic exploration of CPD result with different choice of hyperparameters
]

== EMDB ribosome data

#idea[
  demonstrate turning EMDB map into point cloud and performing registration
]

== GroEL subunits

#idea[
  demonstrate finding all global optima
]

== PDB structures

#idea[
  demonstrate alignment of PDB structures based on sequence alignment.
  analyse distribution of pairwise distances
]

== Restarts

#idea[
  show how many restarts each iterative method needs to find its "own" global
  optimum
]

== Python interface

#idea[
  use PCReg.jl from Python via juliacall, compare runtime with Python
  implementations
]

== Units

#idea[
  demonstrate using PCReg.jl functions with point clouds with units.
  compare runtime with and without units (should be equal)
]

