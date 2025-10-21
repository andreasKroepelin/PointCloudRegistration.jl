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

when start with two point clouds with perfect correspondence (e.g. from sequence
alingment) and then increasingly shuffle around the indices, observe how well
the rigid registration still performs

== Stanford Bunny

=== Assembly from raw scans

pairwise registration of all raw scans of bunny, analyse how well this works.
explain why it does not work and that we use less data than in the original
assembly, namely surface information

=== Reassembly from artificial cuts

cut the assembled bunny point cloud in two with varying amount of overlap,
analyse performance of rigid registration

works better with more overlap

success depends on initial scale of annealing (less success when scale is to
large)

== Coherent Point Drift hyperparameters

systematic exploration of CPD result with different choice of hyperparameters

== EMDB ribosome data

demonstrate turning EMDB map into point cloud and performing registration

== GroEL subunits

demonstrate finding all global optima

== PDB structures

demonstrate alignment of PDB structures based on sequence alignment.
analyse distribution of pairwise distances

== Restarts

show how many restarts each iterative method needs to find its "own" global
optimum

== Python interface

use PCReg.jl from Python via juliacall, compare runtime with Python
implementations

== Units

demonstrate using PCReg.jl functions with point clouds with units.
compare runtime with and without units (should be equal)

