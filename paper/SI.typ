#set text(
  font: "Gentium",
  number-type: "old-style",
)
#show math.equation: set text(font: "TeX Gyre Pagella Math")
#show raw: set text(font: "Atkinson Hyperlegible Mono", size: 1.1em)

#set heading(numbering: "1.1")

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
- sorting
  - LAP-correspondence sorting
