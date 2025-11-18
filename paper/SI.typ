// #set page(margin: 1mm, width: 210mm - 25mm, height: 297mm - 25mm)
#set text(
  size: 13pt,
  font: "Gentium",
  number-type: "old-style",
)
#show math.equation: set text(font: "TeX Gyre Pagella Math")
#show raw: set text(font: "Atkinson Hyperlegible Mono", size: 1.1em)

#set heading(numbering: "1.1")
#set par(justify: true)
#let numbered(body) = {
  set math.equation(numbering: "(1)")
  body
}
#show bibliography: bib => {
  show link: set text(font: "Atkinson Hyperlegible Mono", size: .8em)
  bib
}

#let idea = text.with(fill: gray, style: "italic")

#let SE = $op("SE")$
#let trp = $upright(sans(T))$

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


== Rigid registration
We implement different algorithms that solve some version of the problem
#numbered[$
    min_(T in SE(d)) sum_(i = 1)^I sum_(j = 1)^J c_(i j) p_i q_j
    dot
    rho(norm(bold(x)_i - T(bold(y)_j)))
  $ <general-optimization-problem>]
where
- $SE(d)$ is the _special Euclidean group_ of rigid transformations in $d$
  dimensions,
- the _target_ $X$ is a $d$-dimensional point cloud with points $bold(x)_1, ..., bold(x)_I$,
- $p_i$ is the weight of the $i$-th point in the target,
- the _source_ $Y$ is a $d$-dimensional point cloud with points $bold(y)_1, ..., bold(y)_J$,
- $q_j$ is the weight of the $j$-th point in the source,
- $c_(i j)$ is the strength of the correspondence between the $i$-the point of
  the target and the $j$-th point of the source, and
- $rho$ is a cost function.
The algorithms differ by the choice of $c$ and $rho$ for which they try to solve
@general-optimization-problem[Problem].

=== Kabsch algorithm

The Kabsch algorithm @kabsch uses
$
  rho(r) &= r^2 \
  c_(i j) &= cases(1 &"if" i = j , 0 &"else")
$
which implies $I = J$, so source and target must have the same size.
The optimization problem then has a closed form solution, namely
$
  bold(t) &= macron(bold(x)) - bold(R) macron(bold(y)) \
  bold(R) &= bold(U)
  med
  op("Diag")(underbrace(1\, ...\, 1, d - 1 "times"), det(bold(U) bold(V)^trp))
  med
  bold(V)^trp
$
where $bold(S) = bold(U) bold(Sigma) bold(V)^trp$ is the singular value
decomposition of $bold(S)$ and
$
  macron(bold(x)) = sum_(i = 1)^I w_i bold(x)_i wide
  macron(bold(y)) = sum_(i = 1)^I w_i bold(y)_i wide
  bold(S) = sum_(i = 1)^I w_i bold(x)_i bold(y)_i^trp
  thick - thick
  macron(bold(x)) macron(bold(y))^trp
$
with weights
$
  w_i &= (p_i q_i) / (sum_(j = 1)^I p_j q_j).
$

=== Geman-McClure Majorization Minimization
For this appoach, we also assume known pairwise correspondences but we employ
the _Geman-McClure_ cost function:
$
  rho(r) &= r^2 / (2sigma^2 + r^2) \
  c_(i j) &= cases(1 &"if" i = j , 0 &"else")
$
The scale parameter $sigma$ controls if the cost is more sensitive to larger
or smaller deviations.
Note that the parameterization slightly differs from how the cost function is
usually formulated in order to achieve
$
  rho''(0) = 1 / sigma^2
$
matching the behavior of the kernel correlation cost function.

Since $rho$ is non-convex, we cannot perform the optimization using a closed
form solution but rather have to proceed with an iterative approach, namely
_majorization minimization_:
We find a simpler objective function that _majorizes_ the original one and then
_minimize_ it.
Specifically, we want to find additional weights $k_1, ..., k_I$ such that
$
  sum_(i = 1)^I p_i q_i dot rho(norm(bold(x)_i - T(bold(y)_i)))
  <=
  sum_(i = 1)^I p_i q_i dot (k_i norm(bold(x)_i - T(bold(y)_i))^2 + C_i)
$
with additive constants $C_i$ that do not depend on $T$.
For any fixed $hat(T) in SE(d)$, this is satisfied by
$
  k_i = (2sigma^2) / (2sigma^2 + norm(bold(x)_i - hat(T)(bold(y)_i))^2)^2
$
as can be shown in the following way, using
$r_i := norm(bold(x)_i - T(bold(y)_i))$ and
$hat(r)_i := norm(bold(x)_i - hat(T)(bold(y)_i))$):
$
  rho(r_i)
  &= r_i^2 / (2sigma^2 + r_i^2) \
  &= r_i^2 / (2sigma^2 + r_i^2)
  - hat(r)_i^2 / (2sigma^2 + hat(r)_i^2)
  + hat(r)_i^2 / (2sigma^2 + hat(r)_i^2) \
  &= (2sigma^2 (r_i^2 - hat(r)_i^2)) / ((2sigma^2 + r_i^2) (2sigma^2 + hat(r)_i^2))
  + ((2sigma^2 + hat(r)_i^2) hat(r)_i^2) / (2sigma^2 + hat(r)_i^2)^2 \
  &<= (2sigma^2 (r_i^2 - hat(r)_i^2)) / (2sigma^2 + hat(r)_i^2)^2
  + ((2sigma^2 + hat(r)_i^2) hat(r)_i^2) / (2sigma^2 + hat(r)_i^2)^2 \
  \
  &=
  underbrace((2sigma^2) / (2sigma^2 + hat(r)_i^2)^2, = k_i) dot r_i^2
  + underbrace(hat(r)_i^4 / (2sigma^2 + hat(r)_i^2)^2, = C_i)
$
where the inequality in the third step comes from substituting $r_i$ in the
denominator of the first term by $hat(r)_i$.
If $r_i > hat(r)_i$, this decreases the denominator and increases the term.
If $r_i < hat(r)_i$, this increases the denominator and decreases the absolute
value of the term, but the $r_i^2 - hat(r)_i^2$ in the numerator then makes the
whole term negative, so it is overall still increased.
For $r_i = hat(r)_i$, the inequality is trivially true.

We can drop the constants $C_i$ for the minimization and the remaining problem
$
  min_(T in SE(d))
  sum_(i = 1)^I
  p_i q_i k_i
  norm(bold(x)_i - T(bold(y)_i))^2
$
differs from the one solved by the Kabsch algorithm only in the weights (with
$p_i q_i k_i$ as opposed to $p_i q_i$), so this is the only thing we need to
adapt:
$
  w_i &= (p_i q_i k_i) / (sum_(j = 1)^I p_j q_j k_j)
$
The computation of $bold(macron(x))$, $macron(bold(y))$, $bold(S)$ and then
$bold(t)$ and $bold(R)$ stays the same.

=== Kernel Correlation Majorization Minimization
=== Optimization strategies
#idea[simulated annealing and stochastic MM]

==== Stochastic Majorization Minimization
A very fruitful observation is that in all central computations of the iterative
algorithms presented here, we find sums over the source points.
Specifically, they have the form
$
  sum_(j = 1)^J q_j med dots.c
$
which lends itself to computing this sum stocastically.
That is, for the sum $sum_(j = 1)^J q_j A(j)$ with some (scalar or vector
valued) quantity $A$ that depends on $j$, we draw $m$ times from the
categorical distribution with weights $q_1, ..., q_J$ and obtain indices
$j_1, ..., j_m in {1, ..., J}$.
We then compute $Z sum_(l = 1)^m A(j_l)$ with $Z = 1/m sum_(j = 1)^J q_j$.
This is unbiased since
$
  EE_q [(sum_(j = 1)^J q_j) / m sum_(l = 1)^m A(j_l)]
  &=
  (sum_(j = 1)^J q_j) / m sum_(l = 1)^m EE_q [A(j_l)]
  \ &=^(j_l "are iid.")
  (sum_(j = 1)^J q_j) / m m EE_q [A(j)]
  \ &=
  (sum_(j = 1)^J q_j) (sum_(j = 1)^J q_j A(j)) / (sum_(j = 1)^J q_j)
  \ &=
  sum_(j = 1)^J q_j A(j) .
$
These sums either occur in pairs where two of them are divided by each other
or when their constant scaling does not matter.
We can hence actually ignore $Z$ and compute $sum_(l = 1)^m A(j_l)$ instead of
$sum_(j = 1)^J q_j A(j)$.


=== Iterative Closest Point
Because of its widespread adoption we also implemented the _Iterative Closest
Point_ method.
Here, we tackle the following optimisation problem in every iteration:
$
  min_(T in SE(d))
  sum_((i, j) in C)
  p_i q_j
  norm(bold(x)_i - T(bold(y)_j))^2
$
where
$
  C = { (i, j)
    med : med
    i = op("argmin", limits: #true)_(i' = 1, ..., I) norm(bold(x)_i' - hat(T)(bold(y)_j))
    "and"
    norm(bold(x)_i - hat(T)(bold(y)_j)) <= D }
$
with a cut off distance $D$.



== Non-rigid registration
=== Coherent Point Drift
== Thinning
=== DP-means
=== $k$-means

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

== Restarts and Optimization

#idea[
  explore how many restarts reach the global optimum for each iterative method
  and different optimization strategies; justify default parameters

  measure optimization strategy the following way:
  proportion $p$ of restarts reach global optimum, so after $k$ restarts, the
  probability of not having found the global maximum is $(1 - p)^k$.
  we want to limit this to some $q$, i.e. $(1 - p)^k <= q$ iff
  $
    k >= log(q) / log(1 - p)
  $
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


#bibliography("bibliography.bib")
