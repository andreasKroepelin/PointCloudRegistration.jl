#import "@preview/lovelace:0.3.0": pseudocode-list
// #set page(margin: 1mm, width: 210mm - 25mm, height: 297mm - 25mm)
#set text(
  size: 13pt,
  font: "Gentium",
  number-type: "old-style",
)
// #show math.equation: set text(font: "TeX Gyre Pagella Math")
#show math.equation: set text(
  font: (
    (name: "TeX Gyre Pagella Math", covers: regex("[ℝ|ℤ]")),
    "Libertinus Math",
  ),
)
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
#let Diag = $op("Diag")$

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
=== Bayesian Coherent Point Drift
It is convenient to collect the target points $bold(x)_1, ..., bold(x)_I in
RR^D$ and the source points $bold(y)_1, ..., bold(y)_J in RR^D$ into matrices
$bold(X) in RR^(D times I)$ and $bold(Y) in RR^(D times J)$, respectively.
Compute $bold(G) in RR^(J times J)$ with entries
$
  g_(i j) = exp(- norm(bold(y)_i - bold(y)_j)^2 / (2 beta^2)) .
$
Compute the volume $B$ of the target's bounding box.

Initialize:
- $bold(Sigma) = bold(I) in RR^(J times J)$
- $macron(sigma)^2 = 0$
- $bold(b) = 1 / J exp(- D / (2sigma^2)) bold(1) in RR^J$
- $bold(v)_1, ..., bold(v)_J = bold(0) in RR^D$
- $c_(i j) = 1$
- $Z = I J$

In every iteration:
#pseudocode-list[
  + $r_(i j) := norm(bold(y)_j + bold(v)_j - bold(x)_i)^2$
  + $sigma^2 := 1 / (Z D) sum_(i j) c_(i j) r_(i j) quad + quad macron(sigma)^2$
  + Set $c_(i j) := p_i q_j b_j exp(- r_(i j) / (2 sigma^2))$
  + Set $o := omega / ((1 - omega) B) (2pi sigma^2)^(D slash 2)$
  + Set $d_i := sum_j c_(i j)$
  + Set $c_(i j) := c_(i j) slash (d_i + o)$
  + Set $d_i := d_i slash (d_i + o)$
  + Set $e_j := sum_i c_(i j)$
  + $Z := sum_j e_j$
  + Set $hat(bold(X)) := bold(X) bold(C)$
  // + Set $hat(bold(V)) := hat(bold(X)) - bold(Y) Diag(bold(e))$
  + $hat(bold(v))_j := hat(bold(x))_j - e_j bold(y)_j$
  + Set $bold(Sigma) := (lambda G^(-1) + 1/sigma^2 Diag(bold(e)))^(-1)$
  + $bold(V) := 1 / sigma^2 hat(bold(V)) bold(Sigma)$
  + $b_j := exp(psi(kappa + e_j) - psi(kappa J + Z) - (D sigma_j^2) / (2 sigma^2))$
  + $macron(sigma)^2 := 1 / Z sum_j e_j sigma_j^2$
]

$
  sigma^2 &:=
  1 / (hat(I) D)
  ( sum_i d_i bold(x)_i^trp bold(x)_i
    - 2sum_(i j) c_(j i) bold(x)_i^trp (bold(y)_j + bold(v)_j)
    + sum_j e_j (bold(y)_j + bold(v)_j)^trp (bold(y)_j + bold(v)_j) )
  + macron(sigma)^2
  \ &=
  1 / (hat(I) D)
  ( sum_(i j) c_(j i) bold(x)_i^trp bold(x)_i
    - 2sum_(i j) c_(j i) bold(x)_i^trp (bold(y)_j + bold(v)_j)
    + sum_(i j) c_(j i) (bold(y)_j + bold(v)_j)^trp (bold(y)_j + bold(v)_j) )
  + macron(sigma)^2
  \ &=
  1 / (hat(I) D)
  sum_(i j) c_(j i) (bold(x)_i^trp bold(x)_i
    - 2 bold(x)_i^trp (bold(y)_j + bold(v)_j)
    + (bold(y)_j + bold(v)_j)^trp (bold(y)_j + bold(v)_j) )
  + macron(sigma)^2
  \ &=
  1 / (hat(I) D)
  sum_(i j) c_(j i) norm(bold(y)_j + bold(v)_j - bold(x)_i)^2
  + macron(sigma)^2
$

==== CPD as Gaussian Process regression
Recall that, in Gaussian process regression, the posterior mean can be found by
computing
$
  f(bold(z)_"test") = bold(G)_("test","train") (bold(G)_"train" + bold(N))^(-1) f(bold(z)_"train")
$
where $bold(G)_("test","train")$ is the Gram matrix of test and training data,
$bold(G)_"train"$ is the Gram matrix of the training data with itself and
$bold(N)$ is a diagonal matrix representing input noise.

In CPD, the training _and_ test data are both the source points
$bold(y)_1, ..., bold(y)_J$.
The observed regression values are the difference vectors between these source
points and their corresponding aligned target points
$
  hat(bold(x))_j = (sum_i c_(i j) bold(x)_i) / (sum_i c_(i j)),
$
i.e.
$
  hat(bold(v))_j = hat(bold(x))_j - bold(y)_j.
$
In matrix notation, we have
$
  hat(bold(V)) = bold(X) bold(C) underbrace(Diag(bold(1)_I^trp bold(C)), =: bold(Gamma))^(-1) - bold(Y).
$
Due to the coincidence of training and test data, we denote $bold(G) =
bold(G)_("test","train") = bold(G)_"train"$ with entries
$
  G_(j j') = sigma_v^2 exp(- norm(bold(y)_j - bold(y)_(j'))^2 / (2 beta^2)).
$
The regression acts per dimension of the displacement.
Therefore, the general GP regression posterior mean formula becomes:
$
  bold(V)_(k,:)^trp = bold(G) (bold(G) + bold(N))^(-1) hat(bold(V))_(k,:)^trp
$
for $k = 1, ..., D$.
By concatenating these dimensions, we obtain the matrix expression
$
  bold(V)^trp = bold(G) (bold(G) + bold(N))^(-1) hat(bold(V))^trp
$
or equivalently (due to the symmetry of $bold(G)$ and $bold(N)$)
$
  bold(V) = hat(bold(V)) (bold(G) + bold(N))^(-1) bold(G)
  = hat(bold(V)) (bold(G) + bold(N))^(-1) (bold(G)^(-1))^(-1)
  = hat(bold(V)) (bold(G)^(-1) (bold(G) + bold(N)))^(-1)
  = hat(bold(V)) (bold(I) + bold(G)^(-1) bold(N))^(-1).
$

We choose $bold(N) = sigma^2 bold(Gamma)^(-1)$ such that source
points that correspond only weakly to target points have their displacement
considered to be noisier.
Let us manipulate the regression equation to decrease the computational effort.
$
  bold(V) &= hat(bold(V)) (bold(I) + bold(G)^(-1) bold(N))^(-1) \
  &= hat(bold(V)) (bold(I) + bold(G)^(-1) sigma^2 bold(Gamma)^(-1))^(-1) \
  &= hat(bold(V)) ((sigma^(-2) bold(Gamma) + bold(G)^(-1)) sigma^2 bold(Gamma)^(-1))^(-1) \
  &= hat(bold(V)) sigma^(-2) bold(Gamma) (sigma^(-2) bold(Gamma) + bold(G)^(-1))^(-1) \
  &= (bold(X) bold(C) bold(Gamma)^(-1) - bold(Y)) sigma^(-2) bold(Gamma) (sigma^(-2) bold(Gamma) + bold(G)^(-1))^(-1) \
  &= sigma^(-2) (bold(X) bold(C) - bold(Y) bold(Gamma)) (sigma^(-2) bold(Gamma) + bold(G)^(-1))^(-1) \
  &= sigma_v^2 / sigma^2 (bold(X) bold(C) - bold(Y) bold(Gamma)) (sigma_v^2 / sigma^2 bold(Gamma) + sigma_v^2 bold(G)^(-1))^(-1) \
$
The last reformulation helps with keeping certain operations such as the
solution of the linear system unit-free.


=== Divergence Free Interpolation
We implement the method described in @divfree.
It presents a parameterization of a displacement field that is inherently
divergence free.
The paper elaborates the three-dimensional case, which we will generalize to
arbitrary dimensions $D$ in what follows.

Given source points $bold(y)_1, ..., bold(y)_J in RR^D$ and target points
$bold(x)_1, ..., bold(x)_I in RR^D$, we want to find a velocity field
$bold(v): [0, 1]^D -> RR^D$ such that $hat(bold(y))_j = bold(t)_j (1)$ where
$bold(t)_j: RR -> RR^D$ is a solution of the initial value problem
$
  bold(t)'_j (bold(z)) &= bold(v)(z) \
  bold(t)_j (0) &= bold(y)_j
$
and the $bold(x)_i$ have a high likelihood in the Gaussian mixture centered at
the $hat(bold(y))_j$.

We want $bold(v)$ to have the following two properties.
It is divergence free ($op("div") bold(v) equiv 0$) and there is no flow in and
out of the domain $[0, 1]^D$.
The idea is to construct $bold(v)$ as the sum of basis functions that fulfill
these properties and exploit their linearity to make them hold for $bold(v)$ as
well.

First, we set a maximum frequency $F in ZZ_(>0)$.
Then, for every $bold(f) in {1, ..., F}^D$ and every unordered pair ${m, n}
subset {1, ..., D}$, we define a basis function
$
  bold(v)_{m, n}^bold(f) (bold(z))
  =
  c_n^bold(f) (bold(z)) bold(e)_m
  -
  c_m^bold(f) (bold(z)) bold(e)_n
$
where $bold(e)_i$ is the $i$-th canonical basis vector and $c_i^bold(f)$ is
defined as
$
  c_i^bold(f) (bold(z))
  =
  1 / 2^D f_i pi cos(f_i pi z_i)
  dot
  product_(j = 1 \ j != i)^D
  sin(f_j pi z_j)
  .
$
With coefficients $a_{m, n}^bold(f) in RR$, we construct the velocity field as
$
  bold(v)(bold(z))
  =
  sum_(bold(f) in {1, ..., F}^D)
  sum_(m, n in {1, ..., D} \ m < n)
  a_{m, n}^bold(f) bold(v)_{m, n}^bold(f) (bold(z))
  .
$
Note that there are $F^D dot binom(D, 2)$ basis vectors/coefficents.

We show that $op("div") bold(v)_{m, n}^bold(f) equiv 0$ always holds.
Recall that the divergence is the sum of the partial derivatives of the
respective output dimensions, i.e. we have:
$
  op("div") bold(v)_{m, n}^bold(f) (bold(z))
  =
  partial_(z_m) c_n^bold(f) (bold(z))
  -
  partial_(z_n) c_m^bold(f) (bold(z))
  .
$
The two partial derivatives turn out to be equal:
$
  partial_(z_m) c_n^bold(f) (bold(z))
  &=
  1 / 2^D f_n pi cos(f_n pi z_n)
  dot
  f_m pi cos(f_m pi z_m)
  dot
  product_(j = 1 \ j != m, n)^D
  sin(f_j pi z_j)
  \
  partial_(z_n) c_m^bold(f) (bold(z))
  &=
  1 / 2^D f_m pi cos(f_m pi z_m)
  dot
  f_n pi cos(f_n pi z_n)
  dot
  product_(j = 1 \ j != m, n)^D
  sin(f_j pi z_j)
$
Thus, the divergence of the basis function is zero.

To show that there is no flow in and out of the domain $[0, 1]^D$, we have to
ensure $chevron.l bold(v)(bold(z)), bold(n)(bold(z)) chevron.r = 0$ for every
$bold(z) in partial [0, 1]^D$ on the boundary of the unit cube and its normal
vector $bold(n)$.
Note that $bold(z)$ is on the boundary iff one of its components is zero or one.
Suppose the $k$-th entry of $bold(z)$ is zero or one.
The normal vector of the $(D - 1)$-dimensional face $bold(z)$ sits on is either
$bold(e)_k$ or $-bold(e)_k$ where the sign is irrelevant for checking that the
dot product vanishes.
Again, due to the linearity of the dot product, it suffices to show
$chevron.l bold(v)_{m, n}^bold(f) (bold(z)), bold(e)_k chevron.r = 0$ for every
$bold(f) in {1, ..., F}^D$ and $1 <= m < n <= D$.

$bold(v)_{m, n}^bold(f) (bold(z))$ is trivially zero in every entry except the
$m$-th and $n$-th ones so the dot product is zero unless $k = m$ or $k = n$.
Let now $k = m$ (the case $k = n$ is analogous).
Recall that the $m$-th entry in the velocity basis vector is
$
  c_n^bold(f) (bold(z))
  =
  1 / 2^D f_n pi cos(f_n pi z_n)
  dot
  product_(j = 1 \ j != n)^D
  sin(f_j pi z_j)
  ,
$
so it contains a factor $sin(f_m pi z_m)$.
By assumption, $z_m = 0$ or $z_m = 1$, i.e. $z_m in ZZ$.
Since also $f_m in ZZ$ and $sin(i pi) = 0$ for all $i in ZZ$, it follows that
the $m$-th entry of the basis vector is zero and thus the whole dot product
with the normal vector vanishes.


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
