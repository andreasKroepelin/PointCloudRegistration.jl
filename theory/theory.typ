#import "@preview/lovelace:0.3.0": *
// #set page(margin: 1cm)
#set text(font: "Gentium", number-type: "old-style")
// #set text(size: 20pt)
#show math.equation: set text(font: "TeX Gyre Pagella Math")
#show math.equation: set block(breakable: true)
#set heading(numbering: "1.1")
#show heading: smallcaps
#set enum(indent: 1em)

#let trp = $sans(upright(T))$
#let kernel = $phi_i (bold(y)'_j)$
#let Diag = $op("Diag")$

#text(
  2em,
  weight: "bold",
)[Some theoretical notes on \ implementing rigid registration]
#line(length: 75%)

= Kernel Correlation

We are given target points $bold(x)_i in RR^d$ with weights $q_i in RR$ and
source points $bold(y)_j in RR^d$ with weights $p_j in RR$.

The _kernel correlation_ of the source and the target is defined as
$
  kappa
  &=
  sum_(i j) q_i p_j phi_sigma (norm(bold(x)_i - bold(y)_j)) \
  &=
  sum_j p_j sum_i q_i phi_i (bold(y)_j)\
  &=
  sum_j p_j tilde(q)(bold(y)_j)
$
where $phi_i (bold(z)) = phi_sigma (norm(bold(x)_i - bold(z)))$ and
$tilde(q): RR^d -> RR$ is the KDE of the target.
When $tilde(q)$ can be evaluated quickly, computing $kappa$ becomes fast.

For the current pose $(bold(R), bold(t))$, let us use the notation
$bold(y)'_j := bold(R) bold(y)_j + bold(t)$.

The MM-algorithm consists of repeating these steps:
+ Compute
  $ w_(i j) = phi_i (bold(y)'_j) $
  for all $i$, $j$.
+ Divide $w_(i j)$ by $sum_(i j) q_i p_j w_(i j)$.
+ Compute $bold(macron(x)) = sum_(i j) w_(i j) q_i p_j bold(x)_i$
  and $bold(macron(y)) = sum_(i j) w_(i j) q_i p_j bold(y)_j$.
+ Compute
  $
    bold(S) =
    sum_(i j)
    w_(i j) q_i p_j
    (bold(x)_i - bold(macron(x)))
    (bold(y)_j - bold(macron(y)))^trp
  $
+ Find new $bold(R)$ and $bold(t)$ based on $bold(S)$, $bold(macron(x))$, and
  $bold(macron(y))$.

== Gridding trick

Define a suitable grid $cal(G) subset RR^d$.

Key idea:
$
  phi_sigma (norm(bold(x)_i - bold(R) bold(y)_j - bold(t)))
  &approx
  sum_(bold(z)_k in cal(G))
  phi_sigma (norm(bold(x)_i - bold(z)_k))
  med
  delta(norm(bold(z)_k - R bold(y)_j - t)) \
  &=
  sum_k phi_(i k) delta_(j k)
  =:
  phi_(i k_j)
$
where we obtain $k_j$, the grid index corresponding to
$bold(R) bold(y)_j + bold(t)$, by truncated division by the grid cell width.
It is then
$
  w_(i j) approx phi_(i k_j) / (sum_(i' j') q_i' p_j' phi_(i' k_j'))
  .
$

For computing $macron(bold(x))$, we observe
$
  macron(bold(x))
  &=
  sum_(i j) w_(i j) q_i p_j bold(x)_i \
  // &approx
  // sum_(i j) phi_(i k_j) / (sum_(i' j') q_i' p_j' phi_(i' k_j')) q_i p_j bold(x)_i
  &=
  (sum_(i j) kernel q_i p_j bold(x)_i) / (sum_(i j) q_i p_j kernel ) \
  &=
  (sum_j p_j sum_i kernel q_i bold(x)_i) / (sum_j p_j sum_i kernel q_i)
  =
  (sum_j p_j tilde(bold(x))(bold(y)'_j)) / (sum_j p_j tilde(q)(bold(y)'_j))
$
where
$
  tilde(bold(x))(bold(z)) = sum_i phi_i (bold(z)) q_i bold(x)_i,
  quad quad
  tilde(q)(bold(z)) = sum_i phi_i (bold(z)) q_i.
$

Similarly, for $macron(bold(y))$, we have
$
  macron(bold(y))
  &=
  sum_(i j) w_(i j) q_i p_j bold(y)_j \
  &approx
  (sum_(i j) phi_(i k_j) q_i p_j bold(y)_j) / (sum_(i j) q_i p_j phi_(i k_j)) \
  &=
  (sum_j p_j bold(y)_j sum_i phi_(i k_j) q_i) / (sum_j p_j tilde(q)_k_j)
  =
  (sum_j p_j tilde(q)_k_j bold(y)_j) / (sum_j p_j tilde(q)_k_j)
  .
$

Finally, the covariance matrix $bold(S)$ becomes, (up to a constant factor that
does not matter for finding the rotation):
$
  bold(S)
  &=
  sum_(i j) w_(i j) q_i p_j (bold(x)_i - macron(bold(x))) (bold(y)_j - macron(bold(y)))^trp \
  & prop^approx
  sum_(i j) phi_(i k_j) q_i p_j (bold(x)_i - macron(bold(x))) (bold(y)_j - macron(bold(y)))^trp \
  &=
  sum_j p_j (sum_i q_i phi_(i k_j) (bold(x)_i - macron(bold(x)))) (bold(y)_j - macron(bold(y)))^trp \
  &=
  sum_j p_j (sum_i q_i phi_(i k_j) bold(x)_i - macron(bold(x)) sum_i q_i phi_(i k_j)) (bold(y)_j - macron(bold(y)))^trp \
  &=
  sum_j p_j (tilde(bold(x))_k_j - tilde(q)_k_j macron(bold(x))) (bold(y)_j - macron(bold(y)))^trp \
$

=== On computing $tilde(bold(x))$ and $tilde(q)$
To find $tilde(bold(x))$ and $tilde(q)$, we need to compute $d + 1$ kernel
density estimations.
A KDE of the target points $bold(x)_i$ with weights $q_i$ gives us $tilde(q)$.
Using the weights $q_i x_(i ell)$, where $x_(i ell)$ is the $ell$-th entry in
$bold(x)_i$ for $ell = 1,...,d$, for the KDE instead, we obtain the $ell$-th
entry for every $tilde(bold(x))_k$.


=== Optimized MM procedure
First, define grid $cal(G)$ with suitable cell width ($sigma slash 4$?).
Precompute $tilde(bold(x))$ and $tilde(q)$ as KDEs (via FFT).
Then, in every MM-iteration (repeat until no $k_j$ changes):
+ Compute $k_j$ by placing every $bold(R) bold(y)_j + bold(t)$ on $cal(G)$.
+ Compute
  $macron(bold(x)) = sum_j p_j tilde(bold(x))_k_j$,
  $macron(bold(y)) = sum_j p_j tilde(q)_k_j bold(y)_j$, and
  $kappa = sum_j p_j tilde(q)_k_j$.
+ Divide $macron(bold(x))$ and $macron(bold(y))$ by $kappa$.
+ Compute
  $
    bold(S) = sum_j p_j (tilde(bold(x))_k_j - tilde(q)_k_j macron(bold(x))) (bold(y)_j - macron(bold(y)))^trp
  $
+ Find new $bold(R)$ and $bold(t)$ based on $bold(S)$, $bold(macron(x))$, and
  $bold(macron(y))$.

Note that the first two steps can be performed in a single pass over all source
points.
A second pass is then needed for $bold(S)$.



=== Approximation error
What is the relative error of evaluating a Gaussian kernel
$phi(r) = exp(- r^2 / (2 sigma^2))$ at the inflection point $r = sigma$ (maximum
derivative) with an error in $r$ of $(sqrt(d) Delta) / 2$ (maximum distance to
a grid cell center with grid spacing $Delta$) when we set $Delta = alpha sigma$?

$
  (phi(r) - phi(r + (sqrt(d) Delta) / 2)) / phi(r)
  =
  1 - phi((1 + (sqrt(d) alpha) / 2) sigma) / phi(sigma)
  =
  1 - exp(- 1 / 2 (1 + (sqrt(d) alpha) / 2)^2 + 1 / 2)
$

== Random Fourier Features

Let us assume we can compute a $R times I$ matrix $bold(Z)(X)$ for a point cloud
$X$ with $I$ points such that
$bold(Z)(X)^trp bold(Z)(Y) approx (phi_sigma (x_i, y_j))_(i j) = bold(Phi)$.
If $bold(p)$ and $bold(q)$ are the weight vectors of the target $X$ and the
source $Y$, respectively, we can approximate the kernel correlation as
$
  kappa(X, Y)
  =
  bold(p)^trp bold(Phi) bold(q)
  approx
  bold(p)^trp (bold(Z)(X)^trp bold(Z)(Y)) bold(q)
  =
  (bold(Z)(X) bold(p))^trp (bold(Z)(Y) bold(q))
$

For $kappa macron(bold(x)) = sum_(i j) phi_(i j) p_i q_j bold(x)_i$ we then have
$
  sum_(i j) phi_(i j) p_i q_j bold(x)_i
  &=
  sum_i p_i bold(x)_i (sum_j phi_(i j) q_j)
  =
  sum_i p_i bold(x)_i (bold(Phi) bold(q))_i \
  &=
  sum_i (bold(X) Diag(bold(p)))_(: i) (bold(Phi) bold(q))_i \
  &=
  bold(X) Diag(bold(p)) bold(Phi) bold(q) \
  &approx
  bold(X) Diag(bold(p)) bold(Z)(X)^trp bold(Z)(Y) bold(q) .
$

Similarly, for $kappa macron(bold(y)) = sum_(i j) phi_(i j) p_i q_j bold(y)_j$
we have
$
  sum_(i j) phi_(i j) p_i q_j bold(y)_j
  &=
  sum_i p_i (sum_j phi_(i j) q_j bold(y)_j) \
  &=
  sum_i p_i (sum_j phi_(i j) (bold(Y) Diag(bold(q)))_(: j))
  =
  sum_i p_i (sum_j phi_(i j) (bold(Y) Diag(bold(q)))^trp_(j :)) \
  &=
  sum_i p_i (bold(Phi) Diag(bold(q)) bold(Y)^trp)_(i :) \
  &=
  (bold(Phi) Diag(bold(q)) bold(Y)^trp)^trp bold(p) \
  &=
  bold(Y) Diag(bold(q)) bold(Phi)^trp bold(p) \
  &approx
  bold(Y) Diag(bold(q)) bold(Z)(Y)^trp bold(Z)(X) bold(p) .
$

And then for
$bold(S) =
sum_(i j) phi_(i j) p_i q_j
(bold(x)_i - macron(bold(x)))
(bold(y)_j - macron(bold(y)))^trp$:

$
  sum_(i j) phi_(i j) p_i q_j
  (bold(x)_i - macron(bold(x)))
  (bold(y)_j - macron(bold(y)))^trp
  &=
  sum_j q_j
  (sum_i phi_(i j) p_i (bold(x)_i - macron(bold(x))))
  (bold(y)_j - macron(bold(y)))^trp \
  &=
  sum_j q_j
  (sum_i phi_(i j) p_i bold(x)_i - (sum_i phi_(i j) p_i) macron(bold(x)))
  (bold(y)_j - macron(bold(y)))^trp \
  &=
  sum_j q_j
  ((bold(Phi)^trp Diag(bold(p)) bold(X)^trp)_(j :) - (bold(Phi)^trp bold(p))_j macron(bold(x)))
  (bold(y)_j - macron(bold(y)))^trp \
  &=
  sum_j q_j
  ((bold(X) Diag(bold(p)) bold(Phi))_(: j) - (macron(bold(x)) bold(p)^trp bold(Phi))_(: j))
  (bold(y)_j - macron(bold(y)))^trp \
  &=
  sum_j q_j
  ((bold(X) Diag(bold(p)) - macron(bold(x)) bold(p)^trp) bold(Phi))_(: j)
  (bold(y)_j - macron(bold(y)))^trp \
  &=
  sum_j
  ((bold(X) Diag(bold(p)) - macron(bold(x)) bold(p)^trp) bold(Phi))_(: j)
  (bold(Y) Diag(bold(q)) - macron(bold(y)) bold(q)^trp)^trp_(j :) \
  &=
  (bold(X) Diag(bold(p)) - macron(bold(x)) bold(p)^trp)
  bold(Phi)
  (bold(Y) Diag(bold(q)) - macron(bold(y)) bold(q)^trp)^trp \
  &approx
  (bold(X) Diag(bold(p)) - macron(bold(x)) bold(p)^trp)
  bold(Z)(X)^trp bold(Z)(Y)
  (bold(Y) Diag(bold(q)) - macron(bold(y)) bold(q)^trp)^trp \
  &=
  (bold(X) Diag(bold(p)) bold(Z)(X)^trp - macron(bold(x)) (bold(Z)(X) bold(p))^trp)
  (bold(Z)(Y) Diag(bold(q)) bold(Y)^trp - bold(Z)(Y) bold(q) macron(bold(y))^trp) \
$

In summary, we do the following computations:
- unrelated to the source:
  + $bold(Z)(X) in RR^(R times I)$
  + $tilde(bold(p)) := bold(Z)(X) bold(p) in RR^R$
  + $tilde(bold(X)) := bold(X) Diag(bold(p)) bold(Z)(X)^trp in RR^(d times R)$
- for each (potentially transformed) source $Y$:
  + $bold(Z)(Y) in RR^(R times J)$
  + $tilde(bold(q)) := bold(Z)(Y) bold(q) in RR^R$
  + $tilde(bold(Y)) := bold(Y) Diag(bold(q)) bold(Z)(Y)^trp in RR^(d times R)$
  + $kappa = tilde(bold(p))^trp tilde(bold(q))$
  + $macron(bold(x)) = 1 / kappa tilde(bold(X)) tilde(bold(q)) in RR^d$
  + $macron(bold(y)) = 1 / kappa tilde(bold(Y)) tilde(bold(p)) in RR^d$
  + $bold(S) &=
      (tilde(bold(X)) - macron(bold(x)) tilde(bold(p))^trp)
      (tilde(bold(Y)) - macron(bold(y)) tilde(bold(q))^trp)^trp
      =
      tilde(bold(X)) tilde(bold(Y))^trp
      - tilde(bold(X)) tilde(bold(q)) macron(bold(y))^trp
      - macron(bold(x)) tilde(bold(p))^trp tilde(bold(Y))^trp
      + macron(bold(x)) tilde(bold(p))^trp tilde(bold(q)) macron(bold(y))^trp
      \ &=
      tilde(bold(X)) tilde(bold(Y))^trp
      - kappa macron(bold(x)) macron(bold(y))^trp
      - (kappa macron(bold(y)) macron(bold(x))^trp)^trp
      + macron(bold(x)) kappa macron(bold(y))^trp
      \ &=
      tilde(bold(X)) tilde(bold(Y))^trp
      - kappa macron(bold(x)) macron(bold(y))^trp
      in RR^(d times d)$

#pagebreak()

= One-pass covariance matrix

Algorithm:
#pseudocode-list[
  - *input:* points $bold(x)_1, ..., bold(x)_n$
  - *output:* mean $bold(mu)$ and covariance matrix $bold(C)$
  + $bold(mu) <- bold(0)$
  + $bold(C) <- bold(0)$
  + *for* $i = 1, ..., n$
    + $bold(d) <- bold(x)_i - bold(mu)$
    + $bold(mu) <- bold(mu) + bold(d) / i$
    + $bold(C) <- bold(C) + (i - 1) / i bold(d) bold(d)^trp$
  + *end*
  + $bold(C) <- bold(C) slash n$
]

Proof of correctness:

Assume $bold(mu)_(i - 1) = 1 / (i - 1) sum_k^(i - 1) bold(x)_k$.
We are then supposed to compute $bold(mu)_i$ as
$
  bold(mu)_i
  =
  bold(mu)_(i - 1) + (bold(x)_i - bold(mu)_(i - 1)) / i
  =
  (i bold(mu)_(i - 1) + bold(x)_i - bold(mu)_(i - 1)) / i
  =
  (bold(x)_i + (i - 1) bold(mu)_(i - 1)) / i
  =
  (bold(x)_i + sum_k^(i - 1) bold(x)_k) / i
  =
  (sum_k^i bold(x)_k) / i
  .
$
Next, assume that $bold(C)_(i - 1) = sum_k^(i - 1) (bold(x)_k - bold(mu)_(i - 1))
(bold(x)_k - bold(mu)_(i - 1))^trp$.
We then compute $bold(C)_i$ as
$
  bold(C)_i
  &=
  bold(C)_(i - 1) + (i - 1) / i
  (bold(x)_i - bold(mu)_(i - 1))
  (bold(x)_i - bold(mu)_(i - 1))^trp \
  &=
  bold(C)_(i - 1) + (i - 1) / i
  (((i - 1) bold(x)_i - sum_k^(i - 1) bold(x)_k) / (i - 1))
  (((i - 1) bold(x)_i - sum_k^(i - 1) bold(x)_k) / (i - 1))^trp \
  &=
  bold(C)_(i - 1) + (i - 1) / i
  ((i bold(x)_i - sum_k^i bold(x)_k) / (i - 1))
  ((i bold(x)_i - sum_k^i bold(x)_k) / (i - 1))^trp \
  &=
  bold(C)_(i - 1) + (i - 1) / i i^2 / (i - 1)^2
  (bold(x)_i - bold(mu)_i)
  (bold(x)_i - bold(mu)_i)^trp \
  &=
  bold(C)_(i - 1) + i / (i - 1)
  (bold(x)_i - bold(mu)_i)
  (bold(x)_i - bold(mu)_i)^trp \
  &=
  (
  (i - 1) sum_k^(i - 1) (bold(x)_k - bold(mu)_(i - 1)) (bold(x)_k - bold(mu)_(i - 1))^trp
  +
  i (bold(x)_i - bold(mu)_i) (bold(x)_i - bold(mu)_i)^trp
  ) / (i - 1) \
$

aside:
$
  &#hide($=$)
  (bold(x)_k - bold(mu)_i)
  (bold(x)_k - bold(mu)_i)^trp
  -
  (bold(x)_k - bold(mu)_(i - 1))
  (bold(x)_k - bold(mu)_(i - 1))^trp \
  &=
  s
$
