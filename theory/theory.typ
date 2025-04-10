// #set page(margin: 1cm)
#set text(font: "Atkinson Hyperlegible Next")
// #set text(size: 20pt)
#show math.equation: set text(font: "Lete Sans Math")
#show math.equation: set block(breakable: true)
#set heading(numbering: "1.1")
#set enum(indent: 1em)

#let trp = $sans(upright(T))$

#text(
  2em,
  weight: "bold",
)[Some theoretical notes on \ implementing rigid registration]
#line(length: 75%)

= Kernel Correlation

We are given target points $bold(x)_i in RR^d$ with weights $q_i in RR$ and
source points $bold(y)_j in RR^d$ with weights $p_j in RR$.

The MM-algorithm consists of repeating these steps:
+ Compute
  $ w_(i j) = phi_sigma (norm(bold(x)_i - bold(R) bold(y)_j - bold(t))) $
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
  &approx
  sum_(i j) phi_(i k_j) / (sum_(i' j') q_i' p_j' phi_(i' k_j')) q_i p_j bold(x)_i
  =
  (sum_(i j) phi_(i k_j) q_i p_j bold(x)_i) / (sum_(i j) q_i p_j phi_(i k_j)) \
  &=
  (sum_j p_j sum_i phi_(i k_j) q_i bold(x)_i) / (sum_j p_j sum_i phi_(i k_j) q_i)
  =
  (sum_j p_j tilde(bold(x))_k_j) / (sum_j p_j tilde(q)_k_j)
$
where
$
  tilde(bold(x))_k = sum_i phi_(i k) q_i bold(x)_i,
  quad quad
  tilde(q)_k = sum_i phi_(i k) q_i.
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
Then, in every MM-iteration:
+ Compute $k_j$ by placing every $bold(R) bold(y)_j + bold(t)$ on $cal(G)$.
+ Compute
  $macron(bold(x)) = sum_j p_j tilde(bold(x))_k_j$,
  $macron(bold(y)) = sum_j p_j tilde(q)_k_j bold(y)_j$, and
  $Z = sum_j p_j tilde(q)_k_j$.
+ Divide $macron(bold(x))$ and $macron(bold(y))$ by $Z$.
+ Compute
  $
    bold(S) = sum_j p_j (tilde(bold(x))_k_j - tilde(q)_k_j macron(bold(x))) (bold(y)_j - macron(bold(y)))^trp
  $

Note that the first two steps can be performed in a single pass over all source
points.
A second pass is then needed for $bold(S)$.
