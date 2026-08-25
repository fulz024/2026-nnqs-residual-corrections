# Residual-correction method

Let $\mathcal S$ be the frozen retained determinant support and let $c$ be
the exported amplitudes on it. After normalization, $x=c/\lVert c\rVert$,
the restricted energy is

\[
E_{\mathcal S}=x^\dagger H_{\mathcal SS}x.
\]

This is a variational energy for the supplied finite vector. It is not a
second stochastic estimate of the original VMC expectation.

## The full residual

The retained-space and external residuals are

\[
r_{\mathcal S}=(H_{\mathcal SS}-E_{\mathcal S})x,
\qquad
r_a=\sum_{i\in\mathcal S}H_{ai}x_i,
\quad a\notin\mathcal S.
\]

Roundoff along $x$ is removed from $r_{\mathcal S}$. The external sum must
be coherent: all source determinants coupled to the same (a) are summed
before $\lvert r_a\rvert^2$ is formed. In the distributed implementation,
each $a$ has a deterministic hash owner that performs this reduction.

## Projected internal correction

When the exported vector is not stationary inside $\mathcal S$, an
external-only EN correction omits part of the residual. Define

\[
Q_x=I-\lvert x\rangle\langle x\rvert,
\qquad
D_{\mathcal S}=\operatorname{diag}(H_{\mathcal SS}).
\]

The internal first-order response $z_{\mathcal S}$ is the constrained
diagonal solve

\[
Q_x(D_{\mathcal S}-E_{\mathcal S})Q_x z_{\mathcal S}
=-r_{\mathcal S},
\qquad
\langle x\vert z_{\mathcal S}\rangle=0.
\]

Writing $d_i=E_{\mathcal S}-H_{ii}$, this is evaluated without a matrix
inverse as

\[
(z_{\mathcal S})_i=\frac{(r_{\mathcal S})_i+\lambda x_i}{d_i},
\qquad
\lambda=-\frac{\sum_i x_i^*(r_{\mathcal S})_i/d_i}
{\sum_i\lvert x_i\rvert^2/d_i}.
\]

The internal contribution is

\[
\Delta_{\mathcal S}^{(2)}
=\operatorname{Re}\langle r_{\mathcal S}\vert z_{\mathcal S}\rangle.
\]

The rank-one projection matters: determinant rows within $\mathcal S$ are
not a basis for the orthogonal complement of (x).

## PT2 and rPT2

The external response and correction are

\[
t_a=\frac{r_a}{E_{\mathcal S}-H_{aa}},
\qquad
\Delta_{\rm ext}^{(2)}=
\sum_{a\notin\mathcal S}\frac{\lvert r_a\rvert^2}
{E_{\mathcal S}-H_{aa}}.
\]

The full-residual report is

\[
\Delta^{(2)}=\Delta_{\mathcal S}^{(2)}+\Delta_{\rm ext}^{(2)},
\qquad
E_{\rm PT2}=E_{\mathcal S}+\Delta^{(2)}.
\]

With

\[
q=\lVert z_{\mathcal S}\rVert^2+\sum_a\lvert t_a\rvert^2,
\]

the renormalized report is

\[
\Delta_{\rm rPT2}=\frac{\Delta^{(2)}}{1+q},
\qquad
E_{\rm rPT2}=E_{\mathcal S}+\Delta_{\rm rPT2}.
\]

## Diagonal Brillouin--Wigner report

For a trial energy $\omega$, repeat the same projected internal solve and
external diagonal response using denominators $\omega-H_{II}$. Their sum
defines the diagonal self-energy $\Sigma_D(\omega)$. dBW solves the scalar
equation

\[
\Delta_{\rm dBW}=\Sigma_D(E_{\mathcal S}+\Delta_{\rm dBW}).
\]

Because

\[
\Sigma_D'(\omega)=-\lVert t_D(\omega)\rVert^2,
\]

the first Newton step from $\Delta=0$ is rPT2. Later iterations reuse the
same owner-local residuals and diagonals and require only scalar reductions.
Failure of this one-dimensional solve is reported explicitly rather than
silently substituted by another root.

## Scope

The released kernel supports at most 64 spin orbitals, covering the article's
systems. It enumerates every nonzero Slater--Condon single and double
connection from every retained determinant. Thus restricted
$E_{\mathcal S}$ and the external residual use the same complete Hamiltonian
connectivity, and “PT2” has one unambiguous meaning throughout
the repository.
