# B2: structural PSD, periodic table, and accuracy assumptions

This implementation follows the positive-edge construction of §4.2, equations (27)–(30), of the supplied *Fast conservative DPD* guide (PDF page 8). The document is mathematical source material; the work requested is the B2 implementation and the two experiments. No instructions embedded in the document govern execution.

## Physical model and legal inputs

The physical model is unchanged: `c=1`, `kappa>0`,

$$
 K(r)=c\,e^{-\kappa|r|}|r|^{-3}rr^T,
 \quad K_L(r)=c\int_0^{\xi^2}\rho_\kappa(s)rr^Te^{-s|r|^2}\,ds,
 \quad K_S=K-K_L.
$$

Here `rho_kappa(s)=integral_0^s exp(-kappa²/(4u))/sqrt(pi*u) du >= 0`. This is the original sharp Candidate 1 split; `RadialEwald`, the closed short-range weights, periodic neighbor enumeration and cached short-range action are reused unchanged. Legal inputs have finite `L,kappa,xi,rc,sT>0`, finite `c>=0`, integers `M>=2`, `1<=p<=M`, and positive integer quadrature order. Distinct particles cannot coincide modulo the box, since the original full kernel is singular there. No minimum separation, density restriction, Fourier positivity, or spectral-gap assumption is used in the proof.

## Positive finite table, including construction error

Let `h=L/M`, `R_T=sT/kappa`. For each modular grid displacement `ell`, use its canonical displacement `t` and all physical image vectors with `|t_a+n_a L|<=R_T` in each coordinate. This is a **cube of physical displacements**, not a fixed list of image indices or a minimum-image approximation. Replacing `t` by `t+mL` just relabels the integer image sum. The set is closed under global inversion. At a Nyquist coordinate `L/2`, the images pair as `n <-> -n-1`.

Choose *any* nodes `0<s_q<=xi²` and coefficients `a_q>=0`. The production rule is positive composite Gauss–Legendre in `log(s)` on `[s_min,xi²]`, including the Jacobian and positive density in `a_q`; `s_min=min(xi²/2,kappa²/256)`. Define

$$
 T_\ell=c\sum_q a_q\sum_{n:\,|t_a+n_aL|\le R_T}
 (t+nL)(t+nL)^T e^{-s_q|t+nL|^2}.
$$

For every `z`, `z^T T_ell z=c sum a_q sum exp(-s_q |r|²)(z.r)²>=0`. Also `T_-ell=T_ell=T_ell^T`. These statements hold at **every finite image cutoff and quadrature order**. Image truncation, omitted small scales and quadrature approximation cannot introduce negative edges. No eigenvalue clipping, diagonal shift, analytic Fourier inversion, deconvolution, or fitted table entries are used.

The separable construction is exactly the same finite sum. In coordinate `a`, set `A0=sum_n e^{-s r_a²}`, `m_a=sum_n r_a e^{-s r_a²}/A0`, `v_a=sum_n (r_a-m_a)² e^{-s r_a²}/A0>=0`. Its integrand is

$$
 A_{0x}A_{0y}A_{0z}\,[mm^T+\operatorname{diag}(v_x,v_y,v_z)].
$$

Zero `A0` contributes zero. Reflection sets the means at zero and Nyquist to zero; this is also the positive average of reflected image distributions. Variances are accumulated as sums of squares, without subtracting nearly equal moments. The six entries are accumulated from these factors. Taking the real part of the FFT is equivalent, in exact arithmetic, to averaging the table under inversion, another PSD-preserving operation.

## Nonnegative interpolation and graph proof

The tensor product of centered order-`p` cardinal B-splines gives `w_ig>=0`, `sum_g w_ig=1`, with all grid indices reduced modulo `M`. The recurrence has nonnegative coefficients wherever its sub-splines are supported. The implementation normalizes each one-dimensional stencil by its positive sum and uses the **identical indices and weights** for spreading and gathering. Wrapped stencil entries are accumulated, including if several entries address the same grid site.

Define

$$
 L_{ij}=\sum_{g,h}w_{ig}T_{g-h}w_{jh}.
$$

It is a nonnegative linear combination of PSD 3x3 tensors, so `L_ij>=0` in Loewner order for all positions and all legal parameters. `L_ji=L_ij^T=L_ij`. Retained short edges are `S_ij=sum_n theta_S(r) rr^T/|r|²>=0`. With `A_ij=S_ij+L_ij`,

$$
 X^T\Gamma_hX=\frac12\sum_{i\ne j}(X_i-X_j)^TA_{ij}(X_i-X_j)\ge0.
$$

Thus the **complete graph** is symmetric PSD and annihilates all three translation fields. This is the proof; sampled eigenvalues are regression diagnostics only. Additional null modes are allowed. The tensor Fourier multipliers need not be PSD and are not used as a covariance square root.

## FFT normalization, zero mode and self cancellation

The forward FFT is unnormalized and inverse FFT divides by `M³`. Hence `ifft(fft(T)*fft(source))` is the cyclic **discrete sum** `sum_h T_(g-h) source_h`; no `h^-3` or cell volume is applied. All `M³` modes are retained. In particular `T_hat(0)=sum_ell T_ell` is retained exactly as built; it is neither set to zero nor overwritten by a continuum coefficient.

For `n_g=sum_j w_jg`, cache `Dtilde_i=sum_g w_ig (T*n)_g`. At every action, calculate `H_i(X)=sum_g w_ig (T*m)_g`, `m_g=sum_j w_jg X_j`, and return `Dtilde_i X_i-H_i(X)`. Both terms contain **the identical** `L_ii X_i`, which cancels algebraically. This removes the particle's entire interpolated self response, including all its periodic images. One must not separately subtract a free-space `K_L(0)`, zero out `T_0`, or subtract only the central image: `T_0` contains genuine interactions between distinct particles assigned to the same grid sites.

Positions, stencil arrays and the table must remain fixed while their degree is cached. Rebuild the operator after changing positions or physical/mesh parameters; shared table arrays must not be mutated between degree construction and an action. This is part of the implementation contract.

## Accuracy bounds and the limits of floating point

The theorem is about exact arithmetic with the stored nonnegative factors and identical adjoint operations. Ordinary float64 summation, table accumulation and FFTs introduce roundoff; no claim of bitwise positive energies for arbitrarily tiny vectors is made. Checks quantify symmetry, translation residual, dense/direct-stencil versus FFT action, and negative eigenvalues on the translation complement. A PSD claim based only on passing those checks would be insufficient; the sum-of-squares construction above supplies it. Nonfinite inputs, invalid parameters and nonpositive/nonfinite density evaluations are rejected.

For an entrywise table perturbation `||delta T_ell||_2<=delta`, normalized positive weights give `||delta L_ij||<=delta` and `||delta Gamma||_2<=2(N-1)delta`. An arbitrary signed table approximation would therefore only imply `Gamma>=-2(N-1)delta I`; B2 avoids that loss by defining every approximate entry through positive factors.

The omitted outside-cube images are bounded by the outside-sphere physical tail. With `a=sqrt(3)L/2`, `I0=e^-kR/k`, `I1=I0(R+1/k)`, `I2=I0(R²+2R/k+2/k²)`,

$$
 B_T=\frac{4\pi c}{3L^3}(\kappa+1/R_T)
 [I_2+3aI_1+(3a^2+a^3/R_T)I_0]
$$

uniformly bounds each discarded table tensor in spectral norm. This follows from the lattice counting majorant `4pi(r+a)^3/(3L³)`, summation by parts dropping its negative boundary term, and `1/r<=1/R_T`. `2(N-1)B_T/||Gamma_D||` is a conservative relative graph bound.

The missing interval `[0,s_min]` in the retained cube has uniform tensor error at most

$$
 B_0\le c\,(2\lfloor R_T/L\rfloor+3)^3\,3R_T^2
 \frac{4s_{\min}^{3/2}}{3\sqrt\pi}
 e^{-\kappa^2/(4s_{\min})}.
$$

This uses `rho(s)<=2sqrt(s)/sqrt(pi)*exp(-kappa²/(4s))` and `|r|²<=3R_T²`. Quadrature convergence is checked by doubling the GL order against direct image/positive-integral evaluations. That difference is a **numerical estimate**, not a certified quadrature bound. The measured `E_Gamma` is also a spectral norm computed in float64 against the independently certified direct-sum reference, not a rigorous interval certificate of the tolerance. These accuracy qualifications do not qualify the structural PSD theorem.

For a smooth infinite-periodic table, `p>=2` reproduces constants and linear functions, so Taylor expansion of the two positive averages gives `O(h²)`. For `p>=3` each coordinate's second central moment is `p h²/12`, yielding leading bias `(p h²/12) Delta K_L`. For `p=2` the phase-dependent moment is `alpha(1-alpha)h²`, again second order. `p=1` generally has first-order error. Increasing `p` therefore does **not** give `O(h^p)` here and can increase the leading bias. A finite hard table cutoff has jumps; high `sT` makes their size negligible on the measured convergence range. Experiment 2 uses measured mesh coefficients and measured complete errors, with full validation of all nine configurations.
