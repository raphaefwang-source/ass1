# Stage 1: theory-guided initialization, kappa=0.2

Only Candidate 1 (theta=0), c=1, N=256, rho=0.75 is considered. No fixed error-budget fractions are imposed, and sR/sF are independent in the free search. sqrt(log(1/epsilon))=3.034854 for epsilon=1e-4 is an initial dimensionless scale, not an accuracy certificate.

## Which theory is being used

The local Zhao-Dou-Lei Landau preprint, Section 3.3 (pp. 9-13, Theorem 3.2, equations 3.36-3.43), telescopes short-range, finite-Fourier, particle-mesh, quadrature, self and arithmetic errors. It concerns free-space Coulomb diffusion/drift coefficient norms with normalized particle weights. Those bounds are **not** directly the periodic screened longitudinal spectral-operator bound required here. Its discussion also states that nominal assignment order does not determine observed convergence order. We use its componentwise error-control approach, with this project's periodic bounds from parameter_optimizer.py. Fast DPD manuscript sections 3.2-3.5 specify the screened split, negative longitudinal Fourier eigenvalues, graph Laplacian, and same-mode mesh comparison. No Fourier eigenvalue is clipped.

## Periodic operator bounds

Let Lambda=||Gamma_D||_2, rho=N/V, Khat_L(k)=A(k)I+B(k)kk^T, and w(k)=max(|A(k)|,|A(k)+B(k)|k|^2|).

\[
B_R = \frac{2}{\Lambda}\max_i\lambda_{\max}\!\left(\sum_{j\ne i,n:\,r_{ijn}>r_c}K_S(r_{ijn})\right).
\]

RealSpaceTail evaluates the tail explicitly to Rfar=7/xi, then bounds the remaining positive Gaussian tail by its lattice-count majorant. The degree bound uses PSD short-range edge tensors (true for theta=0). It is an upper bound for the discarded graph Laplacian, not an exact error value.

\[
B_F=\frac{2\rho}{\Lambda}\sum_{|k|>k_c}w(k),\qquad
B_M=\frac{2\rho}{\Lambda}\sum_{|k|\le k_c}\alpha(k)(2+\alpha(k))w(k),
\]

\[
\alpha=C(p)\left[\max_a\frac{|k_a|h}{2\pi-|k_a|h}\right]^p,
\quad C(p)=(1+2\zeta(p))^3-1,\quad h=L/M.
\]

FourierShells sums the actual reciprocal-lattice shells and bounds its remote Gaussian remainder. eps_mesh bounds the complete finite-stencil response including transfer/aliasing after deconvolution. It requires retained modes strictly below Nyquist and stable assignment/influence modes; the production check_mesh conditions are applied separately. Excluded influence modes would contribute another tail term; candidates with exclusions are rejected as invalid here. The real/Fourier bounds assume exact coefficients and exact arithmetic. In numerical work the triangle bound is B_R+B_F+B_M plus coefficient/quadrature, far-limit, reference and floating-point remainders. The quadrature remainder below is a numerical refinement estimate, not a rigorous certificate.

The legacy optimizer uses kappa=0.7, non-unit amplitude and a preset 0.33/0.33/0.33 split with p=7. We do not reuse those physical defaults or the selection routine. A scoped adapter injects E.kernel(0.2,xi,0).rad for the bound calls and restores C.kernel afterward; no production source is changed. Its Fourier/real/mesh bound implementations are otherwise reused. Final normalization uses the newly computed kappa=0.2 Gamma_D norm. The initialization CSV is explicitly marked as using the **historical kappa=0.2 norm scale only** while the independent references are built. Lambda_lb=max_i lambda_max(Gamma_D,ii)<=Lambda can alternatively give a more conservative normalization; both are recorded for final bound diagnostics.

## Kernel and numerical coefficient checks

The physical kernel is exp(-0.2 r)/r, amplitude exactly 1. physical_kernel_audit.json checks full profiles and independent split reconstruction. coefficient_audit.json compares production A/B to a refined tau Gauss-Legendre rule (n=30, twice the grading segments, cutoff 100 rather than 80) and compares the short profile closed form to quadrature. Each production tau rule is restored after the audit. These checks are numerical estimates; they do not justify treating a theoretical sum below epsilon as the final acceptance criterion.

## Search initialization and expansion

Gaussian-tail scales sR,sF around 3 motivate coarse values 2.2-3.4, including lower values to test conservatism. Coarse xi covers 0.6-2.0, orders 3-7, and legal low meshes starting at 12. First scan equal-cutoff coarse slices, then independently scan sR/sF around fast meshes, refine all five variables near promising points, and extend any fastest boundary. The local M/p tradeoff includes 12,14,16,18,20,22,24 and larger values as needed. Theory is only a guide: no valid small-M/small-p candidate is discarded because a bound is large.

The O((xi h)^p) estimate suggests M~xi L(C_M/epsilon)^(1/p), with an unknown, configuration- and norm-dependent prefactor. It cannot specify a unique M without calibration. For kappa=0.2 the **full physical periodic kernel's** zero Fourier mode contributes rho*4*pi/(3*kappa^2)=78.539816 to the non-translation subspace; the historical full norm is about 85-90. The long-range Ewald mesh alone has A_L(0), with the complementary short-range zero-mode integral completing this physical value; independent_coefficient_checks.csv verifies that identity. This differs strongly from kappa=0.7, so previous relative errors cannot be transferred. The reference cache key includes kappa,c,rho,L,configuration and position hash; no kappa=0.7 matrix is reused.

Source paths and SHA256 values are in results/theory_sources.json. The preprint is local unpublished source material; it is not redistributed.

## Empirical scale calibration (single configuration, not acceptance)

The pilot compares the exact same-shell mesh-free and mesh operators. For each point, results/mesh_scale_calibration.csv records C_M=E_M/(xi L/M)^p, the resulting M scale for E_M=epsilon, and the minimum legal integer M from the actual retained shell and eta<1. A mesh-only predictor below that legal limit cannot be used: reducing sF may be necessary before reducing M. This is only a local predictor: C_M changes with p, h, geometry and norm. It allocates no mandatory mesh budget and does not account for cutoff error or matrix cancellation. All final M/p choices are accepted by the complete nine-configuration spectral error.
