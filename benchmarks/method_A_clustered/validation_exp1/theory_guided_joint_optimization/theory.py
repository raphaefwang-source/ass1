"""Use existing parameter bounds with an explicitly scoped kappa=.2, c=1 kernel."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import experiment as X
from experiment import E,C,np
import parameter_optimizer as P
from verify_ewald_longitudinal import graded_rule
from scipy.integrate import quad

@contextmanager
def physical_bound_kernel():
    # P's legacy default C.kernel is kappa=.7 with a different amplitude.
    # Inject the actual theta=0 RadialEwald only while evaluating bounds, then
    # restore the shared function. No production source or kernel cache changes.
    original=P.C.kernel
    P.C.kernel=lambda xi:E.kernel(X.KAPPA,xi,0.).rad
    try:
        assert P.C.kernel(1.).kap==.2 and P.C.kernel(1.).amp==1.
        yield
    finally:
        P.C.kernel=original

def fourier_shells(xi,L):
    with physical_bound_kernel(): return P.FourierShells(xi,L)

def coefficient_audit(xi,L,kc,M,p):
    fs=fourier_shells(xi,L)
    with physical_bound_kernel():
        estimate,relative=P.eps_quad_estimate(xi,L,kc,M,p,.75,fs)
        radial=P.theta_S_closed_vs_quad(xi,.5,8/xi)
    return dict(quadrature_bound_estimate_abs=estimate,coefficient_relative_max=relative,
                short_profile_closed_vs_quadrature=radial)

def independent_coefficient_checks():
    rows=[]
    for xi in (.6,1.,1.4,2.):
        K=E.kernel(.2,xi,0.)
        for k in (0.,2*np.pi/X.configs()[0]['L'],3.,6.):
            # Independent adaptive s-integral, with no production tau rule.
            def integral(power):
                return quad(lambda s:E.rho_density(s,.2)*s**(-power)*np.exp(-k*k/(4*s)),
                            0.,xi*xi,epsabs=1e-12,epsrel=5e-13,limit=300)[0]
            ai=np.pi**1.5/2*integral(2.5)
            bi=-np.pi**1.5/4*integral(3.5)
            ap,bp=K.AB(np.array([k]));ap,bp=float(ap[0]),float(bp[0])
            relative=max(abs(ap-ai)/max(abs(ai),1e-300),abs(bp-bi)/max(abs(bi),1e-300))
            assert relative<1e-10
            row=dict(xi=xi,k=k,adaptive_A=ai,adaptive_B=bi,production_A=ap,production_B=bp,
                     max_relative_difference=relative,kappa=K.kap,c=K.amp)
            if k==0.:
                short_zero=4*np.pi/3*quad(lambda r:r*r*float(K.theta_S(np.array([r]))[0]),
                                          0.,12/xi,epsabs=1e-12,epsrel=5e-13)[0]
                full_zero=4*np.pi/(3*.2**2)
                row.update(short_zero_mode=short_zero,full_zero_mode=full_zero,
                           zero_mode_split_relative_error=abs(ai+short_zero-full_zero)/full_zero)
                assert row['zero_mode_split_relative_error']<1e-11
            rows.append(row)
    X.write(X.RESULTS/'independent_coefficient_checks.csv',rows)

def main():
    cfg=X.configs()[0]
    previous=next(r for r in E.read_csv(X.HERE.parent/'accuracy_sweeps/results/references.csv')
                  if r['config']==cfg['name'] and r['kappa']==.2 and r['rho']==.75)
    # Historical norm is an initialization scale only, never a new error result.
    denominator=previous['norm2']
    rows=[]
    rst=P.RealSpaceTail(cfg['q'],cfg['L'],7/.6)
    for xi in (.6,.8,1.,1.2,1.4,1.6,1.8,2.):
        with physical_bound_kernel():
            prep=rst.prepare(xi)
            fs=P.FourierShells(xi,cfg['L'])
            for s in (2.2,2.6,3.,float(np.sqrt(np.log(1/X.TOL))),3.4,4.):
                par=X.pars(xi,s,s,24,5)
                rc,kc,shell=E.cutoffs(par,cfg['L'])
                br=rst.eps_S(prep,rc)/denominator
                bf=fs.eps_F(kc,.75)/denominator
                for M,p in ((12,3),(16,3),(16,4),(20,5),(24,7),(32,7)):
                    bm=P.eps_mesh(fs,kc,M,p,.75)/denominator
                    rows.append(dict(xi=xi,sR=s,sF_req=s,sF_eff=kc/(2*xi),M=M,p=p,
                                     B_R=br,B_F=bf,B_M=bm,B_sum=br+bf+bm,
                                     denominator=denominator,denominator_source='historical kappa=.2 direct norm, initialization only'))
    X.write(X.RESULTS/'theory_initialization.csv',rows)
    audits=[]
    for xi in (.6,1.,1.4,2.):
        rc,kc,_=E.cutoffs(X.pars(xi,3.1,3.1,40,5),cfg['L'])
        audits.append(dict(xi=xi,**coefficient_audit(xi,cfg['L'],kc,40,5)))
    X.jsave(X.RESULTS/'coefficient_audit.json',audits)
    independent_coefficient_checks()
    sources={}
    for p in (Path('/Users/wfy666/Downloads/fast_particle_Landau_preprint.pdf'),
              Path('/Users/wfy666/Documents/ass1/manuscript/main.tex'),
              X.HERE.parent.parent/'parameter_optimizer.py',X.HERE.parent/'ewald_exp_lib.py'):
        sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    X.jsave(X.RESULTS/'theory_sources.json',sources)
    text=r'''# Stage 1: theory-guided initialization, kappa=0.2

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
'''
    pilot=X.read(X.RESULTS/'pilot_error_decomposition.csv')
    if pilot:
        calibration=[]
        for r in pilot:
            cid=r['candidate_id']
            import re
            m=re.fullmatch(r'xi([\d.]+)_R([\d.]+)_F([\d.]+)_M(\d+)_p(\d+)',cid)
            xi,sr,sf,M,p=map(float,m.groups());M=int(M);p=int(p)
            scale=xi*cfg['L']/M
            empirical=r['E_M']/scale**p
            rc,kc,_=E.cutoffs(X.pars(xi,sr,sf,M,p),cfg['L'])
            mc=int(np.floor(kc*cfg['L']/(2*np.pi)+1e-12))
            legal=max(2*mc+1,int(np.floor(kc*cfg['L']/np.pi))+1,p)
            calibration.append(dict(candidate_id=cid,config=r['config'],xi=xi,M=M,p=p,
                                    E_M=r['E_M'],xi_h=scale,empirical_C_M=empirical,
                                    predicted_M_for_E_M_equals_target=xi*cfg['L']*(empirical/X.TOL)**(1/p),
                                    minimum_legal_integer_M=legal))
        X.write(X.RESULTS/'mesh_scale_calibration.csv',calibration)
        text+='\n## Empirical scale calibration (single configuration, not acceptance)\n\n'
        text+='The pilot compares the exact same-shell mesh-free and mesh operators. For each point, results/mesh_scale_calibration.csv records C_M=E_M/(xi L/M)^p, the resulting M scale for E_M=epsilon, and the minimum legal integer M from the actual retained shell and eta<1. A mesh-only predictor below that legal limit cannot be used: reducing sF may be necessary before reducing M. This is only a local predictor: C_M changes with p, h, geometry and norm. It allocates no mandatory mesh budget and does not account for cutoff error or matrix cancellation. All final M/p choices are accepted by the complete nine-configuration spectral error.\n'
    (X.HERE/'stage1_theory.md').write_text(text)
    X.log('theoretical initialization and coefficient audits complete')

if __name__=='__main__': main()
