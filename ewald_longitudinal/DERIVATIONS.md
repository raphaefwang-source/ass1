# Derivation check: Ewald split of the screened longitudinal kernel

Notation: κ > 0 is the physical screening, ξ > 0 the Ewald parameter, k = |k_vec| the
wave number (k = 0 allowed), z = κ/(2ξ), h(z) = e^{-z²} − √π z erfc(z).
K_κ(r_vec) = e^{-κr}/r · P and P = r_vec r_vec^T/r². There is no transverse projector.

Every formula in the task was re-derived independently. **I found no discrepancy.**
The section at the end lists observations that the task does not state explicitly.
The equations marked *(extra)* are my own additions. The script uses them only as
cross-checks or diagnostics, never as primary results.

## 1. Gaussian-scale representation

Let w(s) = e^{-κ²/(4s)}/√(πs). The standard integral
∫₀^∞ s^{-1/2} e^{-a/s − bs} ds = √(π/b) e^{-2√(ab)} with a = κ²/4 and b = r² gives
∫₀^∞ w(s) e^{-sr²} ds = e^{-κr}/r.

If you differentiate the given ρ_κ, the two κ²s^{-3/2} terms cancel, which leaves
**ρ_κ'(s) = w(s)**. Also ρ_κ(0⁺) = 0, so ρ_κ(s) = ∫₀^s w > 0. Integrating by parts then gives

  ∫₀^∞ ρ_κ(s) e^{-sr²} ds = (1/r²) ∫₀^∞ w(s) e^{-sr²} ds = e^{-κr}/r³.  ✔

Numerical checks: ρ = ∫w (quad) agrees to 1e-14. The representation (quad) agrees to 9e-16.

## 2. Split, positivity, pure longitudinal structure

K_κ = r_vec r_vec^T e^{-κr}/r³, so cutting the s-integral at ξ² gives K_S + K_L = K_κ
exactly. It also gives K_{S,L} = θ_{S,L} P with

  θ_S = r² ∫_{ξ²}^∞ ρ e^{-sr²} ds,  θ_L = r² ∫₀^{ξ²} ρ e^{-sr²} ds.

Both are ≥ 0 because ρ > 0.  ✔

*(extra, closed form used as a cross-check)* Substituting ρ = ∫w and swapping the integrals:

  θ_S(r) = (1/2r)[e^{κr} erfc(ξr+z) + e^{-κr} erfc(ξr−z)] + ρ_κ(ξ²) e^{-ξ²r²}
      = e^{-ξ²r²} { e^{-z²}/(2r) [erfcx(ξr+z) + erfcx(ξr−z)] + ρ_κ(ξ²) }.

## 3. K_L near r = 0

K_L = g_L(r²) r_vec r_vec^T with g_L(u) = ∫₀^{ξ²} ρ e^{-su} ds. This is entire in u, so
K_L is C^∞ and K_L = C r_vec r_vec^T − D r² r_vec r_vec^T + …, where

  C = ∫₀^{ξ²} ρ ds = (4ξ³/(3√π)) [(1+z²) e^{-z²} − √π z (3/2 + z²) erfc z]  *(extra)*,
  D = ∫₀^{ξ²} s ρ ds.

For κ = ξ = 1: C = 0.172902 and D = 0.123082.

## 4. Real-space asymptotic

ρ_κ(ξ²) = (2ξ/√π)[e^{-z²} − √π z erfc z] = (2ξ/√π) h(z)  ✔ (substitute κ = 2ξz).
Watson's lemma at the endpoint s = ξ² gives

  θ_S = ρ_κ(ξ²) e^{-ξ²r²} [1 + c_θ/r² + O(r⁻⁴)],  c_θ = e^{-z²}/(2ξ² h)  *(extra)*.

## 5. Fourier transform (convention ∫ e^{-ik·r} K dr)

FT[x_i x_j e^{-sr²}] = −∂_{k_i}∂_{k_j} (π/s)^{3/2} e^{-k²/4s}
  = π^{3/2} [ δ_ij s^{-5/2}/2 − k_i k_j s^{-7/2}/4 ] e^{-k²/4s}.

This reproduces the task's A(k) and B(k) exactly.  ✔

- **k = 0 is finite** only because κ > 0. As s → 0, ρ_κ(s) ≈ 4 s^{3/2} e^{-κ²/4s}/(κ²√π).
  As κ → 0, A(0) = (4π/3)∫ r² θ_L dr ~ 4π/(3κ²) diverges.
- Eigenvalues: λ_⊥ = A (twice) and λ_∥ = A + B k². Since A > 0 and B < 0, the norm is
  ‖K̂_L‖₂ = max(A, |A + Bk²|).

Large k: write J_n = ∫₀^{ξ²} ρ s^{-n} e^{-k²/4s} ds. Substitute s = 1/(v + ξ⁻²), then apply Watson's lemma:

  A = (4πh/k²) e^{-k²/4ξ²} [1 + c_A/k² + …],        c_A = 2ξ²(1 − e^{-z²}/h)  *(extra)*
  B k² = −(2πh/ξ²) e^{-k²/4ξ²} [1 + O(k⁻²)]
  λ_∥ = −(2πh/ξ²) e^{-k²/4ξ²} [1 + c_N/k² + …],     c_N = 2ξ²(2 − e^{-z²}/h)  *(extra)*

So the stated A-asymptotic and the operator-norm asymptotic ‖K̂_L‖₂ ~ (2πh/ξ²) e^{-k²/4ξ²}
are both correct.  ✔

## 6. Cutoff tails

‖K_S‖₂ = θ_S. Using ∫_{rc}^∞ r² e^{-ξ²r²} dr ≈ rc e^{-ξ²rc²}/(2ξ²):

  E_S = (4√π/ξ) h rc e^{-ξ²rc²} [1 + c_ES/rc² + …],   c_ES = (1 + e^{-z²}/h)/(2ξ²)  *(extra)*  ✔

(1/2π²)∫_{kc}^∞ k² dk is the radial part of ∫_{|k|>kc} d³k/(2π)³. Using
∫_{kc}^∞ k² e^{-k²/4ξ²} dk ≈ 2ξ² kc e^{-kc²/4ξ²}:

  E_F = (2/π) h kc e^{-kc²/4ξ²} [1 + c_EF/kc² + …],   c_EF = 2ξ²(3 − e^{-z²}/h)  *(extra)*  ✔

## 7. Observations (not discrepancies, but worth knowing)

1. **K̂_L is not positive semidefinite.** The longitudinal eigenvalue A + Bk² changes sign at
   k₀ (0.8595 for κ = ξ = 1) and stays negative after that. So the large-k operator norm is |λ_∥|
   of a *negative* eigenvalue, even though θ_L ≥ 0 pointwise in real space. The stated
   A-asymptotic is subdominant: A/‖K̂_L‖ ~ 2ξ²/k².
2. **The norm has a kink** at k* where 2A + Bk² = 0 (k* = 1.3800 for κ = ξ = 1).
   Below k* the norm is A; above it the norm is −(A + Bk²). The E_F quadrature is split there.
3. **All asymptotic laws are leading order**, with relative corrections of O(r⁻²) or O(k⁻²).
   These are slow: for κ = ξ = 1, θ_S/asymptotic is 1.12 at r = 3 and 1.03 at r = 6. The
   fitted log-slopes converge faster, with slope error O(x⁻⁴).
4. **Floating-point conditioning.** The two terms of the literal ρ_κ cancel for s ≪ κ², and those of the
   literal h(z) cancel for large z. The script evaluates the algebraically identical forms
   ρ_κ = (2√s/√π) e^{-κ²/4s} p(κ/2√s) and h = e^{-z²} p(z), where p(x) = 1 − √π x erfcx(x).
   It checks them against the literal formulas.
