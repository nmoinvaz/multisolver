import Mathlib.Analysis.Calculus.MeanValue

/-!
# The a-posteriori Kantorovich quantities are lower bounds

`multisolver::kantorovich` reads from each step of a damped Newton run its damping `λ`, its correction's norm `‖dz‖`
and its contraction `θ = ‖dz̄‖ / ‖dz‖`, and forms `[h] = 2 max(θ - (1 - λ), 0) / λ²` (`2θ` at `λ = 1`) and
`[ω] = [h] / ‖dz‖` (Deuflhard, Newton Methods for Nonlinear Problems, Springer 2004, sections 2.1.1 and 3.3). In
exact arithmetic, on any real normed spaces, for the correction `dz = -J(z)⁻¹ F(z)` and the simplified correction
`dz̄ = -J(z)⁻¹ F(z + λ dz)` with one inverse `A` of the Jacobian `J(z)` at the iterate:

* `simplified_correction_damped`: with `‖J(z)⁻¹ (J(z + t dz) - J(z)) dz‖ ≤ ω ‖t dz‖ ‖dz‖` for `0 ≤ t ≤ λ`,
  `‖dz̄ - (1 - λ) dz‖ ≤ (λ² / 2) ω ‖dz‖²`; Deuflhard's affine covariant Lipschitz condition on a set that holds the
  step gives that hypothesis (`simplified_correction_damped_of_affineLipschitz`);
* `simplified_correction`: at `λ = 1`, `‖dz̄‖ ≤ (ω / 2) ‖dz‖²`;
* `hEst_le` and `omegaEst_le`: for `0 < λ ≤ 1`, `[h] ≤ h = ω ‖dz‖` and `[ω] ≤ ω`.

`Controls.Kantorovich` shows each bound failing with other factors or a moved trial, and attained on a quadratic.
-/

namespace Multisolver

open Set

variable {E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [NormedAddCommGroup F] [NormedSpace ℝ F]

/-- Deuflhard's affine covariant Lipschitz condition on `D`, `‖J(x)⁻¹ (J(y) - J(x)) v‖ ≤ ω ‖y - x‖ ‖v‖`, with
`Jinv x` the inverse of `J x` -/
def AffineLipschitz (J : E → E →L[ℝ] F) (Jinv : E → F →L[ℝ] E) (D : Set E) (ω : ℝ) : Prop :=
  ∀ x ∈ D, ∀ y ∈ D, ∀ v, ‖Jinv x ((J y - J x) v)‖ ≤ ω * ‖y - x‖ * ‖v‖

/-- The damped step's simplified correction: `‖dz̄ - (1 - λ) dz‖ ≤ (λ² / 2) ω ‖dz‖²` -/
theorem simplified_correction_damped {f : E → F} {J : E → E →L[ℝ] F} {A : F →L[ℝ] E} {z dz : E} {lam ω : ℝ}
    (hlam : 0 ≤ lam) (hf : ∀ t ∈ Icc 0 lam, HasFDerivAt f (J (z + t • dz)) (z + t • dz))
    (hA : ∀ v, A (J z v) = v) (hdz : J z dz = -f z)
    (hω : ∀ t ∈ Icc 0 lam, ‖A ((J (z + t • dz) - J z) dz)‖ ≤ ω * ‖t • dz‖ * ‖dz‖) :
    ‖-A (f (z + lam • dz)) - (1 - lam) • dz‖ ≤ lam ^ 2 / 2 * ω * ‖dz‖ ^ 2 := by
  -- g(t) = A F(z + t dz) - A F(z) - t dz: g(0) = 0 and g'(t) = A (J(z + t dz) - J(z)) dz, at most ω t ‖dz‖²
  set g : ℝ → E := fun t => A (f (z + t • dz)) - A (f z) - t • dz with hg
  have hseg : ∀ t : ℝ, HasDerivAt (fun t : ℝ => z + t • dz) dz t := fun t => by
    simpa using ((hasDerivAt_id' t).smul_const dz).const_add z
  have hderiv : ∀ t ∈ Icc 0 lam, HasDerivAt g (A ((J (z + t • dz) - J z) dz)) t := by
    intro t ht
    have h0 : HasDerivAt (fun t : ℝ => f (z + t • dz)) (J (z + t • dz) dz) t :=
      (hf t ht).comp_hasDerivAt t (hseg t)
    have h1 : HasDerivAt (fun t : ℝ => A (f (z + t • dz))) (A (J (z + t • dz) dz)) t :=
      A.hasFDerivAt.comp_hasDerivAt t h0
    have h2 := (h1.sub_const (A (f z))).sub ((hasDerivAt_id' t).smul_const dz)
    convert h2 using 1
    simp [hA]
  have hB : ∀ t : ℝ, HasDerivAt (fun t : ℝ => ω * ‖dz‖ ^ 2 / 2 * (t * t)) (ω * ‖dz‖ ^ 2 * t) t := fun t => by
    convert ((hasDerivAt_id' t).mul (hasDerivAt_id' t)).const_mul (ω * ‖dz‖ ^ 2 / 2) using 1
    ring
  have hle := image_norm_le_of_norm_deriv_right_le_deriv_boundary
    (fun t ht => (hderiv t ht).continuousAt.continuousWithinAt)
    (fun t ht => (hderiv t (Ico_subset_Icc_self ht)).hasDerivWithinAt)
    (by simp [g]) hB
    (fun t ht => by
      have ht' := Ico_subset_Icc_self ht
      calc ‖A ((J (z + t • dz) - J z) dz)‖ ≤ ω * ‖t • dz‖ * ‖dz‖ := hω t ht'
        _ = ω * ‖dz‖ ^ 2 * t := by rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg ht'.1]; ring)
    (right_mem_Icc.mpr hlam)
  -- A F(z) = -dz, so the left side is ‖g(λ)‖
  have hfz : A (f z) = -dz := by rw [show f z = -J z dz by rw [hdz, neg_neg], map_neg, hA]
  have heq : -A (f (z + lam • dz)) - (1 - lam) • dz = -g lam := by
    simp only [g, hfz, sub_smul, one_smul]
    abel
  rw [heq, norm_neg]
  calc ‖g lam‖ ≤ ω * ‖dz‖ ^ 2 / 2 * (lam * lam) := hle
    _ = lam ^ 2 / 2 * ω * ‖dz‖ ^ 2 := by ring

/-- `simplified_correction_damped` from Deuflhard's condition on a set that holds the segment `z + t dz`,
`0 ≤ t ≤ λ` -/
theorem simplified_correction_damped_of_affineLipschitz {f : E → F} {J : E → E →L[ℝ] F} {Jinv : E → F →L[ℝ] E}
    {D : Set E} {z dz : E} {lam ω : ℝ} (hL : AffineLipschitz J Jinv D ω) (hlam : 0 ≤ lam)
    (hD : ∀ t ∈ Icc 0 lam, z + t • dz ∈ D) (hf : ∀ t ∈ Icc 0 lam, HasFDerivAt f (J (z + t • dz)) (z + t • dz))
    (hA : ∀ v, Jinv z (J z v) = v) (hdz : J z dz = -f z) :
    ‖-Jinv z (f (z + lam • dz)) - (1 - lam) • dz‖ ≤ lam ^ 2 / 2 * ω * ‖dz‖ ^ 2 := by
  have hz : z ∈ D := by simpa using hD 0 (left_mem_Icc.mpr hlam)
  refine simplified_correction_damped hlam hf hA hdz fun t ht => ?_
  simpa using hL z hz (z + t • dz) (hD t ht) dz

/-- The full step's simplified correction: `‖dz̄‖ ≤ (ω / 2) ‖dz‖²` -/
theorem simplified_correction {f : E → F} {J : E → E →L[ℝ] F} {A : F →L[ℝ] E} {z dz : E} {ω : ℝ}
    (hf : ∀ t ∈ Icc (0 : ℝ) 1, HasFDerivAt f (J (z + t • dz)) (z + t • dz))
    (hA : ∀ v, A (J z v) = v) (hdz : J z dz = -f z)
    (hω : ∀ t ∈ Icc (0 : ℝ) 1, ‖A ((J (z + t • dz) - J z) dz)‖ ≤ ω * ‖t • dz‖ * ‖dz‖) :
    ‖-A (f (z + dz))‖ ≤ ω / 2 * ‖dz‖ ^ 2 := by
  have h := simplified_correction_damped zero_le_one hf hA hdz hω
  simp only [one_smul, sub_self, zero_smul, sub_zero, one_pow] at h
  linarith

/-- `[h]` of a step of damping `lam` and contraction `θ`, as `kantorovich::from_steps` forms it -/
noncomputable def hEst (lam θ : ℝ) : ℝ :=
  if lam = 1 then 2 * θ else 2 * max (θ - (1 - lam)) 0 / (lam * lam)

/-- `[ω] = [h] / ‖dz‖` of a step of damping `lam`, contraction `θ` and correction's norm `d` -/
noncomputable def omegaEst (lam θ d : ℝ) : ℝ :=
  hEst lam θ / d

/-- `[h] ≤ h = ω ‖dz‖`, with `θ = ‖dz̄‖ / ‖dz‖` -/
theorem hEst_le {f : E → F} {J : E → E →L[ℝ] F} {A : F →L[ℝ] E} {z dz : E} {lam ω : ℝ}
    (hlam0 : 0 < lam) (hlam1 : lam ≤ 1) (hω0 : 0 ≤ ω) (hdz0 : dz ≠ 0)
    (hf : ∀ t ∈ Icc 0 lam, HasFDerivAt f (J (z + t • dz)) (z + t • dz))
    (hA : ∀ v, A (J z v) = v) (hdz : J z dz = -f z)
    (hω : ∀ t ∈ Icc 0 lam, ‖A ((J (z + t • dz) - J z) dz)‖ ≤ ω * ‖t • dz‖ * ‖dz‖) :
    hEst lam (‖-A (f (z + lam • dz))‖ / ‖dz‖) ≤ ω * ‖dz‖ := by
  have hd0 : 0 < ‖dz‖ := norm_pos_iff.mpr hdz0
  have key := simplified_correction_damped hlam0.le hf hA hdz hω
  set dzbar := -A (f (z + lam • dz))
  -- the triangle inequality: ‖dz̄‖ ≤ (1 - λ) ‖dz‖ + (λ² / 2) ω ‖dz‖²
  have tri : ‖dzbar‖ ≤ (1 - lam) * ‖dz‖ + lam ^ 2 / 2 * ω * ‖dz‖ ^ 2 := by
    calc ‖dzbar‖ = ‖(dzbar - (1 - lam) • dz) + (1 - lam) • dz‖ := by rw [sub_add_cancel]
      _ ≤ ‖dzbar - (1 - lam) • dz‖ + ‖(1 - lam) • dz‖ := norm_add_le _ _
      _ ≤ lam ^ 2 / 2 * ω * ‖dz‖ ^ 2 + (1 - lam) * ‖dz‖ := by
        rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (by linarith)]
        linarith
      _ = _ := by ring
  have hθ : ‖dzbar‖ / ‖dz‖ - (1 - lam) ≤ lam ^ 2 / 2 * ω * ‖dz‖ := by
    have : ((1 - lam) + lam ^ 2 / 2 * ω * ‖dz‖) * ‖dz‖ = (1 - lam) * ‖dz‖ + lam ^ 2 / 2 * ω * ‖dz‖ ^ 2 := by
      ring
    have h := (div_le_iff₀ hd0).mpr (tri.trans this.ge)
    linarith
  unfold hEst
  split_ifs with h1
  · subst h1
    linarith
  · rw [div_le_iff₀ (by positivity)]
    have hm : max (‖dzbar‖ / ‖dz‖ - (1 - lam)) 0 ≤ lam ^ 2 / 2 * ω * ‖dz‖ := max_le hθ (by positivity)
    nlinarith

/-- `[ω] ≤ ω` -/
theorem omegaEst_le {f : E → F} {J : E → E →L[ℝ] F} {A : F →L[ℝ] E} {z dz : E} {lam ω : ℝ}
    (hlam0 : 0 < lam) (hlam1 : lam ≤ 1) (hω0 : 0 ≤ ω) (hdz0 : dz ≠ 0)
    (hf : ∀ t ∈ Icc 0 lam, HasFDerivAt f (J (z + t • dz)) (z + t • dz))
    (hA : ∀ v, A (J z v) = v) (hdz : J z dz = -f z)
    (hω : ∀ t ∈ Icc 0 lam, ‖A ((J (z + t • dz) - J z) dz)‖ ≤ ω * ‖t • dz‖ * ‖dz‖) :
    omegaEst lam (‖-A (f (z + lam • dz))‖ / ‖dz‖) ‖dz‖ ≤ ω := by
  rw [omegaEst, div_le_iff₀ (norm_pos_iff.mpr hdz0)]
  exact hEst_le hlam0 hlam1 hω0 hdz0 hf hA hdz hω

end Multisolver
