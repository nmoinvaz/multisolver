import Multisolver.Kantorovich

/-!
# Controls of the Kantorovich bounds

The bounds of `Multisolver.Kantorovich` ask that the correction and the simplified correction share the inverse of
the Jacobian at the iterate, and that the trial be `z + λ dz`. On `ℝ`, with `F(x) = x - 1`, whose `ω` is 0, each
fails without them:

* `approximate_factors`: the factor 2 in place of the Jacobian 1, from `x = 2`, gives `[ω] = 2` (the example of
  `kantorovich`'s documentation);
* `moved_trial`: the exact factor, with the trial `x = 1` moved to `3/2`, gives `[ω] = 1`.

`damped_read_as_full`: on `F(x) = x`, a step of `λ = 1/2` read as a full step gives `[h] = 1` against `h = 0`, where
`hEst`'s damped form gives 0. `attained`: on `F(x) = x² - 2` from `x = 2`, where `ω = 1/2` along the step, every
hypothesis of `omegaEst_le` holds and `[ω] = ω`, so no constant below the `ω / 2` of `simplified_correction` holds.
-/

namespace Controls

open Set Multisolver

/-- `F(x) = x - 1` -/
def f₁ (x : ℝ) : ℝ := x - 1

/-- its Jacobian, 1 everywhere -/
def J₁ (_ : ℝ) : ℝ →L[ℝ] ℝ := ContinuousLinearMap.id ℝ ℝ

theorem hasFDerivAt_f₁ (x : ℝ) : HasFDerivAt f₁ (J₁ x) x :=
  (hasFDerivAt_id x).sub_const 1

/-- `ω = 0` for `F(x) = x - 1`, with any factor `A` -/
theorem lipschitz_f₁ (A : ℝ →L[ℝ] ℝ) (z y v : ℝ) : ‖A ((J₁ y - J₁ z) v)‖ ≤ 0 * ‖y - z‖ * ‖v‖ := by
  simp [J₁]

/-- the factor 2's inverse, in place of the Jacobian 1's -/
noncomputable def A₂ : ℝ →L[ℝ] ℝ := (2⁻¹ : ℝ) • ContinuousLinearMap.id ℝ ℝ

/-- the correction from `x = 2` with the factor 2 -/
noncomputable def dz₂ : ℝ := -A₂ (f₁ 2)

/-- With the factor 2, `A₂` is no inverse of `J₁ 2`, and `[ω] = 2 > ω = 0` -/
theorem approximate_factors :
    ¬(∀ v, A₂ (J₁ 2 v) = v) ∧ omegaEst 1 (‖-A₂ (f₁ (2 + (1 : ℝ) • dz₂))‖ / ‖dz₂‖) ‖dz₂‖ = 2 := by
  refine ⟨fun h => by have := h 1; norm_num [A₂, J₁] at this, ?_⟩
  norm_num [omegaEst, hEst, A₂, f₁, dz₂, abs_of_pos]

/-- With the exact factor and the correction `-1` from `x = 2`, the trial `1` moved to `3/2`: `[ω] = 1 > ω = 0` -/
theorem moved_trial :
    J₁ 2 (-1) = -f₁ 2 ∧ omegaEst 1 (‖-ContinuousLinearMap.id ℝ ℝ (f₁ (3 / 2))‖ / ‖(-1 : ℝ)‖) ‖(-1 : ℝ)‖ = 1 := by
  norm_num [omegaEst, hEst, f₁, J₁]

/-- On `F(x) = x` from `x = 1`, the correction `-1` and the step `λ = 1/2`: `[h]` is 0 in the damped form and 1 read
as a full step, where `h = 0` -/
theorem damped_read_as_full :
    hEst (1 / 2) (‖-((1 : ℝ) + (1 / 2 : ℝ) • (-1))‖ / ‖(-1 : ℝ)‖) = 0 ∧
      hEst 1 (‖-((1 : ℝ) + (1 / 2 : ℝ) • (-1))‖ / ‖(-1 : ℝ)‖) = 1 := by
  norm_num [hEst]

/-- `F(x) = x² - 2` -/
def f₂ (x : ℝ) : ℝ := x * x - 2

/-- its Jacobian, `2x` -/
noncomputable def J₂ (x : ℝ) : ℝ →L[ℝ] ℝ := (2 * x) • ContinuousLinearMap.id ℝ ℝ

theorem hasFDerivAt_f₂ (x : ℝ) : HasFDerivAt f₂ (J₂ x) x := by
  have h := ((hasDerivAt_id' x).mul (hasDerivAt_id' x)).sub_const 2
  have hJ : J₂ x = ContinuousLinearMap.toSpanSingleton ℝ (1 * x + x * 1) := by
    ext
    simp [J₂]
    ring
  rw [hJ]
  exact h.hasFDerivAt

/-- the inverse of `J₂ 2` -/
noncomputable def A₄ : ℝ →L[ℝ] ℝ := (4⁻¹ : ℝ) • ContinuousLinearMap.id ℝ ℝ

/-- the correction from `x = 2` -/
noncomputable def dz₄ : ℝ := -A₄ (f₂ 2)

theorem dz₄_eq : dz₄ = -1 / 2 := by
  norm_num [dz₄, A₄, f₂]

/-- On `F(x) = x² - 2` from `x = 2`: the hypotheses of `omegaEst_le` with `ω = 1/2`, and `‖dz̄‖ = (ω / 2) ‖dz‖²` and
`[ω] = ω` -/
theorem attained :
    (∀ t ∈ Icc (0 : ℝ) 1, HasFDerivAt f₂ (J₂ (2 + t • dz₄)) (2 + t • dz₄)) ∧ (∀ v, A₄ (J₂ 2 v) = v) ∧
      J₂ 2 dz₄ = -f₂ 2 ∧
      (∀ t ∈ Icc (0 : ℝ) 1, ‖A₄ ((J₂ (2 + t • dz₄) - J₂ 2) dz₄)‖ ≤ 1 / 2 * ‖t • dz₄‖ * ‖dz₄‖) ∧
      ‖-A₄ (f₂ (2 + dz₄))‖ = 1 / 2 / 2 * ‖dz₄‖ ^ 2 ∧
      omegaEst 1 (‖-A₄ (f₂ (2 + (1 : ℝ) • dz₄))‖ / ‖dz₄‖) ‖dz₄‖ = 1 / 2 := by
  refine ⟨fun t _ => hasFDerivAt_f₂ _, fun v => ?_, ?_, fun t ht => ?_, ?_, ?_⟩
  · simp [A₄, J₂]
    ring
  · norm_num [J₂, dz₄_eq, f₂]
  · have h : A₄ ((J₂ (2 + t • dz₄) - J₂ 2) dz₄) = t / 8 := by
      simp [A₄, J₂, dz₄_eq]
      ring
    rw [h, dz₄_eq, norm_smul]
    norm_num [abs_of_nonneg ht.1]
    linarith
  · norm_num [A₄, f₂, dz₄_eq]
  · norm_num [omegaEst, hEst, A₄, f₂, dz₄_eq]

end Controls
