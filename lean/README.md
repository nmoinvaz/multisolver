# Lean proofs

The bounds that multisolver's documentation states as theorems, proved in Lean 4 on Mathlib, in exact arithmetic.

| File | Contents |
|---|---|
| `Multisolver/Kantorovich.lean` | `kantorovich`'s lower bounds: with Deuflhard's affine covariant Lipschitz condition at the iterate along the step, the simplified correction of a damped step satisfies `‖dz̄ - (1 - λ) dz‖ ≤ (λ² / 2) ω ‖dz‖²` (`simplified_correction_damped`), and `‖dz̄‖ ≤ (ω / 2) ‖dz‖²` at `λ = 1` (`simplified_correction`); for `0 < λ ≤ 1`, `[h]` and `[ω]` as `from_steps` forms them are at most `h = ω ‖dz‖` and `ω` (`hEst_le`, `omegaEst_le`) |
| `Controls/Kantorovich.lean` | the bounds failing with the factor 2 in place of the Jacobian 1 (`[ω] = 2` where `ω = 0`), with a moved trial (`[ω] = 1`), and with a damped step read as a full one (`[h] = 1` where `h = 0`); and attained on `x² - 2` from `x = 2` (`[ω] = ω = 1/2`) |
| `Controls/Axioms.lean` | no declaration of `Multisolver` rests on an axiom other than `propext`, `Classical.choice` and `Quot.sound`, so none on `sorry`; its control is a theorem from an axiom declared there |

## Building

```
cd lean
lake exe cache get Mathlib.Analysis.Calculus.MeanValue
lake build
```

The build fails on a proof left as `sorry` and on any warning.
