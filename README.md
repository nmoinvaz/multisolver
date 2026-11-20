# multisolver

[![Build](https://github.com/nmoinvaz/multisolver/actions/workflows/build.yml/badge.svg)](https://github.com/nmoinvaz/multisolver/actions/workflows/build.yml)
[![Lint](https://github.com/nmoinvaz/multisolver/actions/workflows/lint.yml/badge.svg)](https://github.com/nmoinvaz/multisolver/actions/workflows/lint.yml)
[![codecov](https://codecov.io/gh/nmoinvaz/multisolver/graph/badge.svg)](https://codecov.io/gh/nmoinvaz/multisolver)
[![crates.io](https://img.shields.io/crates/v/multisolver.svg)](https://crates.io/crates/multisolver)
[![docs.rs](https://docs.rs/multisolver/badge.svg)](https://docs.rs/multisolver)

Solvers of nonlinear systems F(z) = 0 in double-, quad- and octo-double precision.

A problem implements a few traits in one of [multidouble](https://github.com/nmoinvaz/multidouble)'s precisions
(`Double`, `Quad`, `Octo`), and the solvers need nothing else of it.
[multishoot](https://github.com/nmoinvaz/multishoot)'s multiple-shooting layout is one such problem.

| Trait | What the problem gives | What needs it |
|---|---|---|
| `Problem` | the residual F(z) and its exact Jacobian | every solver |
| `Jvp` | the Jacobian-vector product J v | the Krylov solver, the series of a branch, the exact tangent |
| `ResidualJet` | the residual's Taylor series in a parameter | the series of a branch, the exact tangent |
| `Family` | the residual, the Jacobian and dF/dp at any parameter | continuation through folds |

A problem that keeps the lower orders of its series between calls can override `ResidualJet::jet_orders`, so that a
series of order N costs N orders rather than N evaluations of growing length.

| Module | What it does |
|---|---|
| `newton` | the Newton step, the damped Newton loop (Deuflhard's correction rule or a residual rule), the chord and Shamanskii methods that reuse a factorization, the Newton matrix factored in double-double or f64 for a residual in quad- or octo-double, NLEQ-ERR's scaled norms for unknowns of different sizes, and an a-posteriori error estimate from the simplified corrections |
| `quasi` | Newton's method with Broyden's good updates between its factorizations (Deuflhard's QNERR), one residual and one solve per step and no Jacobian |
| `gcr` | GMRES in its GCR form on exact Jacobian-vector products, with an adaptive Krylov direction where GMRES stagnates, and the Newton method whose solves it refines from an approximate factorization |
| `lm` | the Levenberg-Marquardt method, for starts where Newton's method stalls or its Jacobian is singular, its steps solved in double-double in any precision, handing over to Newton near the root |
| `anm` | the asymptotic numerical method: the Taylor series of a solution branch in the problem's parameter, and its radius |
| `predict` | continuation predictors: Lagrange extrapolation, the cubic Hermite, and the exact tangent |
| `arclength` | pseudo-arclength continuation through folds: the bordered corrector, step control by the corrector's contraction, fold detection by the tangent's sign, and the fold's location |
| `deflation` | further roots from the same start: the problem with its known roots deflated (Farrell, Birkisson and Funke), its residual and Jacobian exact so that any solver runs on it, and the roots collected one search at a time |
| `singularity` | the nearest singularities of a series from its coefficients: Pade poles that recur across the approximants' orders, the root test's radius, the ratio method of Domb and Sykes, and a recurrence for a conjugate pair |
| `subspace` | the weakest singular directions of the Jacobian, Newton's method with them removed from its steps, the sensitivities of single unknowns, and the 1-norm condition estimate |
| `winding` | the number of zeros of an analytic function inside a polygon by the argument principle, with adaptive sampling that catches close pairs and rows of zeros, their first moment, and Muller's method for one zero |

**Status:** early. The API may change between minor versions until 1.0.

## Building

```
cargo test --release
cargo test --release --features portable
cargo clippy --all-targets -- -D warnings
```

The tests are planted-truth problems with a known answer. Each accuracy check has a control that feeds it a wrong
result and must fail.

## Related crates

- [multidouble](https://github.com/nmoinvaz/multidouble) is the double-, quad- and octo-double arithmetic this crate
  is written over, scalar and on 8-lane SIMD vectors with bitwise the same results on every backend.
- [multishoot](https://github.com/nmoinvaz/multishoot) solves boundary-value problems by multiple shooting, and its
  layout is a `Problem` that this crate's solvers take.

## License

BSD 3-Clause, see [LICENSE](LICENSE).
