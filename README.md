# multisolver

[![Build](https://github.com/nmoinvaz/multisolver/actions/workflows/build.yml/badge.svg)](https://github.com/nmoinvaz/multisolver/actions/workflows/build.yml)
[![Lint](https://github.com/nmoinvaz/multisolver/actions/workflows/lint.yml/badge.svg)](https://github.com/nmoinvaz/multisolver/actions/workflows/lint.yml)
[![codecov](https://codecov.io/gh/nmoinvaz/multisolver/graph/badge.svg)](https://codecov.io/gh/nmoinvaz/multisolver)
[![crates.io](https://img.shields.io/crates/v/multisolver.svg)](https://crates.io/crates/multisolver)
[![docs.rs](https://docs.rs/multisolver/badge.svg)](https://docs.rs/multisolver)

Solvers of nonlinear systems F(z) = 0 in double-, quad- and octo-double precision.

A problem enters through the `Problem` trait: its residual and its exact Jacobian in one of
[multidouble](https://github.com/nmoinvaz/multidouble)'s precisions (`Double`, `Quad`, `Octo`). The `Jvp` and
`ResidualJet` traits add the Jacobian-vector product and the residual's Taylor series in a parameter, which the Krylov
solver, the series of a branch and the exact tangent need. The series of a branch takes the residual's orders one at a
time (`ResidualJet::jet_orders`): by default each is a fresh `residual_jet` of all the lower orders, and a problem that
keeps its lower orders between calls overrides it, so that a series of order N costs N orders instead of N
evaluations of growing length. The solvers need nothing else of the problem.
[multishoot](https://github.com/nmoinvaz/multishoot)'s multiple-shooting layout is one such problem.

| Module | What it does |
|---|---|
| `newton` | the Newton step, the damped Newton loop (Deuflhard's correction rule or a residual rule), the chord and Shamanskii methods that reuse a factorization, and an a-posteriori error estimate from the simplified corrections |
| `gcr` | GMRES in its GCR form on exact Jacobian-vector products, with an adaptive Krylov direction where GMRES stagnates, and the Newton method whose solves it refines from an approximate factorization |
| `lm` | the Levenberg-Marquardt method, for starts where Newton's method stalls or its Jacobian is singular, handing over to Newton near the root |
| `anm` | the asymptotic numerical method: the Taylor series of a solution branch in the problem's parameter, and its radius |
| `predict` | continuation predictors: Lagrange extrapolation, the cubic Hermite, and the exact tangent |
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

## License

BSD 3-Clause, see [LICENSE](LICENSE).
