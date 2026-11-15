//! Solvers of F(z) = 0 in double-, quad- and octo-double precision, written over a [`Problem`]: a problem gives its
//! residual and its exact Jacobian in one of multidouble's precisions (`Precision`: [`Double`], [`Quad`], [`Octo`]),
//! and the solvers here need nothing else of it. The multiple-shooting layout of the crate multishoot is one such
//! problem.
//!
//! * [`newton`]: the Newton step and the damped Newton loop, and the chord and Shamanskii methods that reuse its
//!   factors;
//! * [`gcr`]: a preconditioned Krylov solver (GMRES in its GCR form) on exact Jacobian-vector products, and the Newton
//!   method whose steps it refines from an approximate factorization;
//! * [`lm`]: the Levenberg-Marquardt method, for starts where Newton's method stalls or its Jacobian is singular;
//! * [`anm`]: the asymptotic numerical method, the Taylor series of a solution branch in the problem's parameter;
//! * [`predict`]: predictors for continuation: Lagrange extrapolation, the cubic Hermite, and the exact tangent;
//! * [`singularity`]: the nearest singularities of a series from its coefficients: Pade poles that recur across the
//!   approximants' orders, the root test's radius, the ratio method and a recurrence for a conjugate pair;
//! * [`subspace`]: the weakest singular directions of the Jacobian, Newton's method with them removed from its steps,
//!   and the sensitivities of single unknowns to the residual;
//! * [`winding`]: the number and the places of the zeros of an analytic function inside a polygon, by the argument
//!   principle, and Muller's method for one zero;
//! * [`prec`]: helpers for states held as part arrays in any of the precisions.
//!
//! A problem is its residual and its exact Jacobian. Here x^2 = 2 in double-double, solved by the damped Newton loop:
//!
//! ```
//! use multisolver::multidouble::linalg::ld_for;
//! use multisolver::newton::{newton, NewtonOpts};
//! use multisolver::{Double, Problem, DD};
//!
//! struct Sqrt2;
//!
//! impl Problem<Double> for Sqrt2 {
//!     fn len(&self) -> usize {
//!         1
//!     }
//!     fn residual(&self, z: &[&[f64]], _par: bool) -> Result<Vec<DD>, String> {
//!         let x = DD { hi: z[0][0], lo: z[1][0] };
//!         Ok(vec![x * x - DD::from_f64(2.0)])
//!     }
//!     fn jacobian(&self, z: &[&[f64]], _par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
//!         let ld = ld_for(1);
//!         let d = DD { hi: z[0][0], lo: z[1][0] }.mul_f(2.0);
//!         let (mut hi, mut lo) = (vec![0.0; ld], vec![0.0; ld]);
//!         (hi[0], lo[0]) = (d.hi, d.lo);
//!         Ok((vec![hi, lo], ld))
//!     }
//! }
//!
//! let mut z = vec![vec![1.0], vec![0.0]];
//! let report = newton::<Double, _>(&Sqrt2, &mut z, &NewtonOpts::default()).unwrap();
//! let x = DD { hi: z[0][0], lo: z[1][0] };
//! assert!(report.converged && (x * x - DD::from_f64(2.0)).hi.abs() <= 1e-30);
//! ```
pub mod anm;
pub mod gcr;
pub mod lm;
pub mod newton;
pub mod prec;
pub mod predict;
pub mod singularity;
pub mod subspace;
pub mod winding;

pub use multidouble;
pub use multidouble::dd::DD;
pub use multidouble::precision::{Double, Octo, Precision, Quad};
pub use multidouble::simd::Native;
pub use newton::{NewtonOpts, NewtonReport};

/// A system F(z) = 0 of `len()` equations in as many unknowns, evaluated in the precision P. The state z is held as
/// P::PARTS part arrays of f64 (part q of unknown i at `z[q][i]`).
pub trait Problem<P: Precision<Native>> {
    /// the number of unknowns, and of residual rows
    fn len(&self) -> usize;
    /// whether the problem has no unknowns
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// F(z)
    fn residual(&self, z: &[&[f64]], par: bool) -> Result<Vec<P::Scalar>, String>;
    /// The exact Jacobian dF/dz at z, as `P::lu_factor` takes it: part q of entry (i, j) at `c[q][j * ld + i]`, with
    /// the leading dimension ld a multiple of 8 and at least `len()` (`multidouble::linalg::ld_for`). Returns (c, ld).
    fn jacobian(&self, z: &[&[f64]], par: bool) -> Result<(Vec<Vec<f64>>, usize), String>;
}

/// A problem whose Jacobian-vector product J v can be formed without the Jacobian, exact to the precision P
pub trait Jvp<P: Precision<Native>>: Problem<P> {
    /// J v at z, z and v as P::PARTS part arrays of length `len()`
    fn jvp(&self, z: &[&[f64]], v: &[&[f64]], par: bool) -> Result<Vec<P::Scalar>, String>;
}

/// A problem whose equations depend on a parameter p, with the residual's Taylor series in it: the coefficients of
/// F(sum_k z_k h^k, p0 + h) in h, exact to the precision P
pub trait ResidualJet<P: Precision<Native>>: Problem<P> {
    /// The coefficients of orders 0..nout - 1, one scalar per row, for the coefficient vectors zs (each P::PARTS part
    /// arrays of length `len()`, z_k zero past them). Order k is linear in z_k: J z_k plus what the lower orders give.
    fn residual_jet(&self, zs: &[Vec<&[f64]>], nout: usize, par: bool) -> Result<Vec<Vec<P::Scalar>>, String>;
}
