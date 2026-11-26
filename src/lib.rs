//! Solvers of F(z) = 0 in double-, quad- and octo-double precision, written over a [`Problem`]: a problem gives its
//! residual and its exact Jacobian in one of multidouble's precisions (`Precision`: [`Double`], [`Quad`], [`Octo`]),
//! and the solvers here need nothing else of it. The multiple-shooting layout of the crate multishoot is one such
//! problem. Continuation in z and p together takes a [`Family`], the residual, the Jacobian and dF/dp at any p.
//!
//! * [`newton`]: the Newton step and the damped Newton loop, the chord and Shamanskii methods that reuse its
//!   factors, the Newton matrix factored in f64, double-double or another precision for a residual in P, and the
//!   floor rule that ends a run at its precision's floor;
//! * [`ladder`]: Newton's method through a ladder of precisions of the residual, the Jacobian and its factors, each
//!   stage ended at its floor by the floor rule;
//! * [`quasi`]: Newton's method with Broyden's updates between its factorizations (Deuflhard's QNERR);
//! * [`gcr`]: a preconditioned Krylov solver (GMRES in its GCR form) on exact Jacobian-vector products, and the Newton
//!   method whose steps it refines from an approximate factorization;
//! * [`lm`]: the Levenberg-Marquardt method, for starts where Newton's method stalls or its Jacobian is singular;
//! * [`anm`]: the asymptotic numerical method, the Taylor series of a solution branch in the problem's parameter;
//! * [`predict`]: predictors for continuation: Lagrange extrapolation, the cubic Hermite, and the exact tangent;
//! * [`arclength`]: pseudo-arclength continuation of a branch of a [`Family`] F(z, p) through its folds, and the
//!   folds' location;
//! * [`deflation`]: further roots from the same start, with the known ones deflated (Farrell, Birkisson and Funke);
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
pub mod arclength;
pub mod deflation;
pub mod gcr;
pub mod ladder;
pub mod lm;
pub mod newton;
pub mod prec;
pub mod predict;
pub mod quasi;
pub mod singularity;
pub mod subspace;
pub mod winding;

pub use multidouble;
pub use multidouble::dd::DD;
pub use multidouble::precision::{Double, Octo, Precision, Quad};
pub use multidouble::simd::Native;
pub use newton::{NewtonOpts, NewtonReport, Stop};

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

/// A family of systems F(z, p) = 0 of `len()` equations in as many unknowns z and a scalar parameter p, evaluated at
/// any p in the precision P: the residual, the Jacobian dF/dz and the derivative dF/dp, exact to P. Continuation in z
/// and p together ([`arclength`]) needs F away from a fixed parameter, which [`ResidualJet`]'s series at p0 does not
/// give. The state z is held as P::PARTS part arrays of f64, as a [`Problem`]'s.
pub trait Family<P: Precision<Native>> {
    /// the number of unknowns z, and of residual rows
    fn len(&self) -> usize;
    /// whether the family has no unknowns
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// F(z, p)
    fn residual_at(&self, z: &[&[f64]], p: P::Scalar, par: bool) -> Result<Vec<P::Scalar>, String>;
    /// dF/dz at (z, p), laid out as [`Problem::jacobian`] lays it out. Returns (c, ld).
    fn jacobian_at(&self, z: &[&[f64]], p: P::Scalar, par: bool) -> Result<(Vec<Vec<f64>>, usize), String>;
    /// dF/dp at (z, p), one scalar per row
    fn dp_at(&self, z: &[&[f64]], p: P::Scalar, par: bool) -> Result<Vec<P::Scalar>, String>;
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

    /// The residual's series one order at a time ([`JetOrders`]), for orders up to `order`. By default each order is
    /// `residual_jet`'s, which forms every lower order again ([`Recompute`]), so a series of order N costs N
    /// evaluations of growing length. A problem that keeps its lower orders between calls overrides it.
    fn jet_orders<'a>(&'a self, order: usize) -> Result<Box<dyn JetOrders<P> + 'a>, String>
    where
        P: 'a,
    {
        let _ = order;
        Ok(Box::new(Recompute::new(self)))
    }
}

/// The residual's series in the parameter formed one order at a time, as the series of a branch needs it: order k
/// once z_0..z_(k-1) are known, with z_k left out.
pub trait JetOrders<P: Precision<Native>> {
    /// Order k of the residual's series (one scalar per row), where z, the newest coefficient vector (P::PARTS part
    /// arrays of length `len()`), is z_(k-1): the first call gives z_0 and returns order 1. z_k and the orders past it
    /// are zero, so the result is F_k of the solve J z_k = -F_k.
    fn next(&mut self, z: &[&[f64]], par: bool) -> Result<Vec<P::Scalar>, String>;
}

/// [`JetOrders`] by [`ResidualJet::residual_jet`]: each call evaluates the series of orders 0..=k on every vector
/// given so far and returns its order k.
pub struct Recompute<'a, P: Precision<Native>, Pr: ResidualJet<P> + ?Sized> {
    pr: &'a Pr,
    zs: Vec<Vec<Vec<f64>>>,
    _p: std::marker::PhantomData<P>,
}

impl<'a, P: Precision<Native>, Pr: ResidualJet<P> + ?Sized> Recompute<'a, P, Pr> {
    /// no vectors given yet
    pub fn new(pr: &'a Pr) -> Self {
        Recompute {
            pr,
            zs: Vec::new(),
            _p: std::marker::PhantomData,
        }
    }
}

impl<P: Precision<Native>, Pr: ResidualJet<P> + ?Sized> JetOrders<P> for Recompute<'_, P, Pr> {
    fn next(&mut self, z: &[&[f64]], par: bool) -> Result<Vec<P::Scalar>, String> {
        self.zs.push(z.iter().map(|p| p.to_vec()).collect());
        let k = self.zs.len();
        let refs: Vec<Vec<&[f64]>> = self.zs.iter().map(|v| prec::parts(v)).collect();
        let mut f = self.pr.residual_jet(&refs, k + 1, par)?;
        if f.len() <= k {
            return Err(format!("residual_jet gave {} orders, not {}", f.len(), k + 1));
        }
        Ok(f.swap_remove(k))
    }
}
