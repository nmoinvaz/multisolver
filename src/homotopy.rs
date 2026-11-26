//! A homotopy in a held offset: from a start z0 that solves its system exactly to the system wanted, along
//!
//! ```text
//! H(z, t) = F(z) - a - (1 - t) b,   t from 0 to 1,
//! ```
//!
//! with a and b chosen so that H(z0, 0) = 0 (for instance a = 0 and b = F(z0), whose path ends at F(z) = 0: Newton's
//! homotopy). It continues a solution through a change of the system too large for Newton's method from z0, and it
//! stops where the path folds.
//!
//! Each rung goes from t to t' = min(t + h, 1). The predictor is z + (t' - t) tau, with the tangent tau = -J(z)^-1 b
//! from the factors at z. The corrector takes full Newton steps on H(., t'). It converges when a correction's max |dz|
//! is at most max(`tol_dz` (t' - t) max |tau|, `floor_dz`) and max |H| is at most max(`tol_f` (t' - t) max |b|,
//! `floor_f`). The rung is refused when the first correction exceeds half the predicted move (t' - t) max |tau|, when a
//! correction exceeds half the one before, when a residual, a Jacobian or a correction is not finite, when a Jacobian
//! does not factor (at an iterate or at the accepted point), or after `max_corrections` corrections. A refused rung
//! retries from the same z with h = (t' - t) / 4. An accepted one sets h = 4 (t' - t) after at most 3 corrections,
//! (t' - t) after at most 5, (t' - t) / 2 otherwise, at most `h_max`. The trace stops after `max_refusals` refusals in
//! a row, or when h falls below `h_min` t, the signature of a fold, where the Jacobian's weakest singular value falls
//! to zero and |tau| grows without bound. At t = 1, Newton's method goes on until max |H| is at most `polish_tol` or a
//! correction stops contracting.
use crate::newton::{exact, Exact, Factored};
use crate::prec::{max_abs, parts};
use crate::Problem;
use multidouble::dd::DD;
use multidouble::precision::Precision;
use multidouble::simd::Native;

/// The settings of [`homotopy`]
#[derive(Clone, Copy, Debug)]
pub struct HomotopyOpts {
    /// the first step in t
    pub h0: f64,
    /// the largest step in t
    pub h_max: f64,
    /// below this fraction of t a step is too small, and the trace stops (a fold)
    pub h_min: f64,
    /// the corrector's tolerances, relative to the predicted move and to the move of the offset, and their floors
    pub tol_dz: f64,
    pub tol_f: f64,
    pub floor_dz: f64,
    pub floor_f: f64,
    /// the corrections of a rung at most
    pub max_corrections: usize,
    /// the refusals in a row at most
    pub max_refusals: usize,
    /// the rungs at most
    pub max_rungs: usize,
    /// the endpoint's Newton stops at this max |H| (or when a correction stops contracting)
    pub polish_tol: f64,
    pub par: bool,
}

impl Default for HomotopyOpts {
    fn default() -> Self {
        HomotopyOpts {
            h0: 0.01,
            h_max: 0.05,
            h_min: 1e-6,
            tol_dz: 1e-12,
            tol_f: 1e-6,
            floor_dz: 0.0,
            floor_f: 0.0,
            max_corrections: 8,
            max_refusals: 12,
            max_rungs: 400,
            polish_tol: 0.0,
            par: true,
        }
    }
}

/// One accepted rung: its t, step h, corrections, and max |H| and max |z - z_prev| at its end
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rung {
    pub t: f64,
    pub h: f64,
    pub corrections: usize,
    pub fnorm: f64,
    pub moved: f64,
}

/// Why the trace stopped
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HomotopyStop {
    /// t = 1 reached and polished
    Reached,
    /// `max_refusals` refusals in a row
    Refused,
    /// the step fell below `h_min` t
    StepTooSmall,
    /// `max_rungs` rungs
    MaxRungs,
}

/// What [`homotopy`] did: max |H(z0, 0)|, the accepted rungs, the refusals, the endpoint polish's max |H| at each step,
/// and why it stopped
#[derive(Clone, Debug)]
pub struct HomotopyReport {
    pub start: f64,
    pub rungs: Vec<Rung>,
    pub refusals: usize,
    pub polish: Vec<f64>,
    pub stop: HomotopyStop,
}

/// H(z, t) = F(z) - a - (1 - t) b in P
fn h_of<P: Precision<Native>, Pr: Problem<P> + ?Sized>(
    pr: &Pr,
    z: &[Vec<f64>],
    a: &[P::Scalar],
    b: &[P::Scalar],
    t: f64,
    par: bool,
) -> Result<Vec<P::Scalar>, String> {
    let f = pr.residual(&parts(z), par)?;
    let s = P::from_dd(DD::ONE - DD::from_f64(t));
    Ok(f.iter().zip(a).zip(b).map(|((&f, &a), &b)| f - a - s * b).collect())
}

/// z + s v in P, v given as scalars
fn axpy<P: Precision<Native>>(z: &[Vec<f64>], v: &[P::Scalar], s: f64) -> Vec<Vec<f64>> {
    let sc = P::from_dd(DD::from_f64(s));
    let n = z[0].len();
    let mut out: Vec<Vec<f64>> = (0..P::PARTS).map(|_| vec![0.0; n]).collect();
    for i in 0..n {
        let x = P::from_raw(&z.iter().map(|p| p[i]).collect::<Vec<_>>()) + sc * v[i];
        for (q, o) in out.iter_mut().enumerate() {
            o[i] = P::part(x, q);
        }
    }
    out
}

/// The exact Jacobian at z, factored: the problem's error is returned, a factorization that fails gives `None`
fn factored<P: Precision<Native>, Pr: Problem<P> + ?Sized>(
    pr: &Pr,
    z: &[Vec<f64>],
    par: bool,
) -> Result<Option<Exact<P>>, String> {
    let (c, ld) = pr.jacobian(&parts(z), par)?;
    Ok(P::lu_factor(pr.len(), ld, c, par).ok().map(Exact))
}

/// The homotopy from z (P::PARTS part arrays, solving H(z, 0) = 0, replaced by the trace's last point) to t = 1 (the
/// module's documentation). An error of the problem ends it with that error; a rung's non-finite values or a Jacobian
/// that does not factor refuse it.
pub fn homotopy<P: Precision<Native>, Pr: Problem<P> + ?Sized>(
    pr: &Pr,
    z: &mut Vec<Vec<f64>>,
    a: &[P::Scalar],
    b: &[P::Scalar],
    opts: &HomotopyOpts,
) -> Result<HomotopyReport, String> {
    let n = pr.len();
    if a.len() != n || b.len() != n || z.len() != P::PARTS {
        return Err(format!("homotopy: a, b and z of {n} unknowns in {} parts", P::PARTS));
    }
    let par = opts.par;
    let start = max_abs::<P>(&h_of::<P, Pr>(pr, z, a, b, 0.0, par)?);
    let bmax = max_abs::<P>(b);
    let mut rep = HomotopyReport {
        start,
        rungs: Vec::new(),
        refusals: 0,
        polish: Vec::new(),
        stop: HomotopyStop::MaxRungs,
    };
    let (mut t, mut h) = (0.0f64, opts.h0.min(opts.h_max));
    let mut in_a_row = 0;
    let mut lu: Exact<P> = exact::<P, Pr>(pr, &parts(z), par)?;
    while rep.rungs.len() < opts.max_rungs {
        if t >= 1.0 {
            break;
        }
        if h < opts.h_min * t.max(f64::MIN_POSITIVE) {
            rep.stop = HomotopyStop::StepTooSmall;
            return Ok(rep);
        }
        let t2 = (t + h).min(1.0);
        let dt = t2 - t;
        // tau = -J^-1 b, as -J^-1 r of r = b
        let tau = lu.try_solve_neg(b)?;
        let move_max = dt * max_abs::<P>(&tau);
        let tol_dz = (opts.tol_dz * move_max).max(opts.floor_dz);
        let tol_f = (opts.tol_f * dt * bmax).max(opts.floor_f);
        let mut zt = axpy::<P>(z, &tau, dt);
        let mut prev = f64::INFINITY;
        let mut count = 0;
        let mut fnorm = f64::NAN;
        // the factors at the accepted point, for the next tangent; None refuses the rung
        let mut next: Option<Exact<P>> = None;
        while count < opts.max_corrections {
            let hv = h_of::<P, Pr>(pr, &zt, a, b, t2, par)?;
            fnorm = max_abs::<P>(&hv);
            let Some(l) = factored::<P, Pr>(pr, &zt, par)? else {
                break;
            };
            let dz = l.try_solve_neg(&hv)?;
            let d = max_abs::<P>(&dz);
            if !fnorm.is_finite() || !d.is_finite() || (count == 0 && d > 0.5 * move_max) || d > 0.5 * prev {
                break;
            }
            zt = axpy::<P>(&zt, &dz, 1.0);
            count += 1;
            prev = d;
            if d <= tol_dz {
                let hv = h_of::<P, Pr>(pr, &zt, a, b, t2, par)?;
                fnorm = max_abs::<P>(&hv);
                if fnorm <= tol_f {
                    next = factored::<P, Pr>(pr, &zt, par)?;
                    break;
                }
            }
        }
        let Some(next) = next else {
            rep.refusals += 1;
            in_a_row += 1;
            if in_a_row >= opts.max_refusals {
                rep.stop = HomotopyStop::Refused;
                return Ok(rep);
            }
            h = dt / 4.0;
            continue;
        };
        in_a_row = 0;
        let moved = (0..n)
            .map(|i| {
                let x = P::from_raw(&zt.iter().map(|p| p[i]).collect::<Vec<_>>());
                let y = P::from_raw(&z.iter().map(|p| p[i]).collect::<Vec<_>>());
                P::to_dd(x - y).hi.abs()
            })
            .fold(0.0, crate::prec::nan_max);
        rep.rungs.push(Rung {
            t: t2,
            h: dt,
            corrections: count,
            fnorm,
            moved,
        });
        *z = zt;
        t = t2;
        lu = next;
        h = match count {
            0..=3 => 4.0 * dt,
            4..=5 => dt,
            _ => dt / 2.0,
        }
        .min(opts.h_max);
    }
    if t < 1.0 {
        return Ok(rep);
    }
    // the endpoint: Newton on H(., 1) until polish_tol or a correction that does not contract
    let mut prev = f64::INFINITY;
    for _ in 0..50 {
        let hv = h_of::<P, Pr>(pr, z, a, b, 1.0, par)?;
        let f = max_abs::<P>(&hv);
        rep.polish.push(f);
        if f <= opts.polish_tol {
            break;
        }
        let dz = exact::<P, Pr>(pr, &parts(z), par)?.try_solve_neg(&hv)?;
        let d = max_abs::<P>(&dz);
        if !d.is_finite() || d >= 0.5 * prev {
            break;
        }
        *z = axpy::<P>(z, &dz, 1.0);
        prev = d;
    }
    rep.stop = HomotopyStop::Reached;
    Ok(rep)
}
