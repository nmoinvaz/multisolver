//! Newton's method on F(z) = 0 in any precision, for any [`Problem`]: the step dz = -J^-1 F with the exact Jacobian
//! and a dense LU, and a damped loop.
//!
//! The default damping is affine covariant (Deuflhard, Newton Methods for Nonlinear Problems, Springer 2004, section
//! 3.3, the error-oriented global Newton method NLEQ-ERR). A trial z + lambda dz is accepted when its simplified
//! correction dz_bar = -J(z)^-1 F(z + lambda dz), formed with the old LU by one solve, passes the restricted
//! monotonicity test |dz_bar| < (1 - lambda / 4) |dz|. lambda is predicted from the last step's contraction and
//! corrected from the trial's. No scaling of the residual's rows changes these tests, so a problem whose rows differ by
//! orders of magnitude takes the steps of its unscaled copy. The residual rule, halving while max |F| does not fall,
//! is kept for comparison.
//!
//! The corrections' norm is the 2-norm by default, in which the largest unknowns dominate. With
//! [`Scaling::Iterates`] it is NLEQ-ERR's scaled norm |v| = (sum_i (v_i / w_i)^2 / n)^(1/2), the weights
//! w_i = max(floor, (|z_i| + |z'_i|) / 2) formed anew at each step from its iterate z and the one before it z' (z'
//! = z at the first), as `nleqerr_rescale` of NLEQ_ERR forms them (Weimann, ZIB NewtonLib 2006, the code of
//! Deuflhard's book); the last step's corrections enter the next lambda's prediction in the new weights. Each unknown
//! is then measured relative to its size above the floor, so a problem whose unknowns differ by orders of magnitude
//! takes the steps of its copy in unknowns of one size, and `tol_dz` is a relative tolerance.
//!
//! The Newton matrix need not be factored in P: with the residual in P, factors of the Jacobian rounded to
//! double-double or f64 ([`Lowered`], [`newton_lowered`]) converge to P's accuracy linearly, at a contraction of
//! about cond(J) 2^-104 or 2^-53 per step (iterative refinement; Moler, J. ACM 14 (1967) 316).
use crate::prec::{max_abs, nan_max, parts};
use crate::Problem;
use multidouble::dd::DD;
use multidouble::linalg::DDLu;
use multidouble::precision::Precision;
use multidouble::simd::Native;
use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

/// A Newton step at z: dz = -J^-1 F (PARTS part arrays), the residual F, the LU of J, and the seconds of the
/// residual, the Jacobian, the factorization and the solve.
pub struct Step<P: Precision<Native>> {
    pub dz: Vec<Vec<f64>>,
    pub f: Vec<P::Scalar>,
    pub lu: P::Lu,
    pub t: [f64; 4],
}

/// How [`newton`] damps its steps
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Damping {
    /// Deuflhard's error-oriented rule: the simplified correction must contract (the module's documentation)
    Correction,
    /// halve the step while max |F| does not fall, at most `max_halvings` times, then take it
    Residual,
}

/// The norm in which the correction rule measures its corrections (the module's documentation)
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scaling {
    /// the 2-norm of the unknowns as they are
    None,
    /// NLEQ-ERR's scaled norm, the weights w_i = max(floor, (|z_i| + |z'_i|) / 2) of each step's iterate z and the one
    /// before it z'; `floor` (a positive normal f64) is the size below which an unknown counts as small in absolute
    /// terms
    Iterates { floor: f64 },
}

/// The settings of [`newton`]
#[derive(Clone, Copy, Debug)]
pub struct NewtonOpts {
    /// stop when max |F| is at most this
    pub tol: f64,
    /// with the correction rule, also stop after a full step whose simplified correction has a norm at most this (0:
    /// off); unlike max |F| it does not depend on the rows' scaling, and with `Scaling::Iterates` it is relative
    pub tol_dz: f64,
    /// The floor rule: a full step (lambda 1) of max |dz| at most `floor_dz` that cuts max |F| by less than this
    /// factor, or that the damping refuses, ends the run unconverged at the precision's floor ([`Stop::Floor`]). 10 is
    /// the usual factor, and 0 (the default) turns the rule off. Only a step with factors formed at its own iterate
    /// stops the run. A chord step that gains this little has its factors formed anew instead
    /// ([`newton_chord`], [`crate::ladder`]). [`newton_with`] takes every factorization it is given as formed at the
    /// step's iterate, so give it reused factors ([`reuse`]) only with the rule off. The rule assumes Newton's
    /// quadratic rate: factors whose linear contraction is above 1 / floor_cut (f64 factors of a Jacobian of
    /// condition above about 10^14) end the run early.
    pub floor_cut: f64,
    /// the floor rule's bound on the step, max |dz| over the unknowns (absolute, from the hi parts): a larger step is
    /// far from the solution, where a small cut is the damping's, not the floor's (default 1e-12)
    pub floor_dz: f64,
    /// Newton steps at most
    pub max_iter: usize,
    /// the correction rule's smallest lambda is 2^-max_halvings, below which it stops, not converged; the residual
    /// rule halves at most this many times
    pub max_halvings: usize,
    /// the correction rule's first lambda (1 for a mildly nonlinear problem)
    pub lambda0: f64,
    pub damping: Damping,
    /// the correction rule's norm of the corrections
    pub scaling: Scaling,
    pub par: bool,
}

impl Default for NewtonOpts {
    fn default() -> Self {
        NewtonOpts {
            tol: 1e-28,
            tol_dz: 0.0,
            floor_cut: 0.0,
            floor_dz: 1e-12,
            max_iter: 20,
            max_halvings: 10,
            lambda0: 1.0,
            damping: Damping::Correction,
            scaling: Scaling::None,
            par: true,
        }
    }
}

/// Why a damped loop stopped
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Stop {
    /// max |F| at most `tol`
    Tol,
    /// a full step's simplified correction at most `tol_dz`
    TolDz,
    /// The floor rule ([`NewtonOpts::floor_cut`]): a full step of max |dz| = `dz` cut max |F| by only `cut`. The step
    /// was taken when it lowered max |F| (cut > 1) and left out when the damping refused it. Max |F| at the end, the
    /// last of [`NewtonReport::fnorm`], is the floor reached, and [`NewtonReport::error_estimate`] the error there.
    Floor { cut: f64, dz: f64 },
    /// the correction rule found no acceptable lambda above 2^-max_halvings
    Refused,
    /// a residual that is not finite
    NotFinite,
    /// `max_iter` steps without another stop
    MaxIter,
}

/// What [`newton`] did: max |F| before each step and at the end, each step's lambda, the correction rule's
/// contraction theta = |dz_bar| / |dz| and simplified correction |dz_bar| = |J(z_k)^-1 F(z_(k+1))| (in the norm of
/// [`NewtonOpts::scaling`]) for each step taken (both empty with the residual rule), whether it converged, and why it
/// stopped.
#[derive(Clone, Debug)]
pub struct NewtonReport {
    pub fnorm: Vec<f64>,
    pub scale: Vec<f64>,
    pub theta: Vec<f64>,
    pub dz_bar: Vec<f64>,
    pub converged: bool,
    pub stop: Stop,
}

impl NewtonReport {
    /// The error of the final iterate, |z - z*| in the norm of [`NewtonOpts::scaling`] (the 2-norm by default, a
    /// relative error with `Scaling::Iterates`), estimated by the last step's simplified correction
    /// |M^-1 F(z)|, formed with the step's factors M (Deuflhard, Newton Methods for Nonlinear Problems, Springer 2004).
    /// With the exact Jacobian's factors (`newton`), M approaches J(z*) and the correction equals -(z - z*) to first
    /// order. With factors frozen at another point or approximate (`newton_chord`, `reuse`, `newton_with`), it is
    /// M^-1 J(z*) times the error, which differs from the error by the factors' mismatch: for F(x) = x^2 - 1 from x = 2
    /// with J frozen at 4 it tends to half the error. None with the residual rule or without a step.
    pub fn error_estimate(&self) -> Option<f64> {
        self.dz_bar.last().copied()
    }

    pub(crate) fn new() -> Self {
        NewtonReport {
            fnorm: Vec::new(),
            scale: Vec::new(),
            theta: Vec::new(),
            dz_bar: Vec::new(),
            converged: false,
            stop: Stop::MaxIter,
        }
    }
}

/// z + s dz in the precision, part by part, dz given as scalars
pub(crate) fn axpy<P: Precision<Native>>(z: &[&[f64]], dz: &[P::Scalar], s: f64) -> Vec<Vec<f64>> {
    let l = z[0].len();
    let mut out: Vec<Vec<f64>> = (0..P::PARTS).map(|_| vec![0.0; l]).collect();
    let sc = P::from_dd(DD::from_f64(s));
    let mut a = vec![0.0; P::PARTS];
    for i in 0..l {
        for q in 0..P::PARTS {
            a[q] = z[q][i];
        }
        let v = P::from_raw(&a) + sc * dz[i];
        for (q, o) in out.iter_mut().enumerate() {
            o[i] = P::part(v, q);
        }
    }
    out
}

/// the Euclidean norm of the scalars, from the hi parts of their double-double roundings, scaled by their largest
/// magnitude so that no square overflows; NaN when any is NaN
pub(crate) fn norm2<P: Precision<Native>>(v: &[P::Scalar]) -> f64 {
    let m = max_abs::<P>(v);
    if m == 0.0 || !m.is_finite() {
        return m;
    }
    m * v.iter().map(|&x| P::to_dd(x).hi / m).map(|x| x * x).sum::<f64>().sqrt()
}

/// NLEQ-ERR's weights of the step from z' to z, w_i = max(floor, (|z_i| + |z'_i|) / 2) from the hi parts, halved
/// before the sum so that it cannot overflow; None for the 2-norm
pub(crate) fn weights(scaling: Scaling, z: &[&[f64]], zp: &[&[f64]]) -> Option<Vec<f64>> {
    match scaling {
        Scaling::None => None,
        Scaling::Iterates { floor } => Some(
            z[0].iter()
                .zip(zp[0])
                .map(|(&a, &b)| floor.max(0.5 * a.abs() + 0.5 * b.abs()))
                .collect(),
        ),
    }
}

/// The norm of the correction rule: `norm2` without weights, else (sum_i (v_i / w_i)^2 / n)^(1/2) from the hi parts,
/// scaled by the largest |v_i / w_i| as `norm2`; NaN when any is NaN
pub(crate) fn wnorm<P: Precision<Native>>(v: &[P::Scalar], w: Option<&[f64]>) -> f64 {
    let Some(w) = w else {
        return norm2::<P>(v);
    };
    let x: Vec<f64> = v.iter().zip(w).map(|(&a, &b)| P::to_dd(a).hi / b).collect();
    let m = x.iter().fold(0.0, |m, &a| nan_max(m, a.abs()));
    if m == 0.0 || !m.is_finite() {
        return m;
    }
    m * (x.iter().map(|&a| (a / m) * (a / m)).sum::<f64>() / x.len() as f64).sqrt()
}

/// The checks of `opts` that every damped loop makes
pub(crate) fn check_opts(opts: &NewtonOpts) -> Result<(), String> {
    if !(opts.lambda0 > 0.0 && opts.lambda0 <= 1.0) {
        return Err(format!("newton: lambda0 must lie in (0, 1], not {}", opts.lambda0));
    }
    // 0.5^1074 is the smallest positive f64; past it the minimum step length would be zero
    if opts.max_halvings > 1074 {
        return Err(format!(
            "newton: max_halvings {} is beyond f64's range (at most 1074)",
            opts.max_halvings
        ));
    }
    // NaN fails each comparison
    let floor_ok = opts.floor_cut.is_finite() && opts.floor_cut >= 0.0 && opts.floor_dz >= 0.0;
    if !floor_ok {
        return Err(format!(
            "newton: floor_cut must be finite and floor_dz non-negative, not {} and {}",
            opts.floor_cut, opts.floor_dz
        ));
    }
    if let Scaling::Iterates { floor } = opts.scaling {
        // a normal floor keeps every 1 / w_i finite
        if !(floor > 0.0 && floor.is_normal()) {
            return Err(format!(
                "newton: the scaling's floor must be a positive normal f64, not {floor}"
            ));
        }
    }
    Ok(())
}

/// -J^-1 r with the factored J
fn solve_neg<P: Precision<Native>>(lu: &P::Lu, r: &[P::Scalar]) -> Vec<P::Scalar> {
    let mut b: Vec<P::Scalar> = r.iter().map(|&v| -v).collect();
    P::lu_solve(lu, &mut b);
    b
}

/// A factored Newton matrix, exact or approximate: the loops' corrections -J^-1 r in P
pub trait Factored<P: Precision<Native>> {
    /// -J^-1 r
    fn solve_neg(&self, r: &[P::Scalar]) -> Vec<P::Scalar>;
    /// -J^-1 r, or the error of a solve that can fail, such as an iterative one that evaluates the problem
    /// ([`crate::gcr::Refined`]); the damped loops call this one
    fn try_solve_neg(&self, r: &[P::Scalar]) -> Result<Vec<P::Scalar>, String> {
        Ok(self.solve_neg(r))
    }
}

impl<P: Precision<Native>, F: Factored<P>> Factored<P> for Rc<F> {
    fn solve_neg(&self, r: &[P::Scalar]) -> Vec<P::Scalar> {
        (**self).solve_neg(r)
    }
    fn try_solve_neg(&self, r: &[P::Scalar]) -> Result<Vec<P::Scalar>, String> {
        (**self).try_solve_neg(r)
    }
}

/// The exact Jacobian's LU in P
pub struct Exact<P: Precision<Native>>(pub P::Lu);

impl<P: Precision<Native>> Factored<P> for Exact<P> {
    fn solve_neg(&self, r: &[P::Scalar]) -> Vec<P::Scalar> {
        solve_neg::<P>(&self.0, r)
    }
}

/// The exact Jacobian of `pr` at z, factored
pub fn exact<P: Precision<Native>, Pr: Problem<P> + ?Sized>(
    pr: &Pr,
    z: &[&[f64]],
    par: bool,
) -> Result<Exact<P>, String> {
    let (c, ld) = pr.jacobian(z, par)?;
    Ok(Exact(P::lu_factor(pr.len(), ld, c, par)?))
}

/// The rounding of a Jacobian that [`Lowered`] factors
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lower {
    /// to double-double, its first two parts (the parts of a quad- or octo-double do not overlap): the factors of
    /// `DDLu::factor`, which leave Newton's error contracting by about cond(J) 2^-104 per step
    Double,
    /// to f64, its first part: the f64 LU of `DDLu::factor_f64`, which leaves a contraction of about cond(J) 2^-53
    F64,
}

/// A Newton matrix factored in double-double or in f64 for a residual in P: each solve -M^-1 r rounds r to
/// double-double, solves in double-double with the factors, and takes the result into P exactly. Since the
/// residual is P's, Newton's method converges to P's accuracy, linearly by the contraction of [`Lower`], at the cost
/// of a factorization in double-double or f64: at n = 256 a double-double LU costs about 1/20 of a quad-double one
/// and 1/120 of an octo-double one, an f64 LU about a sixth of a double-double one. With
/// [`crate::gcr::newton_refined`] it preconditions the exact steps. For P = `Double` and
/// `Lower::Double` the factors are `P::lu_factor`'s and the solves [`Exact`]'s, bit for bit.
pub struct Lowered(pub DDLu);

impl Lowered {
    /// The Jacobian given as `Problem::jacobian` returns it (part q of entry (i, j) at `c[q][j * ld + i]`, any number
    /// of parts), rounded by `to` and factored
    pub fn factor(n: usize, ld: usize, mut c: Vec<Vec<f64>>, to: Lower, par: bool) -> Result<Lowered, String> {
        if c.is_empty() {
            return Err("Lowered::factor: a Jacobian without parts".into());
        }
        let hi = std::mem::take(&mut c[0]);
        let lu = match to {
            Lower::Double => {
                let lo = c.get_mut(1).map_or_else(|| vec![0.0; hi.len()], std::mem::take);
                DDLu::factor(n, ld, hi, lo, par)?
            }
            Lower::F64 => DDLu::factor_f64(n, ld, hi, par)?,
        };
        Ok(Lowered(lu))
    }
}

impl<P: Precision<Native>> Factored<P> for Lowered {
    fn solve_neg(&self, r: &[P::Scalar]) -> Vec<P::Scalar> {
        let mut b: Vec<DD> = r.iter().map(|&v| -P::to_dd(v)).collect();
        self.0.solve(&mut b);
        b.into_iter().map(P::from_dd).collect()
    }
}

/// The exact Jacobian of `pr` at z, rounded by `to` and factored ([`Lowered`])
pub fn lowered<P: Precision<Native>, Pr: Problem<P> + ?Sized>(
    pr: &Pr,
    z: &[&[f64]],
    to: Lower,
    par: bool,
) -> Result<Lowered, String> {
    let (c, ld) = pr.jacobian(z, par)?;
    Lowered::factor(pr.len(), ld, c, to, par)
}

/// x in the precision Q: its leading min(P::PARTS, Q::PARTS) parts, exact when Q holds them all and otherwise
/// within the ulp of the last part kept, as the parts do not overlap
pub fn convert<P: Precision<Native>, Q: Precision<Native>>(x: P::Scalar) -> Q::Scalar {
    let k = P::PARTS.min(Q::PARTS);
    let mut p = [0.0f64; 8];
    for (q, v) in p.iter_mut().enumerate().take(k) {
        *v = P::part(x, q);
    }
    Q::new(&p[..k])
}

/// A Newton matrix factored in the precision Q for a residual in P: each solve -M^-1 r takes r to Q ([`convert`]),
/// solves there with `Q::lu_solve` and takes the result back to P. Each step then contracts Newton's error by about
/// cond(J) times Q's rounding: with Q = `Quad` for P = `Octo`, by cond(J) 2^-209, where double-double factors
/// ([`Lowered`]) stall once cond(J) 2^-104 nears 1.
pub struct FactorsIn<Q: Precision<Native>>(pub Q::Lu);

impl<Q: Precision<Native>> FactorsIn<Q> {
    /// The Jacobian given as `Problem::jacobian` returns it (part q of entry (i, j) at `c[q][j * ld + i]`, any number
    /// of parts), its entries taken to Q's parts (the leading ones kept, or zero parts added) and factored by
    /// `Q::lu_factor`
    pub fn factor(n: usize, ld: usize, mut c: Vec<Vec<f64>>, par: bool) -> Result<Self, String> {
        let Some(len) = c.first().map(Vec::len) else {
            return Err("FactorsIn::factor: a Jacobian without parts".into());
        };
        c.truncate(Q::PARTS);
        c.resize(Q::PARTS, vec![0.0; len]);
        Ok(FactorsIn(Q::lu_factor(n, ld, c, par)?))
    }
}

impl<P: Precision<Native>, Q: Precision<Native>> Factored<P> for FactorsIn<Q> {
    fn solve_neg(&self, r: &[P::Scalar]) -> Vec<P::Scalar> {
        let mut b: Vec<Q::Scalar> = r.iter().map(|&v| -convert::<P, Q>(v)).collect();
        Q::lu_solve(&self.0, &mut b);
        b.into_iter().map(convert::<Q, P>).collect()
    }
}

/// A factorization reused for `every` steps: the returned closure calls `factor` at the first step and again every
/// `every` steps, and hands out the last factors in between. The loops of [`newton_with`] call it once per step, so
/// with it they run Shamanskii's method, the chord method for `every` at least the step count (Kelley, Iterative
/// Methods for Linear and Nonlinear Equations, SIAM 1995, section 5.4). Each reuse saves a Jacobian and a
/// factorization, and the steps within a run converge linearly, contracting by about the change of J since its
/// factorization. `every` is at least 1.
pub fn reuse<F>(
    every: usize,
    factor: impl Fn(&[&[f64]]) -> Result<F, String>,
) -> impl Fn(&[&[f64]]) -> Result<Rc<F>, String> {
    assert!(every >= 1, "reuse: every must be at least 1");
    let last: RefCell<Option<(usize, Rc<F>)>> = RefCell::new(None);
    move |z| {
        let mut last = last.borrow_mut();
        if let Some((uses, f)) = last.as_mut() {
            if *uses < every {
                *uses += 1;
                return Ok(f.clone());
            }
        }
        let f = Rc::new(factor(z)?);
        *last = Some((1, f.clone()));
        Ok(f)
    }
}

/// The Newton step at z.
pub fn newton_step<P: Precision<Native>, Pr: Problem<P> + ?Sized>(
    pr: &Pr,
    z: &[&[f64]],
    par: bool,
) -> Result<Step<P>, String> {
    let l = pr.len();
    let mut t = [0.0f64; 4];
    let t0 = Instant::now();
    let f = pr.residual(z, par)?;
    t[0] = t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    let (c, ld) = pr.jacobian(z, par)?;
    t[1] = t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    let lu = P::lu_factor(l, ld, c, par)?;
    t[2] = t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    let b = solve_neg::<P>(&lu, &f);
    t[3] = t0.elapsed().as_secs_f64();
    let dz = (0..P::PARTS)
        .map(|q| b.iter().map(|&v| P::part(v, q)).collect())
        .collect();
    Ok(Step { dz, f, lu, t })
}

/// Newton's method from z (PARTS part arrays, updated in place), damped by `opts.damping`. A residual that is not
/// finite (NaN or infinite) is never converged: at the start of a step it stops the method; in a trial it fails
/// the correction rule's test (a NaN theta), and stops the residual rule if its last halving still has one.
pub fn newton<P: Precision<Native>, Pr: Problem<P> + ?Sized>(
    pr: &Pr,
    z: &mut Vec<Vec<f64>>,
    opts: &NewtonOpts,
) -> Result<NewtonReport, String> {
    newton_with::<P, Pr, _>(pr, z, opts, |z| exact::<P, Pr>(pr, z, opts.par))
}

/// Newton's method with the exact Jacobian formed and factored only every `every` steps and reused in between
/// ([`reuse`]): Shamanskii's method, the chord method for `every` >= `opts.max_iter`, and `newton` bit for bit for
/// `every` = 1. Damped by `opts.damping` as `newton`, the correction rule's simplified corrections solved with the
/// step's factors. With the floor rule on, a reused factorization that gains too little is formed anew at the next
/// step rather than ending the run.
pub fn newton_chord<P: Precision<Native>, Pr: Problem<P> + ?Sized>(
    pr: &Pr,
    z: &mut Vec<Vec<f64>>,
    opts: &NewtonOpts,
    every: usize,
) -> Result<NewtonReport, String> {
    if every == 0 {
        return Err("newton_chord: every must be at least 1".into());
    }
    check_opts(opts)?;
    let mut src = Chord {
        every,
        factor: |z: &[&[f64]]| exact::<P, Pr>(pr, z, opts.par),
        last: None,
        stale: false,
    };
    damped::<P, Pr, Rc<Exact<P>>>(pr, z, opts, &mut src, None)
}

/// The factors of [`newton_chord`]: those of `reuse`, formed at the first step and every `every` steps, and at once
/// when the floor rule finds them stale; fresh only when formed at the step's iterate
struct Chord<G, F> {
    every: usize,
    factor: G,
    last: Option<(usize, Rc<F>)>,
    stale: bool,
}

impl<P: Precision<Native>, F: Factored<P>, G: Fn(&[&[f64]]) -> Result<F, String>> Source<P, Rc<F>> for Chord<G, F> {
    fn get(&mut self, z: &[&[f64]]) -> Result<(Rc<F>, bool), String> {
        if !self.stale {
            if let Some((uses, f)) = self.last.as_mut() {
                if *uses < self.every {
                    *uses += 1;
                    return Ok((f.clone(), false));
                }
            }
        }
        let f = Rc::new((self.factor)(z)?);
        self.last = Some((1, f.clone()));
        self.stale = false;
        Ok((f, true))
    }
    fn stale(&mut self) {
        self.stale = true;
    }
}

/// Newton's method, damped as `newton`, with the exact Jacobian rounded by `to` and factored in double-double or f64
/// at each step ([`Lowered`]): for P = `Quad` or `Octo` it converges to P's accuracy in about as many steps, each at a
/// fraction of the cost.
pub fn newton_lowered<P: Precision<Native>, Pr: Problem<P> + ?Sized>(
    pr: &Pr,
    z: &mut Vec<Vec<f64>>,
    opts: &NewtonOpts,
    to: Lower,
) -> Result<NewtonReport, String> {
    newton_with::<P, Pr, _>(pr, z, opts, |z| lowered::<P, Pr>(pr, z, to, opts.par))
}

/// The damped loop of `newton` with the Newton matrix that `factor` forms and factors at each step's z: the exact
/// Jacobian for `newton`, an approximate one for a mixed-precision Newton (an f64 Jacobian with the residual in P).
pub fn newton_with<P: Precision<Native>, Pr: Problem<P> + ?Sized, F: Factored<P>>(
    pr: &Pr,
    z: &mut Vec<Vec<f64>>,
    opts: &NewtonOpts,
    factor: impl Fn(&[&[f64]]) -> Result<F, String>,
) -> Result<NewtonReport, String> {
    check_opts(opts)?;
    damped::<P, Pr, F>(pr, z, opts, &mut Fresh(factor), None)
}

/// A projection of the state onto a manifold the caller solves itself (P::PARTS part arrays, changed in place), such
/// as an end's data put back on the equations it must meet
pub type Project<'a> = &'a dyn Fn(&mut Vec<Vec<f64>>) -> Result<(), String>;

/// `newton_with` with the projection `project` applied to the start and to every trial (and so to every iterate)
/// before its residual. A trial's simplified correction includes the projection's displacement, the vector from the
/// trial to the next simplified iterate, so that the correction rule's test sees a projection that moves the point
/// away from the root. The identity gives `newton_with`'s steps, bit for bit. A projection may move a point only above
/// a switch of its own (it sees the point, not its residual), as hp's ElimO projects only above its max |F| switch.
pub fn newton_with_projection<P: Precision<Native>, Pr: Problem<P> + ?Sized, F: Factored<P>>(
    pr: &Pr,
    z: &mut Vec<Vec<f64>>,
    opts: &NewtonOpts,
    factor: impl Fn(&[&[f64]]) -> Result<F, String>,
    project: impl Fn(&mut Vec<Vec<f64>>) -> Result<(), String>,
) -> Result<NewtonReport, String> {
    check_opts(opts)?;
    damped::<P, Pr, F>(pr, z, opts, &mut Fresh(factor), Some(&project))
}

/// b - a as scalars of P, a and b part arrays
fn displacement<P: Precision<Native>>(b: &[Vec<f64>], a: &[Vec<f64>]) -> Vec<P::Scalar> {
    let mut x = vec![0.0; P::PARTS];
    let mut y = vec![0.0; P::PARTS];
    (0..a[0].len())
        .map(|i| {
            for q in 0..P::PARTS {
                x[q] = b[q][i];
                y[q] = a[q][i];
            }
            P::from_raw(&x) - P::from_raw(&y)
        })
        .collect()
}

/// The factors a damped loop takes at each step, and whether they were formed at the step's iterate (fresh)
pub(crate) trait Source<P: Precision<Native>, F: Factored<P>> {
    /// the factors for the step at z
    fn get(&mut self, z: &[&[f64]]) -> Result<(F, bool), String>;
    /// the factors in hand gained too little: the next `get` forms new ones
    fn stale(&mut self);
}

/// factors formed at every step's iterate
struct Fresh<G>(G);

impl<P: Precision<Native>, F: Factored<P>, G: Fn(&[&[f64]]) -> Result<F, String>> Source<P, F> for Fresh<G> {
    fn get(&mut self, z: &[&[f64]]) -> Result<(F, bool), String> {
        Ok(((self.0)(z)?, true))
    }
    fn stale(&mut self) {}
}

/// The damped loop of `opts.damping` with the factors of `src` and the projection `proj`
pub(crate) fn damped<P: Precision<Native>, Pr: Problem<P> + ?Sized, F: Factored<P>>(
    pr: &Pr,
    z: &mut Vec<Vec<f64>>,
    opts: &NewtonOpts,
    src: &mut impl Source<P, F>,
    proj: Option<Project>,
) -> Result<NewtonReport, String> {
    if let Some(p) = proj {
        p(z)?;
    }
    match opts.damping {
        Damping::Correction => newton_correction::<P, Pr, F>(pr, z, opts, src, proj),
        Damping::Residual => newton_residual::<P, Pr, F>(pr, z, opts, src, proj),
    }
}

/// The floor rule's test of a full step that took max |F| from f0 to f1 with corrections dz: (cut, max |dz|) when
/// the rule is on, f1 is finite and above `tol`, and the cut f0 / f1 is below `floor_cut`
pub(crate) fn slow<P: Precision<Native>>(opts: &NewtonOpts, f0: f64, f1: f64, dz: &[P::Scalar]) -> Option<(f64, f64)> {
    let cut = f0 / f1;
    // a NaN cut gains too little
    let small = matches!(cut.partial_cmp(&opts.floor_cut), Some(std::cmp::Ordering::Less) | None);
    (opts.floor_cut > 0.0 && f1.is_finite() && f1 > opts.tol && small).then(|| (cut, max_abs::<P>(dz)))
}

/// The error-oriented global Newton method (Deuflhard 2004, section 3.3.3): the prediction `lambda = min(1,
/// |dz_prev| |dz_bar_prev| lambda_prev / (|dz_bar_prev - dz| |dz|))`, the trial's estimate `mu = lambda^2 |dz| / (2
/// |dz_bar - (1 - lambda) dz|)`, a trial accepted when `theta < 1 - lambda / 4` and otherwise retried at
/// `min(mu, lambda / 2)`, and one retry per step at `min(1, mu)` when that is 4 lambda or more, all norms those of
/// `opts.scaling` at the step.
///
/// The floor rule ([`NewtonOpts::floor_cut`]) ends the loop at a full step that gains too little, or that the test
/// refuses when its max |dz| is at most `floor_dz` (no halvings follow then). With factors not formed at the step's
/// iterate it marks them stale instead, so that the next step forms new ones. A refused step with such factors is
/// left out and retried from the same iterate.
fn newton_correction<P: Precision<Native>, Pr: Problem<P> + ?Sized, F: Factored<P>>(
    pr: &Pr,
    z: &mut Vec<Vec<f64>>,
    opts: &NewtonOpts,
    src: &mut impl Source<P, F>,
    proj: Option<Project>,
) -> Result<NewtonReport, String> {
    let mut rep = NewtonReport::new();
    let mut f = pr.residual(&parts(z), opts.par)?;
    let mut prev: Prev<P> = None;
    // the iterate before z, for the weights; z itself at the start
    let mut zp = z.clone();
    for _ in 0..opts.max_iter {
        let fmax = max_abs::<P>(&f);
        rep.fnorm.push(fmax);
        if !fmax.is_finite() {
            rep.stop = Stop::NotFinite;
            return Ok(rep);
        }
        if fmax <= opts.tol {
            rep.converged = true;
            rep.stop = Stop::Tol;
            return Ok(rep);
        }
        let (lu, fresh) = src.get(&parts(z))?;
        let w = weights(opts.scaling, &parts(z), &parts(&zp));
        let t = match correction_step::<P, Pr, F>(pr, &parts(z), &f, &lu, &prev, w.as_deref(), opts, proj)? {
            Trial::Taken(t) => t,
            Trial::Refused | Trial::Floor { .. } if !fresh => {
                // old factors: the step again from z with new ones (one entry of fnorm per step taken)
                src.stale();
                rep.fnorm.pop();
                prev = None;
                continue;
            }
            Trial::Refused => {
                rep.fnorm.push(fmax);
                rep.stop = Stop::Refused;
                return Ok(rep);
            }
            Trial::Floor { f: ft, dz } => {
                rep.fnorm.push(fmax);
                rep.stop = Stop::Floor { cut: fmax / ft, dz };
                return Ok(rep);
            }
        };
        rep.scale.push(t.lam);
        rep.theta.push(t.theta);
        rep.dz_bar.push(t.nbar);
        zp = std::mem::replace(z, t.zt);
        f = t.ft;
        if opts.tol_dz > 0.0 && t.lam == 1.0 && t.nbar <= opts.tol_dz {
            rep.fnorm.push(max_abs::<P>(&f));
            rep.converged = true;
            rep.stop = Stop::TolDz;
            return Ok(rep);
        }
        if t.lam == 1.0 {
            let fnew = max_abs::<P>(&f);
            match slow::<P>(opts, fmax, fnew, &t.dz) {
                Some(_) if !fresh => {
                    src.stale();
                    prev = None;
                    continue;
                }
                Some((cut, dz)) if dz <= opts.floor_dz => {
                    rep.fnorm.push(fnew);
                    rep.stop = Stop::Floor { cut, dz };
                    return Ok(rep);
                }
                _ => {}
            }
        }
        prev = Some((t.dz, t.bar, t.lam));
    }
    let fmax = max_abs::<P>(&f);
    rep.fnorm.push(fmax);
    rep.converged = fmax <= opts.tol;
    rep.stop = if rep.converged { Stop::Tol } else { Stop::MaxIter };
    Ok(rep)
}

/// A step of the correction rule taken from z: its lambda, the Newton correction dz, the trial state z + lambda dz
/// and its residual, and the trial's simplified correction dz_bar, |dz_bar| and theta = |dz_bar| / |dz|
pub(crate) struct Taken<P: Precision<Native>> {
    pub lam: f64,
    pub dz: Vec<P::Scalar>,
    pub zt: Vec<Vec<f64>>,
    pub ft: Vec<P::Scalar>,
    pub bar: Vec<P::Scalar>,
    pub nbar: f64,
    pub theta: f64,
}

/// The outcome of one step of the correction rule
pub(crate) enum Trial<P: Precision<Native>> {
    Taken(Taken<P>),
    /// no lambda above 2^-max_halvings passed the test
    Refused,
    /// the full step refused at the floor rule's max |dz| = dz, the trial's max |F| f (`NewtonOpts::floor_cut`)
    Floor {
        f: f64,
        dz: f64,
    },
}

/// dz, dz_bar and lambda of the last step taken, from which the correction rule predicts the next lambda
pub(crate) type Prev<P> = Option<(
    Vec<<P as Precision<Native>>::Scalar>,
    Vec<<P as Precision<Native>>::Scalar>,
    f64,
)>;

/// One step of the correction rule from z, where the residual is f and `lu` the factors: the trials of
/// `newton_correction`'s documentation until one is accepted, `Trial::Refused` once lambda falls below
/// 2^-max_halvings, and `Trial::Floor` at once when the floor rule is on, the first trial is the full step, it is
/// refused, and max |dz| is at most `floor_dz`; the norms are `wnorm`'s with the weights w (the last step's
/// corrections among them, as NLEQ-ERR rescales them)
#[allow(clippy::too_many_arguments)]
pub(crate) fn correction_step<P: Precision<Native>, Pr: Problem<P> + ?Sized, F: Factored<P>>(
    pr: &Pr,
    z: &[&[f64]],
    f: &[P::Scalar],
    lu: &F,
    prev: &Prev<P>,
    w: Option<&[f64]>,
    opts: &NewtonOpts,
    proj: Option<Project>,
) -> Result<Trial<P>, String> {
    let lam_min = 0.5f64.powi(opts.max_halvings as i32);
    let dz = lu.try_solve_neg(f)?;
    let ndz = wnorm::<P>(&dz, w);
    let mut lam = match prev {
        None => opts.lambda0,
        Some((dz0, bar0, lam0)) => {
            let diff: Vec<P::Scalar> = bar0.iter().zip(&dz).map(|(&a, &b)| a - b).collect();
            let n0 = wnorm::<P>(dz0, w);
            let mu = n0 * wnorm::<P>(bar0, w) * lam0 / (wnorm::<P>(&diff, w) * ndz);
            if mu.is_nan() {
                1.0
            } else {
                mu.min(1.0)
            }
        }
    };
    let mut raised = false;
    let mut first = true;
    loop {
        if lam.is_nan() || lam < lam_min || lam == 0.0 {
            return Ok(Trial::Refused);
        }
        let mut zt = axpy::<P>(z, &dz, lam);
        // the projection's displacement, counted in the simplified correction
        let disp = match proj {
            Some(p) => {
                let before = zt.clone();
                p(&mut zt)?;
                Some(displacement::<P>(&zt, &before))
            }
            None => None,
        };
        let ft = pr.residual(&parts(&zt), opts.par)?;
        let mut bar = lu.try_solve_neg(&ft)?;
        if let Some(d) = disp {
            bar.iter_mut().zip(d).for_each(|(b, d)| *b = *b + d);
        }
        let nbar = wnorm::<P>(&bar, w);
        let theta = nbar / ndz;
        let rest = P::from_dd(DD::from_f64(1.0 - lam));
        let d: Vec<P::Scalar> = bar.iter().zip(&dz).map(|(&b, &d)| b - d * rest).collect();
        let mu = 0.5 * ndz * lam * lam / wnorm::<P>(&d, w);
        if theta.is_nan() || theta >= 1.0 - lam / 4.0 {
            if first && lam == 1.0 && opts.floor_cut > 0.0 {
                let dzmax = max_abs::<P>(&dz);
                if dzmax <= opts.floor_dz {
                    return Ok(Trial::Floor {
                        f: max_abs::<P>(&ft),
                        dz: dzmax,
                    });
                }
            }
            first = false;
            lam = mu.min(lam / 2.0);
            continue;
        }
        let lam_new = mu.min(1.0);
        if !raised && lam < 1.0 && lam_new >= 4.0 * lam {
            raised = true;
            lam = lam_new;
            continue;
        }
        return Ok(Trial::Taken(Taken {
            lam,
            dz,
            zt,
            ft,
            bar,
            nbar,
            theta,
        }));
    }
}

/// The residual rule: steps halved while max |F| does not fall, at most `max_halvings` times, then taken.
///
/// The floor rule ([`NewtonOpts::floor_cut`]) ends the loop at a full step of max |dz| at most `floor_dz` that gains
/// too little, taken when it lowered max |F|. When it did not, the step is left out and no halvings follow. With factors
/// not formed at the step's iterate the rule marks them stale instead, so the next step forms new ones, and a full
/// step they leave without a fall is left out without halvings.
fn newton_residual<P: Precision<Native>, Pr: Problem<P> + ?Sized, F: Factored<P>>(
    pr: &Pr,
    z: &mut Vec<Vec<f64>>,
    opts: &NewtonOpts,
    src: &mut impl Source<P, F>,
    proj: Option<Project>,
) -> Result<NewtonReport, String> {
    let mut rep = NewtonReport::new();
    // the residual at z, kept from the trial that reached z
    let mut f = pr.residual(&parts(z), opts.par)?;
    let mut fn0 = max_abs::<P>(&f);
    let floor_on = opts.floor_cut > 0.0;
    for _ in 0..opts.max_iter {
        rep.fnorm.push(fn0);
        if !fn0.is_finite() {
            rep.stop = Stop::NotFinite;
            return Ok(rep);
        }
        if fn0 <= opts.tol {
            rep.converged = true;
            rep.stop = Stop::Tol;
            return Ok(rep);
        }
        let (lu, fresh) = src.get(&parts(z))?;
        let dz = lu.try_solve_neg(&f)?;
        // the floor rule looks at the full step alone, with no halvings, for old factors or a small step
        let full_only = floor_on && (!fresh || max_abs::<P>(&dz) <= opts.floor_dz);
        let mut s = 1.0;
        let (mut zn, mut ft, mut fnn);
        let mut h = 0;
        loop {
            zn = axpy::<P>(&parts(z), &dz, s);
            if let Some(p) = proj {
                p(&mut zn)?;
            }
            ft = pr.residual(&parts(&zn), opts.par)?;
            fnn = max_abs::<P>(&ft);
            if fnn < fn0 || h == opts.max_halvings || full_only {
                break;
            }
            s *= 0.5;
            h += 1;
        }
        if !fnn.is_finite() {
            rep.fnorm.push(fnn);
            rep.stop = Stop::NotFinite;
            return Ok(rep);
        }
        let fell = fnn < fn0;
        let floor = if s == 1.0 {
            if fell {
                slow::<P>(opts, fn0, fnn, &dz)
            } else {
                (floor_on && fnn > opts.tol).then(|| (fn0 / fnn, max_abs::<P>(&dz)))
            }
        } else {
            None
        };
        if let Some((cut, dzmax)) = floor {
            if !fresh || dzmax <= opts.floor_dz {
                if fell {
                    rep.scale.push(s);
                    *z = zn;
                    f = ft;
                    fn0 = fnn;
                }
                if !fresh {
                    // old factors: the next step forms new ones (one entry of fnorm per step taken)
                    src.stale();
                    if !fell {
                        rep.fnorm.pop();
                    }
                    continue;
                }
                rep.fnorm.push(fn0);
                rep.stop = Stop::Floor { cut, dz: dzmax };
                return Ok(rep);
            }
        }
        rep.scale.push(s);
        *z = zn;
        f = ft;
        fn0 = fnn;
    }
    rep.fnorm.push(fn0);
    rep.converged = fn0 <= opts.tol;
    rep.stop = if rep.converged { Stop::Tol } else { Stop::MaxIter };
    Ok(rep)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Double, DD};

    /// The norm of two entries of 1e160 is 1.414e160, finite. The control is the unscaled sum of squares, which
    /// overflows to infinity, so the scaling is what the check sees.
    #[test]
    fn norm2_does_not_overflow() {
        let v = [DD::from_f64(1e160), DD::from_f64(1e160)];
        let n = norm2::<Double>(&v);
        assert!((n / (1e160 * 2f64.sqrt()) - 1.0).abs() < 1e-15, "norm {n}");
        let naive = v.iter().map(|x| x.hi * x.hi).sum::<f64>().sqrt();
        assert!(naive.is_infinite());
    }

    /// A NaN entry makes the norm NaN, and zeros give zero.
    #[test]
    fn norm2_nan_and_zero() {
        assert!(norm2::<Double>(&[DD::from_f64(1.0), DD::from_f64(f64::NAN)]).is_nan());
        assert_eq!(norm2::<Double>(&[DD::ZERO, DD::ZERO]), 0.0);
    }
}
