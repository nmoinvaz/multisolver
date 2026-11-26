//! Pseudo-arclength continuation of a branch of F(z, p) = 0 ([`Family`]) through its folds (Keller, in Applications
//! of Bifurcation Theory, Academic Press 1977, 359), and the location of a fold.
//!
//! The unknowns are w = (z, p), n + 1 of them. From a point w_0 of the branch with the unit tangent t_0 (J t_z + F_p t_p
//! = 0, |t_0| = 1), the predictor is w_0 + ds t_0 and the corrector is Newton's method on the bordered system
//! ([`Bordered`]) F(z, p) = 0, t_0 . (w - w_0 - ds t_0) = 0: the branch cut by the hyperplane through the prediction
//! orthogonal to t_0. Its Jacobian [[J, F_p], [t_0^T]] stays nonsingular at a simple fold, where J is singular and
//! dz/dp infinite, so the continuation passes the fold where continuation in p stops. The tangent at the corrected
//! point solves the same bordered matrix there with the right-hand side e_(n+1), so t . t_0 = 1 keeps the direction of
//! travel. The solve is refined against the bordered matrix in P, so that factors in double-double or f64 give the
//! tangent to P's accuracy, and t is scaled to |t| = 1 in f64, which leaves its direction exact to P and its length to
//! f64's rounding.
//!
//! The corrector is the local Newton method: a step is refused, and ds halved, when the corrector does not converge,
//! damps a step (lambda < 1, unless `accept_damped`), contracts at first by theta_0 >= `theta_max`, or cannot form or
//! factor its Newton matrix (a singular one at the prediction). [`step_from`] corrects from a prediction the caller
//! gives, such as a series' sum, and can leave the end's tangent to the caller. After a step taken the next ds is ds (theta_target / theta_0)^(1/2), within
//! [ds / 2, 2 ds] and [`ds_min`, `ds_max`], since the tangent predictor's error, and theta_0 with it, grow as ds^2.
//! The rule needs theta_0, so the corrector damps by the correction rule.
//!
//! [`ds_min`]: ArcOpts::ds_min
//! [`ds_max`]: ArcOpts::ds_max
//!
//! A fold lies between two points where the tangent's p component t_p changes sign. [`locate_fold`] finds the point of
//! t_p = 0 between them by the Illinois variant of regula falsi in the step length s from the first point, s in P and
//! each s corrected as a step: near a simple fold p - p_fold is of order (s - s_fold)^2, so p_fold carries about twice
//! the digits of s_fold.
use crate::gcr::recip;
use crate::newton::{newton_with, norm2, Damping, Factored, NewtonOpts, NewtonReport};
use crate::prec::parts;
use crate::{Family, Problem};
use multidouble::dd::DD;
use multidouble::linalg::ld_for;
use multidouble::precision::Precision;
use multidouble::simd::Native;
use std::cell::Cell;

/// the scalars of a state's part arrays
fn scalars<P: Precision<Native>>(w: &[&[f64]]) -> Vec<P::Scalar> {
    let mut buf = vec![0.0; P::PARTS];
    (0..w[0].len())
        .map(|i| {
            for (q, b) in buf.iter_mut().enumerate() {
                *b = w[q][i];
            }
            P::from_raw(&buf)
        })
        .collect()
}

/// part arrays of scalars
fn to_parts<P: Precision<Native>>(v: &[P::Scalar]) -> Vec<Vec<f64>> {
    (0..P::PARTS)
        .map(|q| v.iter().map(|&x| P::part(x, q)).collect())
        .collect()
}

/// z (the first n entries of w's part arrays) and p (the last)
fn split<'w, P: Precision<Native>>(w: &[&'w [f64]], n: usize) -> (Vec<&'w [f64]>, P::Scalar) {
    let p: Vec<f64> = w.iter().map(|q| q[n]).collect();
    (w.iter().map(|q| &q[..n]).collect(), P::from_raw(&p))
}

/// w = (z, p) as part arrays
fn join<P: Precision<Native>>(z: &[&[f64]], p: P::Scalar) -> Vec<Vec<f64>> {
    z.iter()
        .enumerate()
        .map(|(q, zq)| zq.iter().copied().chain([P::part(p, q)]).collect())
        .collect()
}

/// A point of a branch: the state z (P::PARTS part arrays of length n), the parameter p, and the unit tangent
/// t = (t_z, t_p) of the branch there (n + 1 scalars), oriented along the direction of travel
#[derive(Clone, Debug)]
pub struct Point<P: Precision<Native>> {
    pub z: Vec<Vec<f64>>,
    pub p: P::Scalar,
    pub t: Vec<P::Scalar>,
}

impl<P: Precision<Native>> Point<P> {
    /// w = (z, p), n + 1 scalars
    pub fn w(&self) -> Vec<P::Scalar> {
        let mut w = scalars::<P>(&parts(&self.z));
        w.push(self.p);
        w
    }

    /// the tangent's p component t_p, whose sign changes at a fold
    pub fn tp(&self) -> P::Scalar {
        self.t[self.t.len() - 1]
    }
}

/// The member of a family at a fixed parameter, a [`Problem`] in z: such as for the first point of a branch, solved
/// by [`crate::newton::newton`]
pub struct AtParam<'a, P: Precision<Native>, Fa: Family<P> + ?Sized> {
    pub fam: &'a Fa,
    pub p: P::Scalar,
}

impl<P: Precision<Native>, Fa: Family<P> + ?Sized> Problem<P> for AtParam<'_, P, Fa> {
    fn len(&self) -> usize {
        self.fam.len()
    }
    fn residual(&self, z: &[&[f64]], par: bool) -> Result<Vec<P::Scalar>, String> {
        self.fam.residual_at(z, self.p, par)
    }
    fn jacobian(&self, z: &[&[f64]], par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
        self.fam.jacobian_at(z, self.p, par)
    }
}

/// The corrector's system in w = (z, p) (P::PARTS part arrays of length n + 1, p last): F(z, p) and t . (w - pred),
/// the hyperplane through `pred` orthogonal to `t`. Its Jacobian is the bordered [[J, F_p], [t^T]].
pub struct Bordered<'a, P: Precision<Native>, Fa: Family<P> + ?Sized> {
    pub fam: &'a Fa,
    /// the hyperplane's normal, the last tangent (n + 1 scalars)
    pub t: Vec<P::Scalar>,
    /// the point the hyperplane passes through, the prediction (n + 1 scalars)
    pub pred: Vec<P::Scalar>,
}

impl<P: Precision<Native>, Fa: Family<P> + ?Sized> Problem<P> for Bordered<'_, P, Fa> {
    fn len(&self) -> usize {
        self.fam.len() + 1
    }
    fn residual(&self, w: &[&[f64]], par: bool) -> Result<Vec<P::Scalar>, String> {
        let n = self.fam.len();
        let (z, p) = split::<P>(w, n);
        let mut f = self.fam.residual_at(&z, p, par)?;
        if f.len() != n {
            return Err(format!("Bordered: {} residual rows, not {n}", f.len()));
        }
        let d: Vec<P::Scalar> = scalars::<P>(w).iter().zip(&self.pred).map(|(&a, &b)| a - b).collect();
        f.push(P::dot(self.t.iter().copied().zip(d)));
        Ok(f)
    }
    fn jacobian(&self, w: &[&[f64]], par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
        let n = self.fam.len();
        let (z, p) = split::<P>(w, n);
        let (c, ld0) = self.fam.jacobian_at(&z, p, par)?;
        let fp = self.fam.dp_at(&z, p, par)?;
        if c.len() != P::PARTS || fp.len() != n {
            return Err(format!(
                "Bordered: a Jacobian of {} parts and dF/dp of {} rows, not {} and {n}",
                c.len(),
                fp.len(),
                P::PARTS
            ));
        }
        let ld = ld_for(n + 1);
        let mut out = vec![vec![0.0; ld * (n + 1)]; P::PARTS];
        for (q, (oq, cq)) in out.iter_mut().zip(&c).enumerate() {
            for j in 0..n {
                oq[j * ld..j * ld + n].copy_from_slice(&cq[j * ld0..j * ld0 + n]);
                oq[j * ld + n] = P::part(self.t[j], q);
            }
            for (i, &v) in fp.iter().enumerate() {
                oq[n * ld + i] = P::part(v, q);
            }
            oq[n * ld + n] = P::part(self.t[n], q);
        }
        Ok((out, ld))
    }
}

/// The unit tangent of the branch at its point (z, p), oriented by d (n + 1 scalars, not orthogonal to it): the
/// solution of [[J, F_p], [d^T]] t = e_(n+1), the bordered matrix M factored by `factor` at w = (z, p), the solve
/// refined against M in P until a correction no longer halves (at most 4 P::PARTS times), scaled to |t| = 1 in f64.
/// The refinement takes M's exact parts from one more evaluation of the Jacobian and dF/dp.
pub fn tangent<P: Precision<Native>, Fa: Family<P> + ?Sized, F: Factored<P>>(
    fam: &Fa,
    z: &[&[f64]],
    p: P::Scalar,
    d: &[P::Scalar],
    par: bool,
    factor: impl Fn(&Bordered<'_, P, Fa>, &[&[f64]]) -> Result<F, String>,
) -> Result<Vec<P::Scalar>, String> {
    let n = fam.len();
    let b = Bordered {
        fam,
        t: d.to_vec(),
        pred: vec![P::ZERO; n + 1],
    };
    let w = join::<P>(z, p);
    let lu = factor(&b, &parts(&w))?;
    // M^-1 e_(n+1) as the factors' -M^-1 of -e_(n+1)
    let mut e = vec![P::ZERO; n + 1];
    e[n] = -P::ONE;
    let mut t = lu.try_solve_neg(&e)?;
    // iterative refinement: the correction -M^-1 (M t - e_(n+1)), M t in P from M's parts
    let (c, ld) = b.jacobian(&parts(&w), par)?;
    let mut buf = vec![0.0; P::PARTS];
    let mut last = f64::INFINITY;
    for _ in 0..4 * P::PARTS {
        let r: Vec<P::Scalar> = (0..=n)
            .map(|i| {
                let mt = P::dot((0..=n).map(|j| {
                    for (q, x) in buf.iter_mut().enumerate() {
                        *x = c[q][j * ld + i];
                    }
                    (P::from_raw(&buf), t[j])
                }));
                mt + e[i]
            })
            .collect();
        let dt = lu.try_solve_neg(&r)?;
        let nd = norm2::<P>(&dt);
        t.iter_mut().zip(&dt).for_each(|(x, &y)| *x = *x + y);
        if nd.is_nan() || nd >= last / 2.0 {
            break;
        }
        last = nd;
    }
    let nrm = norm2::<P>(&t);
    if !(nrm > 0.0 && nrm.is_finite()) {
        return Err(format!("tangent: a tangent of norm {nrm}"));
    }
    let s = P::from_dd(DD::from_f64(1.0 / nrm));
    Ok(t.iter().map(|&x| x * s).collect())
}

/// The first point of a branch at a solution (z, p) of the family, its tangent ([`tangent`]) oriented by `dir` in p
/// (+1: along increasing p), the bordered matrix factored by `factor`
pub fn start<P: Precision<Native>, Fa: Family<P> + ?Sized, F: Factored<P>>(
    fam: &Fa,
    z: &[&[f64]],
    p: P::Scalar,
    dir: f64,
    par: bool,
    factor: impl Fn(&Bordered<'_, P, Fa>, &[&[f64]]) -> Result<F, String>,
) -> Result<Point<P>, String> {
    let n = fam.len();
    let mut d = vec![P::ZERO; n + 1];
    d[n] = P::from_dd(DD::from_f64(dir));
    let t = tangent::<P, Fa, F>(fam, z, p, &d, par, factor)?;
    Ok(Point {
        z: z.iter().map(|q| q.to_vec()).collect(),
        p,
        t,
    })
}

/// One step of length ds from `at`: the prediction w + ds t, the corrector `newton_with` on [`Bordered`] from it with
/// `opts` and the factors `factor` forms, and the tangent at the corrected point oriented by t. Returns the
/// corrector's report and, when it converged, the new point; no point, and an empty report, when `factor` fails (a
/// Newton matrix singular at a guess, or a Jacobian the family cannot form there). The residual's errors are
/// returned.
pub fn step<P: Precision<Native>, Fa: Family<P> + ?Sized, F: Factored<P>>(
    fam: &Fa,
    at: &Point<P>,
    ds: P::Scalar,
    opts: &NewtonOpts,
    factor: &impl Fn(&Bordered<'_, P, Fa>, &[&[f64]]) -> Result<F, String>,
) -> Result<(NewtonReport, Option<Point<P>>), String> {
    let pred: Vec<P::Scalar> = at.w().iter().zip(&at.t).map(|(&a, &t)| a + ds * t).collect();
    step_from::<P, Fa, F>(fam, at, &pred, true, opts, factor)
}

/// [`step`] from the caller's prediction `pred` (n + 1 scalars, such as a series' sum, far closer to the branch than
/// the tangent's): the corrector on [`Bordered`] with the hyperplane through `pred` orthogonal to `at`'s tangent, from
/// `pred`. With `end_tangent` the new point's tangent is formed at it ([`tangent`], oriented by `at`'s); without, the
/// point carries `at`'s tangent, for a caller that takes the next one from elsewhere (the next base's series) and
/// saves the bordered factorization and the Jacobian the tangent costs. `step` is this with pred = w + ds t and the
/// end tangent, bit for bit.
pub fn step_from<P: Precision<Native>, Fa: Family<P> + ?Sized, F: Factored<P>>(
    fam: &Fa,
    at: &Point<P>,
    pred: &[P::Scalar],
    end_tangent: bool,
    opts: &NewtonOpts,
    factor: &impl Fn(&Bordered<'_, P, Fa>, &[&[f64]]) -> Result<F, String>,
) -> Result<(NewtonReport, Option<Point<P>>), String> {
    let n = fam.len();
    if pred.len() != n + 1 {
        return Err(format!(
            "step_from: a prediction of {} scalars, not {}",
            pred.len(),
            n + 1
        ));
    }
    let pred = pred.to_vec();
    let b = Bordered {
        fam,
        t: at.t.clone(),
        pred: pred.clone(),
    };
    let mut w = to_parts::<P>(&pred);
    let failed = Cell::new(false);
    let rep = match newton_with::<P, _, F>(&b, &mut w, opts, |w| factor(&b, w).inspect_err(|_| failed.set(true))) {
        Ok(rep) => rep,
        Err(_) if failed.get() => return Ok((NewtonReport::new(), None)),
        Err(e) => return Err(e),
    };
    if !rep.converged {
        return Ok((rep, None));
    }
    let wp = parts(&w);
    let (z, p) = split::<P>(&wp, n);
    let t = if end_tangent {
        tangent::<P, Fa, F>(fam, &z, p, &at.t, opts.par, factor)?
    } else {
        at.t.clone()
    };
    let z = z.iter().map(|q| q.to_vec()).collect();
    Ok((rep, Some(Point { z, p, t })))
}

/// The settings of [`continuation`]
#[derive(Clone, Copy, Debug)]
pub struct ArcOpts {
    /// the first step length
    pub ds: f64,
    /// the shortest step: a step taken leaves the next at least this, and the continuation stops when a refused step
    /// would need a shorter one
    pub ds_min: f64,
    /// the longest step
    pub ds_max: f64,
    /// the corrector's first contraction the step length aims at
    pub theta_target: f64,
    /// the corrector's first contraction from which a step is refused
    pub theta_max: f64,
    /// points at most, the start included
    pub max_points: usize,
    /// take a step whose corrector damped (some lambda below 1), as long as its first contraction, normalized to a
    /// full step's when that step was damped, is below `theta_max`; false, the default, refuses it
    pub accept_damped: bool,
    /// the corrector's settings, with the correction rule's damping (`Damping::Correction`)
    pub newton: NewtonOpts,
}

impl Default for ArcOpts {
    fn default() -> Self {
        ArcOpts {
            ds: 0.1,
            ds_min: 1e-8,
            ds_max: 1.0,
            theta_target: 0.125,
            theta_max: 0.5,
            max_points: 100,
            accept_damped: false,
            newton: NewtonOpts::default(),
        }
    }
}

/// A branch as [`continuation`] followed it: its points, the step length from each to the next and the corrector's
/// report of that step, the folds (each i where t_p changes sign between points i and i + 1), the steps refused, and
/// whether it stopped at a step below `ds_min` (else at `max_points` or `stop`)
#[derive(Clone, Debug)]
pub struct Path<P: Precision<Native>> {
    pub points: Vec<Point<P>>,
    pub ds: Vec<f64>,
    pub reports: Vec<NewtonReport>,
    pub folds: Vec<usize>,
    pub refused: usize,
    pub stalled: bool,
}

/// Pseudo-arclength continuation from `start` with the step control of the module's documentation, the bordered
/// matrices factored by `factor` (such as `|b, w| exact(b, w, par)`, or [`crate::newton::lowered`]), until
/// `opts.max_points` points, a step below `opts.ds_min`, or a point for which `stop` returns true (that point kept;
/// `start` among them, which ends the continuation before its first step).
pub fn continuation<P: Precision<Native>, Fa: Family<P> + ?Sized, F: Factored<P>>(
    fam: &Fa,
    start: Point<P>,
    opts: &ArcOpts,
    factor: impl Fn(&Bordered<'_, P, Fa>, &[&[f64]]) -> Result<F, String>,
    mut stop: impl FnMut(&Point<P>) -> bool,
) -> Result<Path<P>, String> {
    if !(opts.ds_min > 0.0 && opts.ds_min <= opts.ds && opts.ds <= opts.ds_max && opts.ds_max.is_finite()) {
        return Err(format!(
            "continuation: need 0 < ds_min <= ds <= ds_max < inf, not {} {} {}",
            opts.ds_min, opts.ds, opts.ds_max
        ));
    }
    if !(opts.theta_target > 0.0 && opts.theta_target < opts.theta_max && opts.theta_max <= 1.0) {
        return Err(format!(
            "continuation: need 0 < theta_target < theta_max <= 1, not {} {}",
            opts.theta_target, opts.theta_max
        ));
    }
    if opts.newton.damping != Damping::Correction {
        return Err("continuation: the step control needs the correction rule's contraction theta_0".into());
    }
    let mut path = Path {
        points: vec![start],
        ds: Vec::new(),
        reports: Vec::new(),
        folds: Vec::new(),
        refused: 0,
        stalled: false,
    };
    if stop(&path.points[0]) {
        return Ok(path);
    }
    let mut ds = opts.ds;
    while path.points.len() < opts.max_points {
        let at = &path.points[path.points.len() - 1];
        let (rep, next) = step::<P, Fa, F>(fam, at, P::from_dd(DD::from_f64(ds)), &opts.newton, &factor)?;
        let full = rep.scale.iter().all(|&l| l == 1.0);
        let theta0 = first_contraction(&rep, opts.accept_damped);
        match next {
            Some(pt) if (full || opts.accept_damped) && theta0 < opts.theta_max => {
                let pos = |x: P::Scalar| P::to_dd(x).hi > 0.0;
                if pos(at.tp()) != pos(pt.tp()) {
                    path.folds.push(path.points.len() - 1);
                }
                path.ds.push(ds);
                path.reports.push(rep);
                let done = stop(&pt);
                path.points.push(pt);
                if done {
                    break;
                }
                let grow = if theta0 > 0.0 {
                    (opts.theta_target / theta0).sqrt().clamp(0.5, 2.0)
                } else {
                    2.0
                };
                ds = (ds * grow).clamp(opts.ds_min, opts.ds_max);
            }
            _ => {
                path.refused += 1;
                ds /= 2.0;
                if ds < opts.ds_min {
                    path.stalled = true;
                    break;
                }
            }
        }
    }
    Ok(path)
}

/// The first step's contraction theta_0, the step rule's measure of the prediction's error. A damped first step (lambda
/// < 1) with `accept_damped` gives the contraction its full step would have, (theta_0 - (1 - lambda)) / lambda^2 (at
/// least 0): its simplified correction is (1 - lambda) dz plus a term of lambda^2 / 2 omega |dz|^2, so that theta_0 at
/// lambda alone would refuse every step damped to lambda <= 1 - theta_max (Deuflhard 2004, section 3.3)
fn first_contraction(rep: &NewtonReport, accept_damped: bool) -> f64 {
    let theta = rep.theta.first().copied().unwrap_or(0.0);
    match rep.scale.first() {
        Some(&lam) if accept_damped && lam < 1.0 => ((theta - (1.0 - lam)) / (lam * lam)).max(0.0),
        _ => theta,
    }
}

/// The settings of [`locate_fold`]
#[derive(Clone, Copy, Debug)]
pub struct FoldOpts {
    /// stop at a point with |t_p| at most this: near the fold z moves by about |t_p| and p by about |t_p|^2
    pub tol: f64,
    /// corrections at most
    pub max_iter: usize,
}

impl Default for FoldOpts {
    fn default() -> Self {
        FoldOpts {
            tol: 1e-15,
            max_iter: 50,
        }
    }
}

/// The fold between the point `a` and the point a step of `ds` from it reaches (such as `path.points[i]` and
/// `path.ds[i]` for i in `path.folds`): the point of the branch with t_p = 0, by the Illinois variant of regula falsi
/// on the step length s in [0, ds], in P, each s corrected from `a` by [`step`] with `newton` and `factor`. Returns
/// the point, with |t_p| <= `opts.tol`, and the corrections it took; an error when t_p does not change sign over the
/// step, a correction fails, or `opts.max_iter` corrections do not reach the tolerance.
pub fn locate_fold<P: Precision<Native>, Fa: Family<P> + ?Sized, F: Factored<P>>(
    fam: &Fa,
    a: &Point<P>,
    ds: f64,
    opts: &FoldOpts,
    newton: &NewtonOpts,
    factor: impl Fn(&Bordered<'_, P, Fa>, &[&[f64]]) -> Result<F, String>,
) -> Result<(Point<P>, usize), String> {
    let hi = |x: P::Scalar| P::to_dd(x).hi;
    let at = |s: P::Scalar| -> Result<Point<P>, String> {
        match step::<P, Fa, F>(fam, a, s, newton, &factor)? {
            (_, Some(pt)) => Ok(pt),
            (_, None) => Err(format!(
                "locate_fold: the corrector did not converge at s = {:e}",
                hi(s)
            )),
        }
    };
    let (mut s0, mut g0) = (P::ZERO, a.tp());
    let mut s1 = P::from_dd(DD::from_f64(ds));
    let mut g1 = at(s1)?.tp();
    if (hi(g0) > 0.0) == (hi(g1) > 0.0) {
        return Err(format!(
            "locate_fold: t_p is {:e} and {:e} at the ends, no change of sign",
            hi(g0),
            hi(g1)
        ));
    }
    let half = P::from_dd(DD::from_f64(0.5));
    // the end replaced last: -1 for s1, +1 for s0; the end kept twice in a row has its value halved (Illinois)
    let mut side = 0i32;
    for it in 1..=opts.max_iter {
        let s = (g0 * s1 - g1 * s0) * recip::<P>(g0 - g1);
        let pt = at(s)?;
        let g = pt.tp();
        if hi(g).abs() <= opts.tol {
            return Ok((pt, it));
        }
        if (hi(g) > 0.0) == (hi(g1) > 0.0) {
            (s1, g1) = (s, g);
            if side == -1 {
                g0 = g0 * half;
            }
            side = -1;
        } else {
            (s0, g0) = (s, g);
            if side == 1 {
                g1 = g1 * half;
            }
            side = 1;
        }
    }
    Err(format!(
        "locate_fold: |t_p| above {:e} after {} corrections",
        opts.tol, opts.max_iter
    ))
}
