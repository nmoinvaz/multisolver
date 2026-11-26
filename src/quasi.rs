//! Newton's method with Broyden's updates in place of new Jacobians: the error-oriented quasi-Newton method QNERR
//! (Deuflhard, Newton Methods for Nonlinear Problems, Springer 2004, chapter 2), between the damped steps of the
//! correction rule ([`crate::newton`]).
//!
//! After a full step s_0 = -J_0^-1 F(x_0) whose contraction theta = |dz_bar| / |s_0| is below `theta_max`, the
//! factors of J_0 are kept and each further step is Broyden's good update J_(k+1) = J_k + F(x_(k+1)) s_k^T / |s_k|^2,
//! applied by Sherman and Morrison without forming it: with v the simplified correction -J_k^-1 F(x_(k+1)) and
//! alpha = s_k . v / |s_k|^2, the step is s_(k+1) = v / (1 - alpha), and J_(k+1)^-1 = (I + s_(k+1) s_k^T / |s_k|^2)
//! J_k^-1. So the simplified correction at x_(k+2) is -J_0^-1 F(x_(k+2)), one solve with J_0's factors, followed by
//! the updates v += s_i (s_(i-1) . v) / |s_(i-1)|^2 for i = 1..k+1. Its contraction Theta = |v| / |s_(k+1)| bounds
//! |alpha| by Cauchy and Schwarz, so while Theta < 1/2 the steps shrink, |s_(k+2)| <= Theta / (1 - Theta) |s_(k+1)|.
//! Each such step costs one residual, one solve with J_0's factors and O(k n) operations in P, and no Jacobian.
//!
//! The updates learn the Jacobian's change one step direction at a time. Where it is confined to a few directions
//! (a few unknowns enter nonlinearly) the convergence soon becomes superlinear, as a secant's; where it is spread over
//! all n, the contraction stays near the chord method's for about n steps (on a linear system Broyden's method ends
//! within 2 n steps; Gay, SIAM J. Numer. Anal. 16 (1979) 623). The method pays where a Jacobian costs many residuals.
//!
//! A step whose Theta reaches `theta_max` is refused: the state stays where it was and a damped Newton step with new
//! factors follows there. After `max_updates` updates the factors are formed anew in the same way, and so they are
//! once a step's |s|^2 leaves f64's range (a step below about 1e-154), where 1 / |s|^2 is not finite. The
//! updates' coefficients are formed in P, so the steps carry P's accuracy; the factors may be approximate (an f64
//! LU, [`crate::newton::Lowered`]), whose mismatch the updates then correct along the steps.
//!
//! With [`crate::newton::Scaling::Iterates`] the norms and the inner products of the updates are the scaled ones,
//! sum_i (a_i / w_i) (b_i / w_i), with the weights of the damped step that started them, fixed while they last (as
//! QNERR of ZIB NewtonLib keeps its scale): Broyden's good update in that inner product, so that the steps of a
//! problem do not depend on the units of its unknowns.
use crate::gcr::recip;
use crate::newton::{
    axpy, check_opts, correction_step, slow, weights, wnorm, Damping, Factored, NewtonOpts, NewtonReport, Prev, Stop,
    Trial,
};
use crate::prec::{max_abs, parts};
use crate::Problem;
use multidouble::dd::DD;
use multidouble::precision::Precision;
use multidouble::simd::Native;

/// The settings of [`newton_quasi`]
#[derive(Clone, Copy, Debug)]
pub struct QuasiOpts {
    /// the largest contraction with which a full Newton step starts the updates and an update's step is taken, in
    /// (0, 1/2]: past 1/2 the steps need not shrink
    pub theta_max: f64,
    /// updates on one set of factors at most, each keeping a vector of P; then the factors are formed anew (0: plain
    /// Newton)
    pub max_updates: usize,
}

impl Default for QuasiOpts {
    fn default() -> Self {
        QuasiOpts {
            theta_max: 0.5,
            max_updates: 20,
        }
    }
}

/// What [`newton_quasi`] did besides its [`NewtonReport`]: the factorizations formed, the steps taken by Broyden's
/// updates, and the updates' steps refused
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QuasiInfo {
    pub factorizations: usize,
    pub quasi_steps: usize,
    pub refused: usize,
}

/// The updates' state: J_0's factors, the steps s_0..s_k taken since they were formed with 1 / |s_i|^2 in P, the
/// simplified correction v = -J_k^-1 F at the current state, and the weights w of the norms with 1 / w_i in P (None:
/// the 2-norm)
struct Updates<P: Precision<Native>, F> {
    lu: F,
    s: Vec<Vec<P::Scalar>>,
    inv: Vec<P::Scalar>,
    v: Vec<P::Scalar>,
    w: Option<Vec<f64>>,
    r: Option<Vec<P::Scalar>>,
}

/// a . b in P's accumulator, sum_i (a_i r_i) (b_i r_i) with the reciprocal weights r
fn dot<P: Precision<Native>>(a: &[P::Scalar], b: &[P::Scalar], r: Option<&[P::Scalar]>) -> P::Scalar {
    match r {
        None => P::dot(a.iter().copied().zip(b.iter().copied())),
        Some(r) => P::dot(a.iter().zip(b).zip(r).map(|((&x, &y), &ri)| (x * ri, y * ri))),
    }
}

/// whether x is finite
fn finite<P: Precision<Native>>(x: P::Scalar) -> bool {
    P::to_dd(x).hi.is_finite()
}

/// Newton's method from z (P::PARTS part arrays, updated in place) with Broyden's updates between its factorizations,
/// the module's documentation: the damped steps are those of [`crate::newton::newton_with`] with the correction rule
/// and the Newton matrix `factor` forms, and the report records every step, the updates' with lambda 1, their
/// contraction Theta and |v|. `opts.tol_dz` ends the run after a step whose simplified correction is at most it, as
/// it ends `newton`'s. With `max_updates` 0 it is `newton_with` bit for bit. `opts.damping` must be
/// `Damping::Correction`, whose simplified corrections the updates continue; `opts.scaling` sets the norms of both.
pub fn newton_quasi<P: Precision<Native>, Pr: Problem<P> + ?Sized, F: Factored<P>>(
    pr: &Pr,
    z: &mut Vec<Vec<f64>>,
    opts: &NewtonOpts,
    qopts: &QuasiOpts,
    factor: impl Fn(&[&[f64]]) -> Result<F, String>,
) -> Result<(NewtonReport, QuasiInfo), String> {
    if opts.damping != Damping::Correction {
        return Err("newton_quasi: the updates continue the correction rule's simplified corrections".into());
    }
    if !(qopts.theta_max > 0.0 && qopts.theta_max <= 0.5) {
        return Err(format!(
            "newton_quasi: theta_max must lie in (0, 1/2], not {}",
            qopts.theta_max
        ));
    }
    check_opts(opts)?;
    let mut rep = NewtonReport::new();
    let mut info = QuasiInfo::default();
    let mut f = pr.residual(&parts(z), opts.par)?;
    let mut prev: Prev<P> = None;
    let mut up: Option<Updates<P, F>> = None;
    // the iterate before z, for the damped steps' weights; z itself at the start
    let mut zp = z.clone();
    for _ in 0..opts.max_iter {
        let fmax = max_abs::<P>(&f);
        rep.fnorm.push(fmax);
        if !fmax.is_finite() {
            rep.stop = Stop::NotFinite;
            return Ok((rep, info));
        }
        if fmax <= opts.tol {
            rep.converged = true;
            rep.stop = Stop::Tol;
            return Ok((rep, info));
        }
        if let Some(u) = up.as_mut() {
            // s_(k+1) = v / (1 - alpha), alpha = s_k . v / |s_k|^2, |alpha| < 1/2
            let k = u.s.len() - 1;
            let (w, r) = (u.w.as_deref(), u.r.as_deref());
            let alpha = dot::<P>(&u.s[k], &u.v, r) * u.inv[k];
            let c = recip::<P>(P::ONE - alpha);
            let s: Vec<P::Scalar> = u.v.iter().map(|&x| x * c).collect();
            let zt = axpy::<P>(&parts(z), &s, 1.0);
            let ft = pr.residual(&parts(&zt), opts.par)?;
            // -J_(k+1)^-1 F(z + s): J_0's solve, then the updates i = 1..=k+1
            let mut v = u.lu.try_solve_neg(&ft)?;
            u.inv.push(recip::<P>(dot::<P>(&s, &s, r)));
            u.s.push(s);
            for i in 1..u.s.len() {
                let c = dot::<P>(&u.s[i - 1], &v, r) * u.inv[i - 1];
                v.iter_mut().zip(&u.s[i]).for_each(|(x, &si)| *x = *x + c * si);
            }
            let nv = wnorm::<P>(&v, w);
            let theta = nv / wnorm::<P>(&u.s[k + 1], w);
            if theta < qopts.theta_max {
                rep.scale.push(1.0);
                rep.theta.push(theta);
                rep.dz_bar.push(nv);
                info.quasi_steps += 1;
                zp = std::mem::replace(z, zt);
                f = ft;
                u.v = v;
                if opts.tol_dz > 0.0 && nv <= opts.tol_dz {
                    rep.fnorm.push(max_abs::<P>(&f));
                    rep.converged = true;
                    rep.stop = Stop::TolDz;
                    return Ok((rep, info));
                }
                if u.s.len() > qopts.max_updates || !finite::<P>(u.inv[k + 1]) {
                    up = None;
                }
                continue;
            }
            // refused: a damped step with new factors from the same state
            info.refused += 1;
            up = None;
        }
        let lu = factor(&parts(z))?;
        info.factorizations += 1;
        let w = weights(opts.scaling, &parts(z), &parts(&zp));
        let t = match correction_step::<P, Pr, F>(pr, &parts(z), &f, &lu, &prev, w.as_deref(), opts, None)? {
            Trial::Taken(t) => t,
            Trial::Refused => {
                rep.fnorm.push(fmax);
                rep.stop = Stop::Refused;
                return Ok((rep, info));
            }
            Trial::Floor { f: ft, dz } => {
                rep.fnorm.push(fmax);
                rep.stop = Stop::Floor { cut: fmax / ft, dz };
                return Ok((rep, info));
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
            return Ok((rep, info));
        }
        // the floor rule on a full step with factors formed at its iterate, as `newton_correction` applies it
        if t.lam == 1.0 {
            let fnew = max_abs::<P>(&f);
            if let Some((cut, dz)) = slow::<P>(opts, fmax, fnew, &t.dz) {
                if dz <= opts.floor_dz {
                    rep.fnorm.push(fnew);
                    rep.stop = Stop::Floor { cut, dz };
                    return Ok((rep, info));
                }
            }
        }
        let r: Option<Vec<P::Scalar>> = w
            .as_ref()
            .map(|w| w.iter().map(|&x| P::from_dd(DD::from_f64(1.0 / x))).collect());
        let inv = recip::<P>(dot::<P>(&t.dz, &t.dz, r.as_deref()));
        if qopts.max_updates > 0 && t.lam == 1.0 && t.theta < qopts.theta_max && finite::<P>(inv) {
            up = Some(Updates {
                lu,
                inv: vec![inv],
                s: vec![t.dz],
                v: t.bar,
                w,
                r,
            });
            prev = None;
        } else {
            prev = Some((t.dz, t.bar, t.lam));
        }
    }
    let fmax = max_abs::<P>(&f);
    rep.fnorm.push(fmax);
    rep.converged = fmax <= opts.tol;
    rep.stop = if rep.converged { Stop::Tol } else { Stop::MaxIter };
    Ok((rep, info))
}
