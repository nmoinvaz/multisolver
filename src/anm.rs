//! The asymptotic numerical method (ANM): the Taylor series z(h) = sum_k z_k h^k of a solution branch of
//! F(z, p0 + h) = 0 in the problem's parameter, from a solution z_0 at p0 (Cochelin, Comput. Struct. 53 (1994) 1181;
//! Cochelin, Damil and Potier-Ferry, Int. J. Numer. Methods Eng. 37 (1994) 1187).
//!
//! Order k of F(z(h), p0 + h) is J z_k plus what the lower orders give, with J the Jacobian at z_0, so each
//! coefficient solves J z_k = -F_k, where F_k is order k of the residual's series with z_k left out, taken one order
//! at a time from the problem's [`ResidualJet::jet_orders`]. One factorization of J serves every order. With `refine` > 0 each solve is refined by the exact
//! products J x ([`Jvp`]): the residual b - J x, solved with the same factors, corrects x.
//!
//! The series is valid up to about [`Branch::radius`], Cochelin's estimate (delta |z_1| / |z_N|)^(1 / (N - 1)), where
//! the last term would be delta times the first. [`chain`] continues a branch by such steps, each corrected by
//! Newton's method at its end, with the step control of hp's anm_run.py.
use crate::newton::{exact, Factored};
use crate::prec::{max_abs, parts};
use crate::{Jvp, ResidualJet};
use multidouble::dd::DD;
use multidouble::precision::Precision;
use multidouble::simd::Native;

/// The Taylor coefficients z_0..z_N of a branch, and for each order k >= 1 the relative residual |J z_k + F_k| /
/// |F_k| of its solve after refinement (max norms, NaN where either holds a NaN, and NaN without refinement)
#[derive(Clone, Debug)]
pub struct Branch<P: Precision<Native>> {
    pub z: Vec<Vec<P::Scalar>>,
    pub residual: Vec<f64>,
}

/// part arrays of scalars
fn to_parts<P: Precision<Native>>(v: &[P::Scalar]) -> Vec<Vec<f64>> {
    (0..P::PARTS)
        .map(|q| v.iter().map(|&x| P::part(x, q)).collect())
        .collect()
}

/// The branch's coefficients z_1..z_order at the solution z0 (P::PARTS part arrays), the module's documentation,
/// each order's solve refined `refine` times.
pub fn series<P: Precision<Native>, Pr: ResidualJet<P> + Jvp<P> + ?Sized>(
    pr: &Pr,
    z0: &[&[f64]],
    order: usize,
    refine: usize,
    par: bool,
) -> Result<Branch<P>, String> {
    series_with::<P, Pr, _>(pr, z0, order, refine, par, |z| exact::<P, Pr>(pr, z, par))
}

/// `series` with the Jacobian at z0 factored by `factor`, such as in double-double or f64
/// ([`crate::newton::Lowered`]): each refinement contracts a solve's error by the factors' mismatch, so a few reach
/// P's accuracy. With the exact factors it is `series`, bit for bit.
pub fn series_with<P: Precision<Native>, Pr: ResidualJet<P> + Jvp<P> + ?Sized, F: Factored<P>>(
    pr: &Pr,
    z0: &[&[f64]],
    order: usize,
    refine: usize,
    par: bool,
    factor: impl FnOnce(&[&[f64]]) -> Result<F, String>,
) -> Result<Branch<P>, String> {
    let lu = factor(z0)?;
    // M^-1 v, as the factors' -M^-1 of -v (negation is exact)
    let solve = |v: &[P::Scalar]| -> Result<Vec<P::Scalar>, String> {
        lu.try_solve_neg(&v.iter().map(|&x| -x).collect::<Vec<_>>())
    };
    let mut orders = pr.jet_orders(order)?;
    let mut last: Vec<Vec<f64>> = z0.iter().map(|p| p.to_vec()).collect();
    let mut br = Branch {
        z: vec![(0..pr.len())
            .map(|i| P::from_raw(&z0.iter().map(|p| p[i]).collect::<Vec<_>>()))
            .collect()],
        residual: Vec::new(),
    };
    for _ in 1..=order {
        let f = orders.next(&parts(&last), par)?;
        let b: Vec<P::Scalar> = f.iter().map(|&v| -v).collect();
        let mut x = solve(&b)?;
        let resid = |x: &[P::Scalar]| -> Result<Vec<P::Scalar>, String> {
            let jx = pr.jvp(z0, &parts(&to_parts::<P>(x)), par)?;
            Ok(b.iter().zip(&jx).map(|(&bi, &ji)| bi - ji).collect())
        };
        for _ in 0..refine {
            let r = solve(&resid(&x)?)?;
            x.iter_mut().zip(&r).for_each(|(xi, &ri)| *xi = *xi + ri);
        }
        let nb = max_abs::<P>(&b);
        br.residual.push(if refine > 0 {
            max_abs::<P>(&resid(&x)?) / if nb.is_nan() { nb } else { nb.max(f64::MIN_POSITIVE) }
        } else {
            f64::NAN
        });
        last = to_parts::<P>(&x);
        br.z.push(x);
    }
    Ok(br)
}

impl<P: Precision<Native>> Branch<P> {
    /// the order N of the series
    pub fn order(&self) -> usize {
        self.z.len() - 1
    }

    /// z(a) = sum_k z_k a^k by Horner's rule in P
    pub fn eval(&self, a: f64) -> Vec<P::Scalar> {
        let a = P::from_dd(DD::from_f64(a));
        let mut out = self.z[self.order()].clone();
        for zk in self.z.iter().rev().skip(1) {
            out.iter_mut().zip(zk).for_each(|(o, &c)| *o = *o * a + c);
        }
        out
    }

    /// z(a) as P::PARTS part arrays, the start of the corrector at p0 + a
    pub fn eval_parts(&self, a: f64) -> Vec<Vec<f64>> {
        to_parts::<P>(&self.eval(a))
    }

    /// Cochelin's step (delta |z_1| / |z_N|)^(1 / (N - 1)) (max norms): where the series' last term is delta times its
    /// first. Infinite when z_N is zero (a polynomial or constant branch), NaN when a coefficient holds a NaN; needs
    /// N >= 2.
    pub fn radius(&self, delta: f64) -> f64 {
        let n = self.order();
        assert!(n >= 2, "radius: the series needs order 2 or more, not {n}");
        let last = max_abs::<P>(&self.z[n]);
        if last == 0.0 {
            return f64::INFINITY;
        }
        (delta * max_abs::<P>(&self.z[1]) / last).powf(1.0 / (n - 1) as f64)
    }
}

/// The settings of [`chain`]
#[derive(Clone, Copy, Debug)]
pub struct ChainOpts {
    /// the series' order N at each base
    pub order: usize,
    /// refinements per order of the series' solves
    pub refine: usize,
    /// Cochelin's delta: a step at most `Branch::radius(eps)`, where the series' last term is eps times its first
    pub eps: f64,
    /// the longest step
    pub h_max: f64,
    /// the first step (0: h_max)
    pub a0: f64,
    /// an accepted step's successor is this many times longer, at most h_max
    pub grow: f64,
    /// grow only after a step accepted at its first attempt
    pub hold: bool,
    /// halvings of a refused step at most
    pub halvings: usize,
    /// accepted steps at most
    pub steps: usize,
    /// the parameter past which no step goes
    pub p_max: f64,
    /// the corrector at p0 + a (`newton` on the problem there)
    pub newton: crate::newton::NewtonOpts,
    pub par: bool,
}

impl Default for ChainOpts {
    fn default() -> Self {
        ChainOpts {
            order: 16,
            refine: 1,
            eps: 1e-16,
            h_max: 1.0,
            a0: 0.0,
            grow: 2.0,
            hold: false,
            halvings: 3,
            steps: 20,
            p_max: f64::INFINITY,
            newton: crate::newton::NewtonOpts::default(),
            par: true,
        }
    }
}

/// One attempt of [`chain`]: the base's parameter, the step a and Cochelin's bound on it, the prediction's max |F| at
/// p0 + a, the corrector's report, and whether the step was accepted (the corrector converged)
#[derive(Clone, Debug)]
pub struct ChainStep {
    pub p: f64,
    pub a: f64,
    pub a_max: f64,
    pub f_pred: f64,
    pub newton: crate::newton::NewtonReport,
    pub accepted: bool,
}

/// Chained steps of the series: from the solution z at the parameter p0 (P::PARTS part arrays, replaced by the last
/// accepted solution), the series of order N at each base (`series` on `at(p)`, the problem with its parameter's
/// origin at p), the step a = min(the current step, Cochelin's bound, p_max - p), the prediction z(a), and Newton's
/// method on `at(p + a)` from it. A converged corrector makes its solution the next base and the next step `grow`
/// times longer (with `hold`, only after an acceptance at the first attempt), at most h_max; a failed one halves the
/// step, at most `halvings` times, after which the chain stops. It stops too at `steps` accepted steps or at p_max.
/// Returns every attempt.
pub fn chain<P, Pr, M>(at: M, z: &mut Vec<Vec<f64>>, p0: f64, opts: &ChainOpts) -> Result<Vec<ChainStep>, String>
where
    P: Precision<Native>,
    Pr: ResidualJet<P> + Jvp<P>,
    M: Fn(f64) -> Pr,
{
    // NaN fails each comparison
    let valid = opts.order >= 2 && opts.h_max > 0.0 && opts.grow >= 1.0;
    if !valid {
        return Err(format!(
            "chain: order >= 2, h_max > 0 and grow >= 1, not {}, {}, {}",
            opts.order, opts.h_max, opts.grow
        ));
    }
    let mut out = Vec::new();
    let mut p = p0;
    let mut step = if opts.a0 > 0.0 {
        opts.a0.min(opts.h_max)
    } else {
        opts.h_max
    };
    let mut accepted = 0;
    while accepted < opts.steps && p < opts.p_max {
        let base = at(p);
        let br = series::<P, Pr>(&base, &parts(z), opts.order, opts.refine, opts.par)?;
        let a_max = br.radius(opts.eps);
        let mut a = step.min(a_max).min(opts.p_max - p);
        let positive = a > 0.0;
        if !positive {
            return Err(format!("chain: no step from p {p} (Cochelin's bound {a_max})"));
        }
        let mut done = false;
        for attempt in 0..=opts.halvings {
            let next = at(p + a);
            let mut zt = br.eval_parts(a);
            let f_pred = max_abs::<P>(&next.residual(&parts(&zt), opts.par)?);
            let rep = crate::newton::newton::<P, Pr>(&next, &mut zt, &opts.newton)?;
            let ok = rep.converged;
            out.push(ChainStep {
                p,
                a,
                a_max,
                f_pred,
                newton: rep,
                accepted: ok,
            });
            if ok {
                *z = zt;
                p += a;
                accepted += 1;
                step = if !opts.hold || attempt == 0 {
                    (a * opts.grow).min(opts.h_max)
                } else {
                    a
                };
                done = true;
                break;
            }
            a *= 0.5;
        }
        if !done {
            break;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Double;

    /// For z_k = r^-k the radius is delta^(1 / (N - 1)) r, and z(a) is the geometric sum. The controls: a series one
    /// order shorter has another radius, and its sum at a misses by its last term.
    #[test]
    fn radius_and_eval_of_a_geometric_series() {
        let r = 0.25f64;
        let z: Vec<Vec<DD>> = (0..=6).map(|k| vec![DD::from_f64(r.powi(-k))]).collect();
        let br = Branch::<Double> { z, residual: vec![] };
        let want = 1e-8f64.powf(1.0 / 5.0) * r;
        let radius_ok = |b: &Branch<Double>| (b.radius(1e-8) / want - 1.0).abs() < 1e-14;
        let a = 0.1;
        let sum: f64 = (0..=6).map(|k| (a / r).powi(k)).sum();
        let eval_ok = |b: &Branch<Double>| (b.eval(a)[0].hi / sum - 1.0).abs() < 1e-15;
        assert!(radius_ok(&br) && eval_ok(&br), "{}", br.radius(1e-8));
        let short = Branch::<Double> {
            z: br.z[..6].to_vec(),
            residual: vec![],
        };
        assert!(!radius_ok(&short), "radius control did not fire");
        assert!(!eval_ok(&short), "eval control did not fire");
    }

    /// The series of a constant branch, z_1 = z_2 = 0, has an infinite radius, and one with a NaN in z_1 a NaN
    /// radius. The control: a last coefficient of 1e-300 gives a finite radius.
    #[test]
    fn radius_of_a_constant_branch() {
        let br = |z1: f64, z2: f64| Branch::<Double> {
            z: vec![vec![DD::ONE], vec![DD::from_f64(z1)], vec![DD::from_f64(z2)]],
            residual: vec![],
        };
        assert_eq!(br(0.0, 0.0).radius(1e-8), f64::INFINITY);
        assert!(br(f64::NAN, 1.0).radius(1e-8).is_nan());
        assert!(br(0.0, 1e-300).radius(1e-8).is_finite(), "control did not fire");
    }
}
