//! Deflation of known roots (Farrell, Birkisson and Funke, SIAM J. Sci. Comput. 37 (2015) A2026): a further root of F
//! from the same start, as a root of G(z) = M(z) F(z) with M(z) = prod_i (|z - r_i|^-p + sigma) over the known roots
//! r_i. Near a simple root r_i, where F is of order |z - r_i|, G grows as |z - r_i|^(1-p), so for p >= 2 Newton's
//! method on G does not converge to it; far from all of them M tends to sigma^k and G to a multiple of F. Since M > 0,
//! the roots of G are those of F other than the r_i.
//!
//! The Jacobian M J + F (grad M)^T = M (J + F w^T), with w = grad log M = -sum_i p d_i / (|d_i|^2 (1 + sigma
//! |d_i|^p)) and d_i = z - r_i, is a rank-one update of M J. By Sherman and Morrison the deflated Newton step is the
//! undeflated one, delta, scaled by 1 / (1 - w . delta): deflation changes the steps' lengths, not their directions.
//! [`Without`] is the problem G, its residual and Jacobian exact in P, so that every solver runs on it unchanged, and
//! its Jacobian-vector product M (J v + F (w . v)) where the problem has one ([`crate::Jvp`], for
//! [`crate::gcr::newton_refined`]); [`roots_from`] collects roots one search at a time.
//!
//! Since the directions are Newton's for F, the damping decides whether a search leaves the first root's basin. The
//! correction rule follows the continuous Newton path, along which G stays parallel to G(z_0) and which deflation does
//! not move, so its searches often find no further root; full steps (the residual rule with `max_halvings` 0, the
//! undamped iteration of the paper's examples) leave the path and reach the other roots.
//!
//! M overflows within about |z - r_i| < 1e-154 (for p = 2) of a known root, where f64's exponent range ends. There, and
//! at a known root itself, the residual is +inf in every row, never NaN: a trial step there fails the correction
//! rule's test and is damped, and a start there ends the Newton loop unconverged. With sigma = 0 the same holds where
//! M, which then vanishes at infinity, falls below f64's range. Far from the known roots, where |z - r_i|^p or
//! |z - r_i|^2 overflows, M and w stay finite.
use crate::gcr::recip;
use crate::newton::NewtonReport;
use crate::{Jvp, Problem};
use multidouble::dd::DD;
use multidouble::precision::Precision;
use multidouble::simd::Native;

/// The deflation operator's settings: M(z) = prod_i (|z - r_i|^-power + shift)
#[derive(Clone, Copy, Debug)]
pub struct DeflationOpts {
    /// p, even and at least 2, so that |z - r_i|^-p needs no square root; the paper's default 2
    pub power: u32,
    /// sigma >= 0: with sigma > 0, M tends to sigma^k away from the known roots and G keeps F's behavior there; with
    /// sigma = 0, G vanishes at infinity, which can draw the iterates away (the paper's default 1)
    pub shift: f64,
}

impl Default for DeflationOpts {
    fn default() -> Self {
        DeflationOpts { power: 2, shift: 1.0 }
    }
}

/// x is finite, by its leading part
fn finite<P: Precision<Native>>(x: P::Scalar) -> bool {
    P::to_dd(x).hi.is_finite()
}

/// the scalars of a state's part arrays
fn scalars<P: Precision<Native>>(z: &[&[f64]]) -> Vec<P::Scalar> {
    let mut buf = vec![0.0; P::PARTS];
    (0..z[0].len())
        .map(|i| {
            for (q, b) in buf.iter_mut().enumerate() {
                *b = z[q][i];
            }
            P::from_raw(&buf)
        })
        .collect()
}

/// The problem `pr` with its known roots deflated: G = M F ([`crate::deflation`]), in P. With no roots it is `pr`, bit
/// for bit.
pub struct Without<'a, P: Precision<Native>, Pr: Problem<P> + ?Sized> {
    pr: &'a Pr,
    roots: Vec<Vec<P::Scalar>>,
    opts: DeflationOpts,
}

impl<'a, P: Precision<Native>, Pr: Problem<P> + ?Sized> Without<'a, P, Pr> {
    /// `pr` with the roots `roots` (each P::PARTS part arrays of length `pr.len()`) deflated by `opts`
    pub fn new(pr: &'a Pr, roots: &[Vec<Vec<f64>>], opts: DeflationOpts) -> Result<Self, String> {
        if opts.power < 2 || !opts.power.is_multiple_of(2) {
            return Err(format!(
                "Without: power must be even and at least 2, not {}",
                opts.power
            ));
        }
        if !(opts.shift >= 0.0 && opts.shift.is_finite()) {
            return Err(format!(
                "Without: shift must be finite and at least 0, not {}",
                opts.shift
            ));
        }
        let n = pr.len();
        let mut rs = Vec::with_capacity(roots.len());
        for r in roots {
            if r.len() != P::PARTS || r.iter().any(|q| q.len() != n) {
                return Err(format!(
                    "Without: a root of {} parts, not {} parts of length {n}",
                    r.len(),
                    P::PARTS
                ));
            }
            let refs: Vec<&[f64]> = r.iter().map(|q| q.as_slice()).collect();
            rs.push(scalars::<P>(&refs));
        }
        Ok(Without { pr, roots: rs, opts })
    }

    /// the known roots' count
    pub fn known(&self) -> usize {
        self.roots.len()
    }

    /// M(z) and w = grad log M(z), or None where M is not finite or, with `shift` 0, below f64's range: at a known
    /// root or within f64's range of one, and with `shift` 0 also far from one
    pub fn operator(&self, z: &[&[f64]]) -> Option<(P::Scalar, Vec<P::Scalar>)> {
        let zs = scalars::<P>(z);
        let sigma = P::from_dd(DD::from_f64(self.opts.shift));
        let p = P::from_dd(DD::from_f64(self.opts.power as f64));
        let mut m = P::ONE;
        let mut w = vec![P::ZERO; zs.len()];
        for r in &self.roots {
            let d: Vec<P::Scalar> = zs.iter().zip(r).map(|(&a, &b)| a - b).collect();
            // q = |d|^2, s = 1 / q, a = |d|^-p, m_i = a + sigma, and grad log m_i = -c d with c = p s a / m_i
            let q = P::dot(d.iter().copied().zip(d.iter().copied()));
            if !finite::<P>(q) {
                // |d| past about 1e154: a underflows, so m_i = sigma, and c |d| < p / (sigma |d|^(p+1)) is below
                // f64's range; with sigma = 0, m_i = a is too
                if self.opts.shift == 0.0 {
                    return None;
                }
                m = m * sigma;
                continue;
            }
            let s = recip::<P>(q);
            let (mut a, mut dp) = (s, q);
            for _ in 1..self.opts.power / 2 {
                (a, dp) = (a * s, dp * q);
            }
            let mi = a + sigma;
            // a / m_i as 1 / (1 + sigma |d|^p) while |d|^p is finite, else as a / m_i with a = |d|^-p < 1
            let c = if self.opts.shift == 0.0 {
                p * s
            } else if finite::<P>(dp) {
                p * s * recip::<P>(P::ONE + sigma * dp)
            } else {
                p * s * a * recip::<P>(mi)
            };
            if !(finite::<P>(s) && finite::<P>(mi) && P::to_dd(mi).hi >= f64::MIN_POSITIVE && finite::<P>(c)) {
                return None;
            }
            m = m * mi;
            w.iter_mut().zip(&d).for_each(|(x, &di)| *x = *x - c * di);
        }
        finite::<P>(m).then_some((m, w))
    }
}

impl<P: Precision<Native>, Pr: Problem<P> + ?Sized> Problem<P> for Without<'_, P, Pr> {
    fn len(&self) -> usize {
        self.pr.len()
    }
    /// M F, or +inf in every row where M is not finite
    fn residual(&self, z: &[&[f64]], par: bool) -> Result<Vec<P::Scalar>, String> {
        let f = self.pr.residual(z, par)?;
        if self.roots.is_empty() {
            return Ok(f);
        }
        Ok(match self.operator(z) {
            Some((m, _)) => f.iter().map(|&v| m * v).collect(),
            None => vec![P::from_dd(DD::from_f64(f64::INFINITY)); f.len()],
        })
    }
    /// M (J + F w^T), entry by entry in P; an error where M is not finite
    fn jacobian(&self, z: &[&[f64]], par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
        let (mut c, ld) = self.pr.jacobian(z, par)?;
        if self.roots.is_empty() {
            return Ok((c, ld));
        }
        let Some((m, w)) = self.operator(z) else {
            return Err("Without: no Jacobian at a known root, where M is not finite".into());
        };
        let f = self.pr.residual(z, par)?;
        let n = self.pr.len();
        if c.len() != P::PARTS || f.len() != n {
            return Err(format!(
                "Without: a Jacobian of {} parts and a residual of {} rows, not {} and {n}",
                c.len(),
                f.len(),
                P::PARTS
            ));
        }
        let mut buf = vec![0.0; P::PARTS];
        for (j, &wj) in w.iter().enumerate() {
            for (i, &fi) in f.iter().enumerate() {
                for (q, b) in buf.iter_mut().enumerate() {
                    *b = c[q][j * ld + i];
                }
                let v = m * (P::from_raw(&buf) + fi * wj);
                for (q, cq) in c.iter_mut().enumerate() {
                    cq[j * ld + i] = P::part(v, q);
                }
            }
        }
        Ok((c, ld))
    }
}

impl<P: Precision<Native>, Pr: Jvp<P> + ?Sized> Jvp<P> for Without<'_, P, Pr> {
    /// M (J v + F (w . v)), from the problem's J v and F; an error where M is not finite
    fn jvp(&self, z: &[&[f64]], v: &[&[f64]], par: bool) -> Result<Vec<P::Scalar>, String> {
        let jv = self.pr.jvp(z, v, par)?;
        if self.roots.is_empty() {
            return Ok(jv);
        }
        let Some((m, w)) = self.operator(z) else {
            return Err("Without: no J v at a known root, where M is not finite".into());
        };
        let f = self.pr.residual(z, par)?;
        if jv.len() != f.len() {
            return Err(format!("Without: J v of {} rows and F of {}", jv.len(), f.len()));
        }
        let wv = P::dot(w.iter().copied().zip(scalars::<P>(v)));
        Ok(jv.iter().zip(&f).map(|(&a, &fi)| m * (a + fi * wv)).collect())
    }
}

/// The roots [`roots_from`] found, the reports of their searches, and why the searches ended before `count`: the
/// search that did not converge, or its error; None when they reached `count`
#[derive(Clone, Debug)]
pub struct Found {
    pub roots: Vec<Vec<Vec<f64>>>,
    pub reports: Vec<NewtonReport>,
    pub ended: Option<String>,
}

/// Up to `count` roots of `pr` from the start z0 (P::PARTS part arrays), one search at a time: search k runs `solve`
/// (such as `|g, z| newton(g, z, &opts)` with full steps, the module's documentation) from z0 on [`Without`] the k
/// roots before it. A search ends the collection
/// when it does not converge or returns an error. Convergence is the solver's on G = M F, which with `shift` >= 1
/// bounds F too (M >= 1). The error is that of invalid `opts`.
pub fn roots_from<P: Precision<Native>, Pr: Problem<P> + ?Sized>(
    pr: &Pr,
    z0: &[Vec<f64>],
    count: usize,
    opts: &DeflationOpts,
    mut solve: impl FnMut(&Without<'_, P, Pr>, &mut Vec<Vec<f64>>) -> Result<NewtonReport, String>,
) -> Result<Found, String> {
    let mut found = Found {
        roots: Vec::new(),
        reports: Vec::new(),
        ended: None,
    };
    while found.roots.len() < count {
        let g = Without::new(pr, &found.roots, *opts)?;
        let mut z = z0.to_vec();
        match solve(&g, &mut z) {
            Ok(rep) if rep.converged => {
                found.roots.push(z);
                found.reports.push(rep);
            }
            Ok(rep) => {
                found.ended = Some(format!(
                    "search {} did not converge: {} steps, max |G| {:e} at the end",
                    found.roots.len() + 1,
                    rep.scale.len(),
                    rep.fnorm.last().copied().unwrap_or(f64::NAN)
                ));
                break;
            }
            Err(e) => {
                found.ended = Some(format!("search {}: {e}", found.roots.len() + 1));
                break;
            }
        }
    }
    Ok(found)
}
