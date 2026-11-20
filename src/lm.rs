//! The Levenberg-Marquardt method on F(z) = 0, for any [`Problem`]: each step solves (G + mu D) dz = -J^T F with
//! G = J^T J and Marquardt's scaling D = diag(G), and mu adapts to how well the step's predicted decrease of
//! |F|^2 / 2 matched the real one (the gain ratio; Madsen, Nielsen and Tingleff, Methods for Non-Linear Least
//! Squares Problems, DTU 2004, section 3.2, after Nielsen 1999). Small mu gives Newton's step, large mu a short step
//! along the scaled gradient, so the method keeps going where the Jacobian is singular or Newton's step overshoots.
//! Near the root it hands over to Newton's method, whose rate is quadratic, with LM's tolerance. If Newton fails there
//! (a singular Jacobian, or no convergence), LM goes on from its last accepted state and does not hand over again. The
//! gain ratio is formed in f64 from |F|^2, which underflows below about 1e-150, so a run asked for a smaller tolerance
//! needs the handover.
//!
//! D is floored at 2^-104 times its largest entry, since a zero column of J would leave G + mu D singular for every
//! mu. A factorization that fails (a singular or non-finite pivot) counts as a refused step. An error of the problem's
//! residual at a trial state is returned, as Newton's method does.
//!
//! The steps are solved in double-double whatever P: G and J^T F are formed once per step from J and F rounded to
//! double-double (their first two parts), on the lanes of double-double (8 rows of a column per lane vector, the dot
//! accumulator of the lane type), only G's upper triangle and with its columns in parallel, and each mu tried costs
//! one factorization of G + mu D in double-double. The steps only have to lower |F| until the handover; the residual,
//! the trial states and the handover stay in P, so that LM alone still converges to P's accuracy, linearly by about
//! cond(J)^2 2^-104 per step near the root. At n = 256 the Gram matrix and the LU in double-double cost about 1/8 and
//! 1/20 of quad-double's, 1/90 and 1/120 of octo-double's. For P = `Double` nothing is rounded. Past cond(J) of about
//! 1e15 at the root the steps lose their accuracy there, and the handover's Newton, in P, finishes.
use crate::newton::{exact, newton_with, norm2, Factored, NewtonOpts, NewtonReport};
use crate::prec::{max_abs, parts};
use crate::Problem;
use multidouble::dd::DD;
use multidouble::dual::Lane;
use multidouble::field::Fl;
use multidouble::linalg::DDLu;
use multidouble::precision::{Double, Precision};
use multidouble::simd::Native;
use rayon::prelude::*;

/// The settings of [`levenberg_marquardt`]
#[derive(Clone, Copy, Debug)]
pub struct LmOpts {
    /// stop when max |F| is at most this
    pub tol: f64,
    /// steps at most
    pub max_iter: usize,
    /// the first mu, relative to the largest diagonal entry of G (tau in Madsen, Nielsen and Tingleff)
    pub tau: f64,
    /// hand over to Newton's method once max |F| is below this (0: never)
    pub newton_after: f64,
    /// the settings of that Newton run, except its tolerance, which is `tol`
    pub newton: NewtonOpts,
    pub par: bool,
}

impl Default for LmOpts {
    fn default() -> Self {
        LmOpts {
            tol: 1e-28,
            max_iter: 50,
            tau: 1e-3,
            newton_after: 1e-12,
            newton: NewtonOpts::default(),
            par: true,
        }
    }
}

/// What [`levenberg_marquardt`] did: max |F| before each step and at the end, the mu of each step taken, the
/// factorizations formed, whether it converged, and the report of the Newton run it handed over to, if that run
/// returned one
#[derive(Clone, Debug, Default)]
pub struct LmReport {
    pub fnorm: Vec<f64>,
    pub mu: Vec<f64>,
    pub factorizations: usize,
    pub converged: bool,
    pub newton: Option<NewtonReport>,
}

/// nv lane vectors of 8 rows from part q of row i at `col(q, i)`, rows past n zero
fn lanes<P: Precision<Native>>(n: usize, nv: usize, col: impl Fn(usize, usize) -> f64) -> Vec<P::V> {
    let mut buf = vec![[0.0f64; 8]; P::PARTS];
    (0..nv)
        .map(|v| {
            for (q, bq) in buf.iter_mut().enumerate() {
                for (k, x) in bq.iter_mut().enumerate() {
                    let i = 8 * v + k;
                    *x = if i < n { col(q, i) } else { 0.0 };
                }
            }
            let sl: [&[f64]; 8] = std::array::from_fn(|q| if q < P::PARTS { &buf[q][..] } else { &[][..] });
            P::load_field::<1>(&sl[..P::PARTS]).0[0]
        })
        .collect()
}

/// The Gram matrix G = J^T J and g = J^T b in P, for J as `Problem::jacobian` gives it (part q of entry (i, j) at
/// `c[q][j * ld + i]`, n x n): G as part arrays with the same layout, filled completely, and g. Rows n..ld of c are
/// taken as zero whatever they hold.
pub fn gram<P: Precision<Native>>(
    n: usize,
    ld: usize,
    c: &[Vec<f64>],
    b: &[P::Scalar],
    par: bool,
) -> (Vec<Vec<f64>>, Vec<P::Scalar>) {
    let nv = ld / 8;
    let cols: Vec<Vec<P::V>> = (0..n).map(|j| lanes::<P>(n, nv, |q, i| c[q][j * ld + i])).collect();
    let bv = lanes::<P>(n, nv, |q, i| P::part(b[i], q));
    let dot = |x: &[P::V], y: &[P::V]| -> P::Scalar {
        let mut a = <P::V as Lane>::acc0();
        for (xv, yv) in x.iter().zip(y) {
            <P::V as Lane>::acc_mut(&mut a, xv, yv);
        }
        let s = Fl::<P::V, 1>([<P::V as Lane>::fin(a)]);
        (0..8).fold(P::ZERO, |t, k| t + P::field_get(&s, k))
    };
    // column j of G's upper triangle, rows 0..=j
    let upper = |j: usize| -> Vec<P::Scalar> { (0..=j).map(|i| dot(&cols[i], &cols[j])).collect() };
    let tri: Vec<Vec<P::Scalar>> = if par {
        (0..n).into_par_iter().map(upper).collect()
    } else {
        (0..n).map(upper).collect()
    };
    let mut g = vec![vec![0.0; ld * n]; P::PARTS];
    for (j, col) in tri.iter().enumerate() {
        for (i, &v) in col.iter().enumerate() {
            for (q, gq) in g.iter_mut().enumerate() {
                gq[j * ld + i] = P::part(v, q);
                gq[i * ld + j] = P::part(v, q);
            }
        }
    }
    let jb = cols.iter().map(|col| dot(col, &bv)).collect();
    (g, jb)
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

/// z + dz as part arrays of P
fn add<P: Precision<Native>>(z: &[&[f64]], dz: &[P::Scalar]) -> Vec<Vec<f64>> {
    let s: Vec<P::Scalar> = scalars::<P>(z).iter().zip(dz).map(|(&a, &b)| a + b).collect();
    (0..P::PARTS)
        .map(|q| s.iter().map(|&v| P::part(v, q)).collect())
        .collect()
}

/// The Levenberg-Marquardt method from z (P::PARTS part arrays, updated in place), the module's documentation: a
/// step with gain ratio rho > 0 is taken and mu multiplied by max(1/3, 1 - (2 rho - 1)^3), and a step that does not
/// lower |F| multiplies mu by nu, which doubles on each refusal in a row.
pub fn levenberg_marquardt<P: Precision<Native>, Pr: Problem<P> + ?Sized>(
    pr: &Pr,
    z: &mut Vec<Vec<f64>>,
    opts: &LmOpts,
) -> Result<LmReport, String> {
    levenberg_marquardt_with::<P, Pr, _>(pr, z, opts, |z| exact::<P, Pr>(pr, z, opts.newton.par))
}

/// `levenberg_marquardt` with the Newton run it hands over to factoring its Newton matrix by `factor`
/// ([`newton_with`]), such as the Jacobian rounded to double-double ([`crate::newton::Lowered`]). With the exact
/// factors it is `levenberg_marquardt`, bit for bit.
pub fn levenberg_marquardt_with<P: Precision<Native>, Pr: Problem<P> + ?Sized, F: Factored<P>>(
    pr: &Pr,
    z: &mut Vec<Vec<f64>>,
    opts: &LmOpts,
    factor: impl Fn(&[&[f64]]) -> Result<F, String>,
) -> Result<LmReport, String> {
    let n = pr.len();
    let mut rep = LmReport::default();
    let mut f = pr.residual(&parts(z), opts.par)?;
    let mut mu = -1.0f64;
    let mut nu = 2.0f64;
    let mut handover = opts.newton_after > 0.0;
    for _ in 0..opts.max_iter {
        let fmax = max_abs::<P>(&f);
        rep.fnorm.push(fmax);
        if !fmax.is_finite() {
            return Ok(rep);
        }
        if fmax <= opts.tol {
            rep.converged = true;
            return Ok(rep);
        }
        if handover && fmax < opts.newton_after {
            handover = false;
            // Newton on a copy, to LM's tolerance; its last fnorm is max |F| at its final state
            let mut zn = z.clone();
            let nopts = NewtonOpts {
                tol: opts.tol,
                ..opts.newton
            };
            if let Ok(nr) = newton_with::<P, Pr, F>(pr, &mut zn, &nopts, &factor) {
                let last = nr.fnorm.last().copied();
                let done = nr.converged && last.is_some_and(|f| f <= opts.tol);
                rep.newton = Some(nr);
                if let (true, Some(f)) = (done, last) {
                    *z = zn;
                    rep.fnorm.push(f);
                    rep.converged = true;
                    return Ok(rep);
                }
            }
        }
        let (c, ld) = pr.jacobian(&parts(z), opts.par)?;
        if c.len() != P::PARTS {
            return Err(format!("lm: a Jacobian of {} parts, not {}", c.len(), P::PARTS));
        }
        // G and J^T F in double-double, from J's first two parts and F rounded
        let nf: Vec<DD> = f.iter().map(|&v| -P::to_dd(v)).collect();
        let (g, jf) = gram::<Double>(n, ld, &c[..2], &nf, opts.par);
        let mut diag: Vec<DD> = (0..n)
            .map(|i| DD {
                hi: g[0][i * ld + i],
                lo: g[1][i * ld + i],
            })
            .collect();
        let floor = diag.iter().map(|d| d.hi).fold(0.0, f64::max) * 2f64.powi(-104);
        for d in diag.iter_mut() {
            if d.hi < floor {
                *d = DD::from_f64(floor);
            }
        }
        if mu < 0.0 {
            mu = opts.tau;
        }
        let f2 = norm2::<P>(&f).powi(2);
        loop {
            // (G + mu D) dz = -J^T F
            let [mut hi, mut lo]: [Vec<f64>; 2] = g.clone().try_into().expect("gram: 2 parts");
            let m = DD::from_f64(mu);
            for (i, &d) in diag.iter().enumerate() {
                let k = i * ld + i;
                let v = DD { hi: hi[k], lo: lo[k] } + m * d;
                (hi[k], lo[k]) = (v.hi, v.lo);
            }
            rep.factorizations += 1;
            let Ok(lu) = DDLu::factor(n, ld, hi, lo, opts.par) else {
                mu *= nu;
                nu *= 2.0;
                if !mu.is_finite() {
                    rep.fnorm.push(fmax);
                    return Ok(rep);
                }
                continue;
            };
            let mut dz = jf.clone();
            lu.solve(&mut dz);
            let dzp: Vec<P::Scalar> = dz.iter().map(|&h| P::from_dd(h)).collect();
            let zt = add::<P>(&parts(z), &dzp);
            let ft = pr.residual(&parts(&zt), opts.par)?;
            // predicted decrease of |F|^2 / 2: dz . (mu D dz - J^T F) / 2, J^T F = -jf
            let pred: f64 = dz
                .iter()
                .zip(&diag)
                .zip(&jf)
                .map(|((&h, &d), &j)| (h * (DD::from_f64(mu) * d * h + j)).hi)
                .sum::<f64>()
                / 2.0;
            let ft2 = norm2::<P>(&ft).powi(2);
            let rho = (f2 - ft2) / 2.0 / pred;
            if ft2.is_finite() && rho > 0.0 {
                *z = zt;
                f = ft;
                rep.mu.push(mu);
                mu *= (1.0f64 / 3.0).max(1.0 - (2.0 * rho - 1.0).powi(3));
                nu = 2.0;
                break;
            }
            mu *= nu;
            nu *= 2.0;
            if !mu.is_finite() {
                rep.fnorm.push(fmax);
                return Ok(rep);
            }
        }
    }
    let fmax = max_abs::<P>(&f);
    rep.fnorm.push(fmax);
    rep.converged = fmax <= opts.tol;
    Ok(rep)
}
