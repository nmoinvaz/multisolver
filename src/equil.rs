//! Row and column scalings that equilibrate a matrix: D_r |A| D_c with rows and columns of comparable size, for the
//! factors of a badly scaled Jacobian and for reading its weak directions in units of its own.
//!
//! * [`ruiz`]: Ruiz's iteration in the max norm, each step dividing every row and column by the square root of its
//!   max-norm, until all are 1 within a tolerance (D. Ruiz, "A scaling algorithm to equilibrate both rows and
//!   columns norms in matrices", Rutherford Appleton Laboratory, RAL-TR-2001-034, 2001). It converges linearly, by
//!   about a half per step.
//! * [`curtis_reid`]: Curtis and Reid's least-squares scaling, rho and gamma minimizing the sum over the nonzeros of
//!   (log |a_ij| - rho_i - gamma_j)^2, D_r = e^-rho, D_c = e^-gamma (A. R. Curtis and J. K. Reid, "On the automatic
//!   scaling of matrices for Gaussian elimination", J. Inst. Maths Applics 10 (1972) 118, doi:10.1093/imamat/10.1.118).
//!   Its normal equations are solved by conjugate gradients, as theirs were. Their solution is unique up to
//!   (rho + t, gamma - t). The iterates stay in the range of the singular matrix, so they converge to the solution
//!   with sum rho = sum gamma. With a floor > 0, each nonzero is weighted by its margin in the scaled matrix (its
//!   value over floor times the smaller of its row's and column's max there, clipped to [1e-8, 1]), refitted until
//!   the weights hold. Entries negligible in their row and column then do not steer the fit, and every step is
//!   covariant under row and column scalings of A.
//! * [`curtis_reid_ruiz`]: Curtis and Reid's scaling as the start of Ruiz's iteration.
//!
//! A matrix is given column-major with leading dimension ld, entry (i, j) at `a[j * ld + i]` (a Jacobian's leading
//! part as `crate::Problem::jacobian` returns it). Only |a_ij| is read. [`Scaling::pow2`] rounds the scalings to
//! powers of two, which scale the parts of any precision exactly.

/// The settings of the scalings
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EquilOpts {
    /// scale the columns too (false: the rows alone, D_c = 1)
    pub cols: bool,
    /// Ruiz's stop: every row's and column's max-norm within tol of 1
    pub tol: f64,
    /// Ruiz's steps at most
    pub it_max: usize,
    /// Curtis and Reid's floor (0: every nonzero weighted 1)
    pub floor: f64,
}

impl Default for EquilOpts {
    fn default() -> Self {
        EquilOpts {
            cols: true,
            tol: 1e-12,
            it_max: 200,
            floor: 0.0,
        }
    }
}

/// Row and column scalings D_r, D_c, with Ruiz's iterations and the largest |1 - max-norm| of a row or column of
/// D_r |A| D_c
#[derive(Clone, Debug, PartialEq)]
pub struct Scaling {
    pub dr: Vec<f64>,
    pub dc: Vec<f64>,
    pub iterations: usize,
    pub err: f64,
}

impl Scaling {
    /// the scalings rounded to the nearest powers of two in the logarithm, which scale every part of a multi-part
    /// number exactly
    pub fn pow2(&self) -> Scaling {
        // within the normal range, so that every factor stays finite and nonzero
        let p = |v: &[f64]| {
            v.iter()
                .map(|&x| 2f64.powi((x.log2().round() as i32).clamp(-1022, 1023)))
                .collect()
        };
        Scaling {
            dr: p(&self.dr),
            dc: p(&self.dc),
            iterations: self.iterations,
            err: f64::NAN,
        }
    }
}

/// |a| of the n x m matrix (column-major, leading dimension ld)
fn abs_matrix(n: usize, m: usize, ld: usize, a: &[f64]) -> Vec<f64> {
    assert!(
        ld >= n && a.len() >= ld * m.saturating_sub(1) + n,
        "equil: a matrix too short for its shape"
    );
    let mut b = vec![0.0; n * m];
    for j in 0..m {
        for i in 0..n {
            b[j * n + i] = a[j * ld + i].abs();
        }
    }
    b
}

/// x = f 2^e with f in [0.5, 1), for x finite and positive
fn frexp(x: f64) -> (f64, i32) {
    if x < f64::MIN_POSITIVE {
        let (f, e) = frexp(x * 2f64.powi(54));
        return (f, e - 54);
    }
    let e = ((x.to_bits() >> 52) & 0x7ff) as i32 - 1022;
    (f64::from_bits((x.to_bits() & !(0x7ffu64 << 52)) | (1022u64 << 52)), e)
}

/// f 2^e, by factors of 2^-1000 or 2^1000 then the rest, so that only the result can overflow or underflow
fn ldexp(mut f: f64, mut e: i32) -> f64 {
    while e > 1000 {
        f *= 2f64.powi(1000);
        e -= 1000;
    }
    while e < -1000 {
        f *= 2f64.powi(-1000);
        e += 1000;
    }
    f * 2f64.powi(e)
}

/// a b c for a, b, c >= 0 without an intermediate overflow or underflow: the fractions' product scaled by the
/// exponents' sum, exactly the plain product's bits where that does not leave the range
fn prod3(a: f64, b: f64, c: f64) -> f64 {
    if a == 0.0 || b == 0.0 || c == 0.0 {
        return 0.0;
    }
    let ((fa, ea), (fb, eb), (fc, ec)) = (frexp(a), frexp(b), frexp(c));
    ldexp(fa * fb * fc, ea + eb + ec)
}

/// the row and column max-norms of D_r B D_c (B n x m, column-major with ld n)
fn maxes(n: usize, m: usize, b: &[f64], dr: &[f64], dc: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let (mut r, mut c) = (vec![0.0f64; n], vec![0.0f64; m]);
    for j in 0..m {
        for i in 0..n {
            let v = prod3(dr[i], b[j * n + i], dc[j]);
            r[i] = r[i].max(v);
            c[j] = c[j].max(v);
        }
    }
    (r, c)
}

/// Ruiz's iteration on |B| from the scalings dr, dc: each step divides the rows by the square roots of their
/// max-norms and (with `cols`) the columns by theirs, until every max-norm is within tol of 1 or after it_max steps.
/// A zero row or column keeps its scaling.
fn ruiz_from(n: usize, m: usize, b: &[f64], mut dr: Vec<f64>, mut dc: Vec<f64>, o: &EquilOpts) -> Scaling {
    let (cols, tol, it_max) = (o.cols, o.tol, o.it_max);
    let mut it = 0;
    loop {
        let (r, c) = maxes(n, m, b, &dr, &dc);
        let dev = |v: &[f64]| {
            v.iter()
                .filter(|&&x| x > 0.0)
                .map(|&x| (1.0 - x).abs())
                .fold(0.0, crate::prec::nan_max)
        };
        let err = if cols { dev(&r).max(dev(&c)) } else { dev(&r) };
        if err < tol || it == it_max {
            return Scaling {
                dr,
                dc,
                iterations: it,
                err,
            };
        }
        for (d, &x) in dr.iter_mut().zip(&r) {
            if x > 0.0 {
                *d /= x.sqrt();
            }
        }
        if cols {
            for (d, &x) in dc.iter_mut().zip(&c) {
                if x > 0.0 {
                    *d /= x.sqrt();
                }
            }
        }
        it += 1;
    }
}

/// Ruiz's scaling of the n x m matrix a (column-major, leading dimension ld) in the max norm, from unit scalings:
/// rows and (with `cols`) columns, to within `tol` of 1 or `it_max` steps
pub fn ruiz(n: usize, m: usize, ld: usize, a: &[f64], o: &EquilOpts) -> Scaling {
    let b = abs_matrix(n, m, ld, a);
    ruiz_from(n, m, &b, vec![1.0; n], vec![1.0; m], o)
}

/// the weights' iterations at most, their tolerance, and the smallest weight (which keeps every nonzero in the fit, so
/// that its pattern stays connected and the solution unique)
const FLOOR_IT: usize = 500;
const FLOOR_TOL: f64 = 1e-10;
const FLOOR_WMIN: f64 = 1e-8;

/// Curtis and Reid's weighted least-squares fit of log b_ij = rho_i + gamma_j over the entries with weight w_ij > 0
/// (gamma = 0 without `cols`), by conjugate gradients on its normal equations [diag(nr) W; W^T diag(nc)] [rho;
/// gamma] = [W l 1; W^T l 1]: (e^-rho, e^-gamma)
fn cr_fit(n: usize, m: usize, b: &[f64], w: &[f64], cols: bool) -> (Vec<f64>, Vec<f64>) {
    let lb: Vec<f64> = b
        .iter()
        .zip(w)
        .map(|(&x, &wi)| if wi > 0.0 { wi * x.ln() } else { 0.0 })
        .collect();
    let (mut nr, mut nc) = (vec![0.0; n], vec![0.0; m]);
    let (mut sr, mut sc) = (vec![0.0; n], vec![0.0; m]);
    for j in 0..m {
        for i in 0..n {
            let k = j * n + i;
            nr[i] += w[k];
            nc[j] += w[k];
            sr[i] += lb[k];
            sc[j] += lb[k];
        }
    }
    if !cols {
        let dr = (0..n)
            .map(|i| if nr[i] > 0.0 { (-sr[i] / nr[i]).exp() } else { 1.0 })
            .collect();
        return (dr, vec![1.0; m]);
    }
    // N x for x = [rho; gamma]
    let apply = |x: &[f64]| -> Vec<f64> {
        let mut y = vec![0.0; n + m];
        for i in 0..n {
            y[i] = nr[i] * x[i];
        }
        for j in 0..m {
            y[n + j] = nc[j] * x[n + j];
            for i in 0..n {
                let wk = w[j * n + i];
                if wk > 0.0 {
                    y[i] += wk * x[n + j];
                    y[n + j] += wk * x[i];
                }
            }
        }
        y
    };
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let rhs: Vec<f64> = sr.iter().chain(&sc).copied().collect();
    let mut x = vec![0.0; n + m];
    let mut r = rhs.clone();
    let mut p = r.clone();
    let mut rr = dot(&r, &r);
    let stop = 1e-28 * rr.max(f64::MIN_POSITIVE);
    for _ in 0..4 * (n + m) {
        if rr <= stop {
            break;
        }
        let q = apply(&p);
        let alpha = rr / dot(&p, &q);
        if !alpha.is_finite() {
            break;
        }
        for k in 0..n + m {
            x[k] += alpha * p[k];
            r[k] -= alpha * q[k];
        }
        let rr2 = dot(&r, &r);
        let beta = rr2 / rr;
        for k in 0..n + m {
            p[k] = r[k] + beta * p[k];
        }
        rr = rr2;
    }
    // the gauge (rho - t, gamma + t) with the two midranges equal, so that both factors stay representable where the
    // products e^-(rho_i + gamma_j) do
    let mid = |v: &[f64]| {
        let (lo, hi) = v
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| (a.min(x), b.max(x)));
        0.5 * (lo + hi)
    };
    let t = 0.5 * (mid(&x[..n]) - mid(&x[n..]));
    (
        x[..n].iter().map(|&v| (t - v).exp()).collect(),
        x[n..].iter().map(|&v| (-v - t).exp()).collect(),
    )
}

/// each nonzero's weight: its margin b'_ij / (floor min(row max, column max)) in B' = D_r B D_c, clipped to
/// [FLOOR_WMIN, 1]
fn floor_weights(n: usize, m: usize, b: &[f64], dr: &[f64], dc: &[f64], floor: f64) -> Vec<f64> {
    let (r, c) = maxes(n, m, b, dr, dc);
    let mut w = vec![0.0; n * m];
    for j in 0..m {
        for i in 0..n {
            let k = j * n + i;
            if b[k] > 0.0 {
                let v = prod3(dr[i], b[k], dc[j]) / (floor * r[i].min(c[j]));
                w[k] = v.clamp(FLOOR_WMIN, 1.0);
            }
        }
    }
    w
}

/// Curtis and Reid's scaling of the n x m matrix a (column-major, leading dimension ld), the module's documentation:
/// rows and (with `cols`) columns, every nonzero weighted 1 with `floor` 0, or by its margin with `floor` > 0. The
/// returned `iterations` count the floor's refits and `err` is the scaled matrix's largest |1 - max-norm|.
pub fn curtis_reid(n: usize, m: usize, ld: usize, a: &[f64], o: &EquilOpts) -> Scaling {
    let (cols, floor) = (o.cols, o.floor);
    let b = abs_matrix(n, m, ld, a);
    let mut w: Vec<f64> = b.iter().map(|&x| if x > 0.0 { 1.0 } else { 0.0 }).collect();
    let (mut dr, mut dc) = cr_fit(n, m, &b, &w, cols);
    let mut refits = 0;
    if floor > 0.0 {
        for _ in 0..FLOOR_IT {
            let wn = floor_weights(n, m, &b, &dr, &dc, floor);
            let moved = wn.iter().zip(&w).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
            w = wn;
            (dr, dc) = cr_fit(n, m, &b, &w, cols);
            refits += 1;
            if moved <= FLOOR_TOL {
                break;
            }
        }
    }
    let at_once = EquilOpts {
        tol: f64::INFINITY,
        it_max: 0,
        ..*o
    };
    let s = ruiz_from(n, m, &b, dr, dc, &at_once);
    Scaling {
        iterations: refits,
        ..s
    }
}

/// Curtis and Reid's scaling as the start of Ruiz's iteration
pub fn curtis_reid_ruiz(n: usize, m: usize, ld: usize, a: &[f64], o: &EquilOpts) -> Scaling {
    let s = curtis_reid(n, m, ld, a, o);
    let b = abs_matrix(n, m, ld, a);
    ruiz_from(n, m, &b, s.dr, s.dc, o)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pass: prod3 gives the plain product's bits for 2000 triples of the normal range whose product stays in it, and
    /// a finite 1e266 for 1e183 1e300 1e-217 (relative 1e-15), with the row and column maxima of that one entry
    /// finite. The control: the plain product of the same triple, left to right, is infinite.
    #[test]
    fn products_stay_in_range() {
        let mut x = 0.123f64;
        let mut next = || {
            x = (x * 9301.0 + 0.49297).fract();
            x
        };
        for _ in 0..2000 {
            let (a, b, c) = (
                10f64.powf(200.0 * next() - 100.0),
                10f64.powf(200.0 * next() - 100.0),
                10f64.powf(200.0 * next() - 100.0),
            );
            assert_eq!(prod3(a, b, c).to_bits(), (a * b * c).to_bits(), "{a:e} {b:e} {c:e}");
        }
        let v = prod3(1e183, 1e300, 1e-217);
        assert!((v / 1e266 - 1.0).abs() <= 1e-15, "{v:e}");
        let (r, c) = maxes(1, 1, &[1e300], &[1e183], &[1e-217]);
        assert!(r[0].is_finite() && c[0].is_finite());
        assert!((1e183 * 1e300 * 1e-217f64).is_infinite(), "control did not fire");
    }
}
