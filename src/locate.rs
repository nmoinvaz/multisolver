//! The nearest singularity of a function from its Taylor coefficients c_0 ... c_K about a point: where it is, its
//! exponent alpha, whether it is real or one of a conjugate pair, and an error bar. It generalizes the estimators of
//! [`crate::singularity`] to f64, double-, quad- and octo-double (the [`Real`] types) and puts a bar on each result.
//!
//! Near its nearest singularity s the function behaves as A (1 - u / s)^alpha, u the distance from the expansion point,
//! so that c_k = A binom(alpha, k) (-1 / s)^k ~ s^-k k^(-alpha - 1). Two models of the coefficients are fitted.
//!
//! * [`Model::Ratio`], the ratio method of Domb and Sykes (Proc. R. Soc. Lond. A 240 (1957) 214): the ratios
//!   r_k = c_k / c_(k-1) = (1 / s)(1 - (1 + alpha) / k) + O(1 / k^2), fitted as a polynomial in 1 / k. The intercept is
//!   1 / s and the slope over the intercept is -(1 + alpha). It needs a real s: for a conjugate pair the ratios
//!   oscillate.
//! * [`Model::ThreeTerm`], a recurrence of three terms, fitted over the same orders:
//!   (1 + 1 / k) c_(k+1) = (pi_0 + pi_1 / k + ...) c_k - (theta_0 + theta_1 / k + ...) c_(k-1). Two singularities of
//!   one exponent satisfy it exactly with pi_0 = 1/s_1 + 1/s_2, theta_0 = 1/(s_1 s_2), pi_1 = -alpha pi_0 and
//!   theta_1 = -(1 + 2 alpha) theta_0, the case of a conjugate pair s, conj(s) and, with s_1 and s_2 real, of two real
//!   singularities. Where the exponents differ, the first-order coefficients still give each exponent
//!   (P_1 - Q_1 z_j = g_j (z_j - z_other) for z_j = 1 / s_j, with P = pi_0, pi_1 - pi_0 and Q = theta_0, theta_1 - theta_0 the
//!   coefficients of c_(k+1) = P_k c_k - Q_k c_(k-1)). The roots of z^2 - P_0 z + Q_0 are 1 / s: complex, a pair
//!   (the recurrence of [`crate::singularity::recurrence`] with the terms in 1 / k, which carry the exponent); real, the
//!   nearest of two real singularities and the farther one behind it. A single real singularity makes the fit rank
//!   deficient, and the ratio model serves.
//!
//! Each model is fitted over a grid of cells (first order of the window, degree in 1 / k) by least squares in the
//! precision of the coefficients (Householder QR). The estimate is the cell whose eight neighbors in the grid agree best
//! with it, and its error bar is `bar_factor` times the largest disagreement. **The bar is the range of the estimator
//! over its fit choices, an estimator range, not a bound**: the fit of any one cell has a truncation error of order
//! (1 / k)^(degree + 1) that the neighbors share in part, and the farther singularities act on the windows alike. It
//! is floored at a few units of the rounding of the arithmetic. Of the two models the one with the smaller
//! disagreement wins, and an estimate whose bar exceeds `max_rel_bar` of |s| is [`Kind::Unresolved`].
//!
//! The coefficients are taken in a variable scaled so that the radius is of order one; the locator needs c_(k-1) != 0
//! (a series in u^2 for an even function), and the orders K of the series should be at least 12.
//!
//! ```
//! use multisolver::locate::{locate, Kind, Opts};
//!
//! // 3 + 1.7 / (1 - u / 2): a pole (alpha = -1) at 2
//! let mut c = vec![3.0 + 1.7];
//! c.extend((1..=30).map(|k| 1.7 / 2f64.powi(k)));
//! let loc = locate(&c, &Opts::default()).unwrap();
//! assert_eq!(loc.kind, Kind::Real);
//! assert!((loc.nearest.re - 2.0).abs() <= loc.nearest.bar);
//! assert!((loc.nearest.alpha + 1.0).abs() <= loc.nearest.alpha_bar);
//! ```
use multidouble::dd::DD;
use multidouble::od::OD;
use multidouble::qd::QD;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// The arithmetic of the locator: f64 and multidouble's scalars ([`DD`], [`QD`], [`OD`], the `Scalar` of `Double`,
/// `Quad` and `Octo`)
pub trait Real:
    Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Neg<Output = Self>
{
    /// 2^(-52 n) for n parts, the scale of a rounding in this arithmetic
    const UNIT: f64;
    /// exactly
    fn from_f64(x: f64) -> Self;
    /// rounded to f64
    fn to_f64(self) -> f64;
    /// for a positive argument
    fn sqrt(self) -> Self;
}

impl Real for f64 {
    const UNIT: f64 = 2.220446049250313e-16;
    fn from_f64(x: f64) -> Self {
        x
    }
    fn to_f64(self) -> f64 {
        self
    }
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
}

macro_rules! real_multidouble {
    ($t:ty, $unit:expr) => {
        impl Real for $t {
            const UNIT: f64 = $unit;
            fn from_f64(x: f64) -> Self {
                <$t>::from_f64(x)
            }
            fn to_f64(self) -> f64 {
                <$t>::to_f64(self)
            }
            fn sqrt(self) -> Self {
                <$t>::sqrt(self)
            }
        }
    };
}

real_multidouble!(DD, 4.930380657631324e-32);
real_multidouble!(QD, 2.4308653429145085e-63);
real_multidouble!(OD, 5.909106315382871e-126);

/// The fitted model of the coefficients
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Model {
    /// the ratios c_k / c_(k-1) as a polynomial in 1 / k: one real singularity
    Ratio,
    /// the recurrence of three terms: a conjugate pair, or two real singularities
    ThreeTerm,
}

/// What the located singularity is
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// a real singularity (on either side of the expansion point)
    Real,
    /// one of a complex conjugate pair, reported with Im s > 0
    Pair,
    /// no estimate whose bar is below `max_rel_bar` of |s|: no singularity within reach of the orders, or none that
    /// the models fit
    Unresolved,
}

/// The settings of [`locate`]
#[derive(Clone, Debug)]
pub struct Opts {
    /// the first order of the fit window, as fractions of K (at least order 2). The late windows (0.7 K and 0.8 K) start
    /// past the orders where a farther singularity or a dressing masks the nearest one, and have few rows: low degrees
    pub starts: Vec<f64>,
    /// the degrees in 1 / k of the ratio fit
    pub ratio_degrees: Vec<usize>,
    /// the degrees in 1 / k of the three-term fit (at least 2: the pair is exact there)
    pub three_term_degrees: Vec<usize>,
    /// the bar is this many times the largest disagreement with a neighbor
    pub bar_factor: f64,
    /// the largest bar, as a fraction of |s|, of a resolved singularity
    pub max_rel_bar: f64,
}

impl Default for Opts {
    fn default() -> Self {
        Opts {
            starts: vec![0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            ratio_degrees: (1..=9).collect(),
            three_term_degrees: (2..=9).collect(),
            bar_factor: 2.0,
            max_rel_bar: 0.05,
        }
    }
}

/// One singularity: s = re + i im, the exponent alpha of (1 - u / s)^alpha, and the bars of the estimator range
#[derive(Clone, Copy, Debug)]
pub struct Singularity<T> {
    pub re: T,
    pub im: T,
    pub alpha: T,
    /// the bar of s: an absolute distance in the complex plane
    pub bar: f64,
    /// the bar of alpha
    pub alpha_bar: f64,
}

impl<T: Real> Singularity<T> {
    /// |s| in f64
    pub fn radius(&self) -> f64 {
        hyp(self.re.to_f64(), self.im.to_f64())
    }
}

/// The result of [`locate`]
#[derive(Clone, Debug)]
pub struct Location<T> {
    pub kind: Kind,
    pub model: Model,
    /// the nearest singularity (Im >= 0)
    pub nearest: Singularity<T>,
    /// the second singularity of a three-term fit with two real roots, its bars from the neighbors that have one
    /// (infinite with none)
    pub farther: Option<Singularity<T>>,
    /// the first order and the degree of the chosen cell
    pub first: usize,
    pub degree: usize,
    /// the bar of s over |s|
    pub rel_bar: f64,
    /// for a pair, |Im g| of the first-order solution of its two exponents (g = -1 - alpha): zero for a conjugate pair
    /// of one exponent, which is what a pair is, so a value above the bars says the three-term model fits poorly; 0
    /// otherwise
    pub skew: f64,
}

/// sqrt(x^2 + y^2) by IEEE operations only, scaled against overflow: std's `hypot` is the platform's libm, whose
/// last bit differs between platforms, and the results of the locator are bitwise the same everywhere
fn hyp(x: f64, y: f64) -> f64 {
    let m = x.abs().max(y.abs());
    if m == 0.0 || !m.is_finite() {
        return m;
    }
    let (a, b) = (x / m, y / m);
    m * (a * a + b * b).sqrt()
}

/// A root of the fit of one cell
#[derive(Clone, Copy, Debug)]
struct Root<T> {
    re: T,
    im: T,
    alpha: T,
}

/// The estimate of one cell
#[derive(Clone, Copy, Debug)]
struct CellEst<T> {
    near: Root<T>,
    far: Option<Root<T>>,
    pair: bool,
    skew: f64,
}

/// least squares of the columns `a` (each of the length of `b`) to `b` by Householder QR in T; None with fewer rows
/// than columns or a diagonal of R below 100 rows * UNIT of the first (a rank-deficient fit)
fn lstsq<T: Real>(mut a: Vec<Vec<T>>, mut b: Vec<T>) -> Option<Vec<T>> {
    let (n, m) = (a.len(), b.len());
    if n == 0 || m < n {
        return None;
    }
    let (zero, two) = (T::from_f64(0.0), T::from_f64(2.0));
    for j in 0..n {
        let norm2 = a[j][j..].iter().fold(zero, |s, &x| s + x * x);
        if norm2.to_f64().is_nan() || norm2.to_f64() <= 0.0 {
            return None;
        }
        let norm = norm2.sqrt();
        let alpha = if a[j][j].to_f64() > 0.0 { -norm } else { norm };
        let mut v: Vec<T> = a[j][j..].to_vec();
        v[0] = v[0] - alpha;
        let vv = v.iter().fold(zero, |s, &x| s + x * x);
        if vv.to_f64() == 0.0 {
            continue;
        }
        let f = two / vv;
        let reflect = |col: &mut [T]| {
            let dot = col.iter().zip(&v).fold(zero, |s, (&x, &y)| s + x * y);
            let g = f * dot;
            for (x, &y) in col.iter_mut().zip(&v) {
                *x = *x - g * y;
            }
        };
        for col in &mut a[j..] {
            reflect(&mut col[j..]);
        }
        reflect(&mut b[j..]);
    }
    let r0 = a[0][0].to_f64().abs();
    let mut x = vec![zero; n];
    for j in (0..n).rev() {
        let d = a[j][j];
        let d_abs = d.to_f64().abs();
        if d_abs.is_nan() || d_abs <= 100.0 * m as f64 * T::UNIT * r0 {
            return None;
        }
        let s = (j + 1..n).fold(b[j], |s, k| s - a[k][j] * x[k]);
        x[j] = s / d;
    }
    Some(x)
}

/// the columns s^0 ... s^degree of the scaled variable s_i
fn power_columns<T: Real>(s: &[T], degree: usize) -> Vec<Vec<T>> {
    let mut cols = vec![vec![T::from_f64(1.0); s.len()]];
    for j in 1..=degree {
        let prev = &cols[j - 1];
        let next = prev.iter().zip(s).map(|(&p, &x)| p * x).collect();
        cols.push(next);
    }
    cols
}

/// the coefficients b_j of r_k = c_k / c_(k-1) = sum_j b_j / k^j over k = first ... K (K = c.len() - 1), fitted in the
/// scaled variable K / k and returned for 1 / k
fn ratio_fit<T: Real>(c: &[T], first: usize, degree: usize) -> Option<Vec<T>> {
    let last = c.len() - 1;
    let ks: Vec<usize> = (first.max(2)..=last).filter(|&k| c[k - 1].to_f64() != 0.0).collect();
    if ks.len() < degree.saturating_add(4) {
        return None;
    }
    let kk = T::from_f64(last as f64);
    let s: Vec<T> = ks.iter().map(|&k| kk / T::from_f64(k as f64)).collect();
    let rhs: Vec<T> = ks.iter().map(|&k| c[k] / c[k - 1]).collect();
    let x = lstsq(power_columns(&s, degree), rhs)?;
    let mut scale = T::from_f64(1.0);
    Some(
        x.into_iter()
            .map(|v| {
                let b = v * scale;
                scale = scale * kk;
                b
            })
            .collect(),
    )
}

/// (pi_j, theta_j) of (1 + 1/k) c_(k+1) = sum_j pi_j c_k / k^j - sum_j theta_j c_(k-1) / k^j over k = first ... K - 1,
/// each row scaled by its largest |c|
fn three_term_fit<T: Real>(c: &[T], first: usize, degree: usize) -> Option<(Vec<T>, Vec<T>)> {
    let last = c.len() - 1;
    let rows: Vec<(usize, T)> = (first.max(2)..last)
        .filter_map(|k| {
            let env = [c[k + 1], c[k], c[k - 1]]
                .into_iter()
                .max_by(|a, b| a.to_f64().abs().total_cmp(&b.to_f64().abs()))?;
            (env.to_f64() != 0.0).then_some((k, env))
        })
        .collect();
    if rows.len() < degree.saturating_add(1).saturating_mul(2).saturating_add(3) {
        return None;
    }
    let kk = T::from_f64(last as f64);
    let s: Vec<T> = rows.iter().map(|&(k, _)| kk / T::from_f64(k as f64)).collect();
    let pw = power_columns(&s, degree);
    let (mut ca, mut cb) = (Vec::new(), Vec::new());
    for p in &pw {
        ca.push(
            rows.iter()
                .zip(p)
                .map(|(&(k, env), &x)| (c[k] / env) * x)
                .collect::<Vec<T>>(),
        );
        cb.push(
            rows.iter()
                .zip(p)
                .map(|(&(k, env), &x)| -((c[k - 1] / env) * x))
                .collect::<Vec<T>>(),
        );
    }
    let rhs: Vec<T> = rows
        .iter()
        .map(|&(k, env)| (T::from_f64((k + 1) as f64) / T::from_f64(k as f64)) * (c[k + 1] / env))
        .collect();
    ca.extend(cb);
    let x = lstsq(ca, rhs)?;
    let mut scale = T::from_f64(1.0);
    let (mut pi, mut th) = (Vec::new(), Vec::new());
    for j in 0..=degree {
        pi.push(x[j] * scale);
        th.push(x[degree + 1 + j] * scale);
        scale = scale * kk;
    }
    Some((pi, th))
}

/// the estimate of one cell; None for degree 0, which fits neither the slope nor the first-order terms
fn cell<T: Real>(c: &[T], model: Model, first: usize, degree: usize) -> Option<CellEst<T>> {
    if degree == 0 {
        return None;
    }
    let zero = T::from_f64(0.0);
    let one = T::from_f64(1.0);
    match model {
        Model::Ratio => {
            let b = ratio_fit(c, first, degree)?;
            if b[0].to_f64() == 0.0 {
                return None;
            }
            let near = Root {
                re: one / b[0],
                im: zero,
                alpha: -one - b[1] / b[0],
            };
            Some(CellEst {
                near,
                far: None,
                pair: false,
                skew: 0.0,
            })
        }
        Model::ThreeTerm => {
            let (pi, th) = three_term_fit(c, first, degree)?;
            let (p0, p1, q0, q1) = (pi[0], pi[1] - pi[0], th[0], th[1] - th[0]);
            let (two, four) = (T::from_f64(2.0), T::from_f64(4.0));
            let disc = p0 * p0 - four * q0;
            if disc.to_f64() < 0.0 {
                // z = P_0 / 2 -+ i w, |z|^2 = Q_0, s = 1 / z = (P_0 / 2 + i w) / Q_0
                let w = (-disc).sqrt() / two;
                let inv = one / q0;
                let a = p1 - q1 * p0 / (two * q0);
                let near = Root {
                    re: p0 / two * inv,
                    im: w * inv,
                    alpha: -one - q1 / (two * q0),
                };
                Some(CellEst {
                    near,
                    far: None,
                    pair: true,
                    skew: (a / (two * w)).to_f64().abs(),
                })
            } else {
                let w = disc.sqrt() / two;
                let (mut za, mut zb) = (p0 / two + w, p0 / two - w);
                if za.to_f64().abs() < zb.to_f64().abs() {
                    std::mem::swap(&mut za, &mut zb);
                }
                let dz = za - zb;
                if za.to_f64() == 0.0 || dz.to_f64() == 0.0 {
                    return None;
                }
                let near = Root {
                    re: one / za,
                    im: zero,
                    alpha: -one - (p1 - q1 / za) / dz,
                };
                let far = (zb.to_f64() != 0.0).then(|| Root {
                    re: one / zb,
                    im: zero,
                    alpha: -one + (p1 - q1 / zb) / dz,
                });
                Some(CellEst {
                    near,
                    far,
                    pair: false,
                    skew: 0.0,
                })
            }
        }
    }
}

fn dist<T: Real>(a: &Root<T>, b: &Root<T>) -> f64 {
    hyp((a.re - b.re).to_f64(), (a.im - b.im).to_f64())
}

/// the cells of one model on a grid (firsts x degrees), None where a fit failed
struct Grid<T> {
    firsts: Vec<usize>,
    degrees: Vec<usize>,
    cells: Vec<Vec<Option<CellEst<T>>>>,
}

/// the values of `v` in their order of first appearance: a repeated value would be a neighbor that agrees exactly
fn distinct(v: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(v.len());
    for &x in v {
        if !out.contains(&x) {
            out.push(x);
        }
    }
    out
}

fn grid<T: Real>(c: &[T], model: Model, firsts: &[usize], degrees: &[usize]) -> Grid<T> {
    let (firsts, degrees) = (distinct(firsts), distinct(degrees));
    let cells = firsts
        .iter()
        .map(|&f| degrees.iter().map(|&d| cell(c, model, f, d)).collect())
        .collect();
    Grid { firsts, degrees, cells }
}

/// the largest disagreements of the cell (i, j) with its existing neighbors: (s, alpha, second singularity's (s, alpha)
/// when the neighbors have one); None with fewer than two neighbors
#[allow(clippy::type_complexity)]
fn spread<T: Real>(g: &Grid<T>, i: usize, j: usize) -> Option<(f64, f64, Option<(f64, f64)>)> {
    let me = g.cells[i][j].as_ref()?;
    let (mut ds, mut da, mut n) = (0.0f64, 0.0f64, 0);
    let (mut fs, mut fa, mut nf) = (0.0f64, 0.0f64, 0);
    let rows = i.saturating_sub(1)..=(i + 1).min(g.firsts.len() - 1);
    let cols = j.saturating_sub(1)..=(j + 1).min(g.degrees.len() - 1);
    for (ni, row) in g.cells.iter().enumerate().filter(|(ni, _)| rows.contains(ni)) {
        for (nj, nb) in row.iter().enumerate().filter(|(nj, _)| cols.contains(nj)) {
            if (ni, nj) == (i, j) {
                continue;
            }
            let Some(nb) = nb.as_ref() else {
                continue;
            };
            n += 1;
            let d_here = dist(&me.near, &nb.near);
            let da_here = (me.near.alpha - nb.near.alpha).to_f64().abs();
            if d_here.is_nan() || da_here.is_nan() {
                return None;
            }
            ds = ds.max(d_here);
            da = da.max(da_here);
            if let (Some(a), Some(b)) = (&me.far, &nb.far) {
                nf += 1;
                fs = fs.max(dist(a, b));
                fa = fa.max((a.alpha - b.alpha).to_f64().abs());
            }
        }
    }
    (n >= 2).then_some((ds, da, (nf > 0).then_some((fs, fa))))
}

/// the cell with the smallest relative disagreement: (i, j, ds / |s|)
fn best_cell<T: Real>(g: &Grid<T>) -> Option<(usize, usize, f64)> {
    let mut best: Option<(usize, usize, f64)> = None;
    for i in 0..g.firsts.len() {
        for j in 0..g.degrees.len() {
            let (Some(me), Some((ds, ..))) = (g.cells[i][j].as_ref(), spread(g, i, j)) else {
                continue;
            };
            let rel = ds / hyp(me.near.re.to_f64(), me.near.im.to_f64());
            if rel.is_finite() && best.is_none_or(|b| rel < b.2) {
                best = Some((i, j, rel));
            }
        }
    }
    best
}

fn unique_firsts(k: usize, starts: &[f64]) -> Vec<usize> {
    let mut f: Vec<usize> = starts
        .iter()
        .map(|&s| ((s * k as f64).round() as usize).clamp(2, k.saturating_sub(4)))
        .collect();
    f.sort_unstable();
    f.dedup();
    f
}

/// the result at cell (i, j) of a grid, with bars from its neighbors
fn assemble<T: Real>(g: &Grid<T>, model: Model, i: usize, j: usize, opts: &Opts) -> Option<Location<T>> {
    let me = g.cells[i][j].as_ref()?;
    let (ds, da, far) = spread(g, i, j)?;
    let radius = hyp(me.near.re.to_f64(), me.near.im.to_f64());
    let floor_s = 8.0 * T::UNIT * radius;
    let floor_a = 64.0 * T::UNIT * me.near.alpha.to_f64().abs().max(1.0);
    let bar = (opts.bar_factor * ds).max(floor_s);
    let nearest = Singularity {
        re: me.near.re,
        im: me.near.im,
        alpha: me.near.alpha,
        bar,
        alpha_bar: (opts.bar_factor * da).max(floor_a),
    };
    let farther = me.far.map(|f| {
        let r = hyp(f.re.to_f64(), f.im.to_f64());
        let (bs, ba) = far.map_or((f64::INFINITY, f64::INFINITY), |(s, a)| {
            (
                (opts.bar_factor * s).max(8.0 * T::UNIT * r),
                (opts.bar_factor * a).max(floor_a),
            )
        });
        Singularity {
            re: f.re,
            im: f.im,
            alpha: f.alpha,
            bar: bs,
            alpha_bar: ba,
        }
    });
    let rel_bar = bar / radius;
    let kind = if rel_bar.is_nan() || rel_bar > opts.max_rel_bar {
        Kind::Unresolved
    } else if me.pair {
        Kind::Pair
    } else {
        Kind::Real
    };
    Some(Location {
        kind,
        model,
        nearest,
        farther,
        first: g.firsts[i],
        degree: g.degrees[j],
        rel_bar,
        skew: me.skew,
    })
}

/// The nearest singularity of the series c_0 ... c_K (K = c.len() - 1 >= 12): both models over the grid of `opts`, and
/// of the two the cell with the smaller disagreement with its neighbors (the ratio model on a tie). None when no cell
/// has two neighbors that fit, or with fewer than 13 coefficients.
pub fn locate<T: Real>(c: &[T], opts: &Opts) -> Option<Location<T>> {
    if c.len() < 13 {
        return None;
    }
    let firsts = unique_firsts(c.len() - 1, &opts.starts);
    let mut best: Option<(f64, Location<T>)> = None;
    for (model, degrees) in [
        (Model::Ratio, &opts.ratio_degrees),
        (Model::ThreeTerm, &opts.three_term_degrees),
    ] {
        let g = grid(c, model, &firsts, degrees);
        let Some((i, j, rel)) = best_cell(&g) else {
            continue;
        };
        if best.as_ref().is_none_or(|b| rel < b.0) {
            if let Some(loc) = assemble(&g, model, i, j, opts) {
                best = Some((rel, loc));
            }
        }
    }
    best.map(|b| b.1)
}

/// The estimate of one cell of one model, with the bar from its eight neighbors (the first orders `first` and
/// `first` +- K / 10, the degrees `degree` and `degree` +- 1): for a window chosen by hand, and the controls of the
/// window. A `first` below 2 is taken as 2, where the fits start. None where the fit fails or the neighbors do, and
/// for `first` past K or a degree of K or more, which no window fits.
pub fn locate_at<T: Real>(c: &[T], model: Model, first: usize, degree: usize, opts: &Opts) -> Option<Location<T>> {
    if c.len() < 13 || first >= c.len() || degree >= c.len() - 1 {
        return None;
    }
    let first = first.max(2);
    let step = ((c.len() - 1) / 10).max(1);
    let mut firsts = Vec::new();
    if first >= step + 2 {
        firsts.push(first - step);
    }
    firsts.extend([first, first + step]);
    let mut degrees = Vec::new();
    if degree > 1 {
        degrees.push(degree - 1);
    }
    degrees.extend([degree, degree + 1]);
    let g = grid(c, model, &firsts, &degrees);
    let i = firsts.iter().position(|&f| f == first)?;
    let j = degrees.iter().position(|&d| d == degree)?;
    assemble(&g, model, i, j, opts)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// y = 3 - 2 s + s^2 / 2 at s = i / 5, i = 1 ... 20, fitted by `lstsq` with the columns 1, s, s^2: the coefficients
    /// to 1e-12 in f64, 1e-27 in double-double and 1e-57 in quad-double (the errors formed in the lane's arithmetic).
    /// The controls: one y moved by 1e-6 gives a coefficient error above 1e-9 in every lane, so the check fails on
    /// it; a NaN coefficient fails it too; and two equal columns are a rank-deficient fit, which gives None.
    fn planted_fit<T: Real>(tol: f64) {
        let s: Vec<T> = (1..=20).map(|i| T::from_f64(i as f64) / T::from_f64(5.0)).collect();
        let y = |bump: f64| -> Vec<T> {
            s.iter()
                .enumerate()
                .map(|(i, &x)| {
                    let v = T::from_f64(3.0) - T::from_f64(2.0) * x + x * x / T::from_f64(2.0);
                    if i == 7 {
                        v + T::from_f64(bump)
                    } else {
                        v
                    }
                })
                .collect()
        };
        let err = |x: &[T]| {
            [3.0, -2.0, 0.5]
                .iter()
                .zip(x)
                .map(|(&w, &g)| (g - T::from_f64(w)).to_f64().abs())
                .fold(0.0, crate::prec::nan_max)
        };
        let cols = power_columns(&s, 2);
        let e = err(&lstsq(cols.clone(), y(0.0)).unwrap());
        assert!(e <= tol, "planted fit error {e:e} above {tol:e}");
        let e = err(&lstsq(cols, y(1e-6)).unwrap());
        assert!(e > 1e-9, "control did not fire: error {e:e}");
        let nan = [3.0, -2.0, f64::NAN].map(T::from_f64);
        let e = err(&nan);
        assert!(e.is_nan(), "control did not fire: a NaN coefficient gives error {e:e}");
        let same = vec![s.clone(), s.clone()];
        assert!(
            lstsq(same, y(0.0)).is_none(),
            "control did not fire: rank-deficient fit"
        );
    }

    #[test]
    fn lstsq_on_a_planted_polynomial() {
        planted_fit::<f64>(1e-12);
        planted_fit::<DD>(1e-27);
        planted_fit::<QD>(1e-57);
    }
}
