//! The nearest singularities of a function from its Taylor coefficients: what limits the radius of a branch's series
//! ([`crate::anm::Branch`]). The root test's radius from the coefficients' envelope, the ratio method of Domb and
//! Sykes (Proc. R. Soc. Lond. A 240 (1957) 214) with the singularity's exponent, a two-term recurrence for a
//! conjugate pair, and the poles of Pade approximants (Baker and Graves-Morris, Pade Approximants, 2nd ed., Cambridge
//! University Press, 1996), which recur across the approximants' orders where the function has a singularity and
//! wander where it has none.
//!
//! The coefficients are c_1..c_N of z(a) - z(0) = sum_k c_k a^k; the estimators read the series g(a) = sum_k c_k
//! a^(k-1), which has the same singularities. A real series has them in conjugate pairs, and each estimator reports
//! the one with Im s >= 0.
//!
//! The approximants, their poles and their zeros are formed in the precision P, the roots by the Aberth-Ehrlich
//! iteration (Ehrlich, Commun. ACM 10 (1967) 107; Aberth, Math. Comp. 27 (1973) 339). A pole with a zero next to it
//! to P's rounding is a doublet, the approximant's excess degrees, which a genuine pole of small residue is not.
//!
//! The type of a real singularity, f ~ A (1 - h/s)^mu, comes from its exponent: a square-root point (mu = 1/2, a fold
//! of the real branch) or a pole (mu = -1), by the ratio method with a second correction ([`ratio_exponent`]) and by
//! Dlog Pade approximants ([`dlog_exponent`]). A quadratic Pade approximant ([`Shafer`]) continues a branch through a
//! square-root point onto its mirror sheet.
use crate::anm::Branch;
use crate::gcr::recip;
use crate::prec::nan_max;
use multidouble::dd::DD;
use multidouble::precision::Precision;
use multidouble::simd::Native;

/// A complex number in the precision P
pub struct Cx<P: Precision<Native>> {
    pub re: P::Scalar,
    pub im: P::Scalar,
}

impl<P: Precision<Native>> Clone for Cx<P> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<P: Precision<Native>> Copy for Cx<P> {}

impl<P: Precision<Native>> Cx<P> {
    pub fn new(re: P::Scalar, im: P::Scalar) -> Self {
        Cx { re, im }
    }
    /// re + i im from f64 values
    pub fn from_f64(re: f64, im: f64) -> Self {
        Cx::new(P::from_dd(DD::from_f64(re)), P::from_dd(DD::from_f64(im)))
    }
    pub fn real(re: P::Scalar) -> Self {
        Cx::new(re, P::ZERO)
    }
    pub fn scale(self, s: P::Scalar) -> Self {
        Cx::new(self.re * s, self.im * s)
    }
    /// 1 / self, through |self|^2 formed after scaling by a power of two, so that it neither overflows nor
    /// underflows
    pub fn inv(self) -> Self {
        let m = P::to_dd(self.re).hi.abs().max(P::to_dd(self.im).hi.abs());
        let sc = if m > 0.0 && m.is_finite() {
            P::from_dd(DD::from_f64(2f64.powi(-(m.log2().floor() as i32).clamp(-1000, 1000))))
        } else {
            P::ONE
        };
        let (re, im) = (self.re * sc, self.im * sc);
        let d = recip::<P>(re * re + im * im) * sc;
        Cx::new(re * d, -(im * d))
    }
    /// (re, im) rounded to f64
    pub fn to_f64(self) -> (f64, f64) {
        (P::to_dd(self.re).hi, P::to_dd(self.im).hi)
    }
    /// |self| in f64
    pub fn abs(self) -> f64 {
        let (re, im) = self.to_f64();
        re.hypot(im)
    }
}

impl<P: Precision<Native>> std::ops::Add for Cx<P> {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Cx::new(self.re + b.re, self.im + b.im)
    }
}

impl<P: Precision<Native>> std::ops::Sub for Cx<P> {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        Cx::new(self.re - b.re, self.im - b.im)
    }
}

impl<P: Precision<Native>> std::ops::Mul for Cx<P> {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        Cx::new(self.re * b.re - self.im * b.im, self.re * b.im + self.im * b.re)
    }
}

impl<P: Precision<Native>> std::ops::Div for Cx<P> {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, b: Self) -> Self {
        self * b.inv()
    }
}

/// p(x) and p'(x) for the real coefficients a (lowest first) by Horner's rule
fn horner<P: Precision<Native>>(a: &[P::Scalar], x: Cx<P>) -> (Cx<P>, Cx<P>) {
    let mut p = Cx::real(a[a.len() - 1]);
    let mut dp = Cx::real(P::ZERO);
    for &c in a[..a.len() - 1].iter().rev() {
        dp = dp * x + p;
        p = p * x + Cx::real(c);
    }
    (p, dp)
}

/// the unit roundoff of P, about
fn unit<P: Precision<Native>>() -> f64 {
    0.5f64.powi(52 * P::PARTS as i32)
}

/// The roots of the polynomial sum_k a_k x^k (lowest first; trailing zeros dropped) by the Aberth-Ehrlich iteration
/// in P, from points on a circle that encloses them (Fujiwara's bound), until every correction is below P's rounding
/// or 500 sweeps. Empty for a constant.
pub fn roots<P: Precision<Native>>(a: &[P::Scalar]) -> Vec<Cx<P>> {
    let mut a = a.to_vec();
    while a.len() > 1 && P::to_dd(a[a.len() - 1]).hi == 0.0 {
        a.pop();
    }
    let n = a.len() - 1;
    if n == 0 {
        return Vec::new();
    }
    let lead = P::to_dd(a[n]).hi.abs();
    let bound = (0..n)
        .map(|k| {
            let r = P::to_dd(a[k]).hi.abs() / lead / if k == 0 { 2.0 } else { 1.0 };
            r.powf(1.0 / (n - k) as f64)
        })
        .fold(0.0, nan_max);
    let r = if bound > 0.0 && bound.is_finite() { bound } else { 1.0 };
    let mut z: Vec<Cx<P>> = (0..n)
        .map(|k| {
            let t = 2.0 * std::f64::consts::PI * k as f64 / n as f64 + 0.4;
            Cx::from_f64(r * t.cos(), r * t.sin())
        })
        .collect();
    let tol = 16.0 * unit::<P>();
    for _ in 0..500 {
        let mut worst = 0.0f64;
        for k in 0..n {
            let (p, dp) = horner::<P>(&a, z[k]);
            if p.abs() == 0.0 {
                continue;
            }
            let ratio = p / dp;
            let mut sum = Cx::real(P::ZERO);
            for (j, &zj) in z.iter().enumerate() {
                if j != k {
                    sum = sum + (z[k] - zj).inv();
                }
            }
            let w = ratio / (Cx::real(P::ONE) - ratio * sum);
            if !w.abs().is_finite() {
                continue;
            }
            z[k] = z[k] - w;
            worst = worst.max(w.abs() / z[k].abs().max(f64::MIN_POSITIVE));
        }
        if worst <= tol {
            break;
        }
    }
    z
}

/// The [L/M] Pade approximant p/q of g(a) = sum_k g_k a^k (k < g.len()), L = g.len() - 1 - M, normalized to q_0 = 1:
/// (p_0..p_L, q_0..q_M) in P, or None when M exceeds g.len() - 1 or the approximant's linear system is singular. The
/// denominator solves sum_(j=1..M) g_(L+i-j) q_j = -g_(L+i), i = 1..M, by Gaussian elimination with partial pivoting
/// in P; the numerator is p_i = sum_(j=0..min(i, M)) q_j g_(i-j).
#[allow(clippy::type_complexity)]
pub fn pade<P: Precision<Native>>(g: &[P::Scalar], m: usize) -> Option<(Vec<P::Scalar>, Vec<P::Scalar>)> {
    if g.is_empty() || m > g.len() - 1 {
        return None;
    }
    let l = g.len() - 1 - m;
    let at = |i: isize| if i >= 0 { g[i as usize] } else { P::ZERO };
    let mut a: Vec<Vec<P::Scalar>> = (1..=m)
        .map(|i| (1..=m).map(|j| at(l as isize + i as isize - j as isize)).collect())
        .collect();
    let mut b: Vec<P::Scalar> = (1..=m).map(|i| -at((l + i) as isize)).collect();
    for c in 0..m {
        let piv = (c..m).max_by(|&x, &y| P::to_dd(a[x][c]).hi.abs().total_cmp(&P::to_dd(a[y][c]).hi.abs()))?;
        let pv = P::to_dd(a[piv][c]).hi;
        if pv == 0.0 || !pv.is_finite() {
            return None;
        }
        a.swap(c, piv);
        b.swap(c, piv);
        let inv = recip::<P>(a[c][c]);
        for r in c + 1..m {
            let f = a[r][c] * inv;
            let (top, bot) = a.split_at_mut(r);
            for (x, &y) in bot[0][c..].iter_mut().zip(&top[c][c..]) {
                *x = *x - f * y;
            }
            b[r] = b[r] - f * b[c];
        }
    }
    let mut q = vec![P::ZERO; m + 1];
    q[0] = P::ONE;
    for c in (0..m).rev() {
        let s = (c + 1..m).fold(b[c], |s, j| s - a[c][j] * q[j + 1]);
        q[c + 1] = s * recip::<P>(a[c][c]);
    }
    let p = (0..=l)
        .map(|i| (0..=i.min(m)).fold(P::ZERO, |t, j| t + q[j] * g[i - j]))
        .collect();
    Some((p, q))
}

/// The settings of [`stable_poles`]
#[derive(Clone, Debug)]
pub struct PoleOpts {
    /// the denominator degrees M of the approximants
    pub ms: Vec<usize>,
    /// a numerator zero within this fraction of |s| marks the pole s a doublet (None: 2^(-26 PARTS), half P's digits)
    pub doublet: Option<f64>,
    /// poles of different M within this fraction of each other form one cluster
    pub cluster: f64,
    /// a cluster of poles from at least this many M is stable
    pub min_count: usize,
    /// a pole nearer than this fraction of the root test's radius lies inside the disc of convergence, where no
    /// singularity can be, so it is the approximant's
    pub inside: f64,
    /// poles farther than this are left out
    pub r_max: f64,
}

impl Default for PoleOpts {
    fn default() -> Self {
        PoleOpts {
            ms: (2..=10).collect(),
            doublet: None,
            cluster: 0.05,
            min_count: 3,
            inside: 0.85,
            r_max: 8.0,
        }
    }
}

/// A stable pole: the median s of its cluster (Im s >= 0), |s|, arg s in radians, the M it recurs in, and the
/// cluster's spread max |s_M - s| / |s|
#[derive(Clone, Debug)]
pub struct Pole {
    pub s: (f64, f64),
    pub abs: f64,
    pub arg: f64,
    pub ms: Vec<usize>,
    pub spread: f64,
}

/// poles of distinct M near the first of them (the seed)
struct Cluster {
    seed: (f64, f64),
    ms: Vec<usize>,
    pts: Vec<(f64, f64)>,
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// The poles of the [L/M] approximants of g(a) = sum_k c_k a^(k-1) (c = c_1..c_N in P) for the M of `opts` that
/// recur across them, nearest first. A pole counts when Im s >= 0 (its conjugate stands for one below), it is no
/// doublet, and `inside` times the root test's radius <= |s| <= `r_max`; poles of distinct M within `cluster` of the
/// cluster's first form one, and a cluster of `min_count` M or more is stable.
pub fn stable_poles<P: Precision<Native>>(c: &[P::Scalar], opts: &PoleOpts) -> Vec<Pole> {
    let c64: Vec<f64> = c.iter().map(|&v| P::to_dd(v).hi).collect();
    let r_min = opts.inside * root_radius(&c64, None);
    let doublet = opts.doublet.unwrap_or(0.5f64.powi(26 * P::PARTS as i32));
    let mut cand: Vec<(usize, (f64, f64))> = Vec::new();
    for &m in &opts.ms {
        let Some((p, q)) = pade::<P>(c, m) else {
            continue;
        };
        let zeros = roots::<P>(&p);
        for s in roots::<P>(&q) {
            let (re, im) = s.to_f64();
            let a = re.hypot(im);
            if !a.is_finite() || im < -1e-12 * a || a > opts.r_max || a < r_min {
                continue;
            }
            if zeros.iter().any(|&z| (s - z).abs() <= doublet * a) {
                continue;
            }
            cand.push((m, (re, im.abs())));
        }
    }
    cand.sort_by(|x, y| x.1 .0.hypot(x.1 .1).total_cmp(&y.1 .0.hypot(y.1 .1)));
    let mut clusters: Vec<Cluster> = Vec::new();
    for (m, s) in cand {
        let near = clusters.iter_mut().find(|cl| {
            (s.0 - cl.seed.0).hypot(s.1 - cl.seed.1) <= opts.cluster * cl.seed.0.hypot(cl.seed.1) && !cl.ms.contains(&m)
        });
        match near {
            Some(cl) => {
                cl.ms.push(m);
                cl.pts.push(s);
            }
            None => clusters.push(Cluster {
                seed: s,
                ms: vec![m],
                pts: vec![s],
            }),
        }
    }
    let mut out: Vec<Pole> = clusters
        .into_iter()
        .filter(|cl| cl.ms.len() >= opts.min_count)
        .map(|Cluster { mut ms, pts, .. }| {
            let s = (
                median(pts.iter().map(|p| p.0).collect()),
                median(pts.iter().map(|p| p.1).collect()),
            );
            let abs = s.0.hypot(s.1);
            let spread = pts.iter().map(|p| (p.0 - s.0).hypot(p.1 - s.1)).fold(0.0, nan_max) / abs;
            ms.sort_unstable();
            Pole {
                s,
                abs,
                arg: s.1.atan2(s.0),
                ms,
                spread,
            }
        })
        .collect();
    out.sort_by(|x, y| x.abs.total_cmp(&y.abs));
    out
}

/// The root test's radius of sum_k c_k a^k (c = c_1..c_N): exp(-slope) of the least-squares line through
/// ln env_k against k, env_k the largest |c| of orders k - 1, k and k + 1 (which a conjugate pair's dips need), over
/// the orders from `from` (N / 2 by default) on. NaN with fewer than two usable orders.
pub fn root_radius(c: &[f64], from: Option<usize>) -> f64 {
    let n = c.len();
    let k0 = from.unwrap_or(n / 2).max(1);
    let pts: Vec<(f64, f64)> = (1..=n)
        .filter(|&k| k >= k0)
        .filter_map(|k| {
            let env = c[k.saturating_sub(2)..(k + 1).min(n)]
                .iter()
                .map(|v| v.abs())
                .fold(0.0, nan_max);
            (env > 0.0).then(|| (k as f64, env.ln()))
        })
        .collect();
    match line(&pts) {
        Some((_, slope)) => (-slope).exp(),
        None => f64::NAN,
    }
}

/// the least-squares line y = a + b x through the points: (a, b), None with fewer than two distinct x
fn line(pts: &[(f64, f64)]) -> Option<(f64, f64)> {
    let n = pts.len() as f64;
    if pts.len() < 2 {
        return None;
    }
    let mx = pts.iter().map(|p| p.0).sum::<f64>() / n;
    let my = pts.iter().map(|p| p.1).sum::<f64>() / n;
    let sxx: f64 = pts.iter().map(|p| (p.0 - mx) * (p.0 - mx)).sum();
    let sxy: f64 = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
    if sxx == 0.0 {
        return None;
    }
    let b = sxy / sxx;
    Some((my - b * mx, b))
}

/// The ratio method of Domb and Sykes for a singularity (1 - a/s)^(-gamma) on the real axis: c_k / c_(k-1) =
/// (1 / s)(1 + (gamma - 1) / k) + O(1/k^2), fitted as a line in 1/k over the last half of the ratios (c = c_1..c_N):
/// (s, gamma), None with fewer than three ratios there or a zero intercept.
pub fn domb_sykes(c: &[f64]) -> Option<(f64, f64)> {
    let r: Vec<(f64, f64)> = (2..=c.len())
        .filter(|&k| c[k - 2] != 0.0)
        .map(|k| (1.0 / k as f64, c[k - 1] / c[k - 2]))
        .collect();
    let tail = &r[r.len() / 2..];
    if tail.len() < 3 {
        return None;
    }
    let (ic, slope) = line(tail)?;
    if ic == 0.0 {
        return None;
    }
    Some((1.0 / ic, 1.0 + slope / ic))
}

/// The two-term recurrence c_k = p c_(k-1) + q c_(k-2) of a conjugate pair of poles, fitted by least squares over
/// windows of `w` orders (c = c_1..c_N, k counted from 1), and the root 1/s of x^2 - p x - q of larger modulus in each:
/// the windows' (first order, s), the median s over the windows from order `from` on, and their spread
/// max |s_w - s| / |s|.
#[derive(Clone, Debug)]
pub struct Recurrence {
    pub windows: Vec<(usize, (f64, f64))>,
    pub s: Option<(f64, f64)>,
    pub spread: f64,
}

/// The [`Recurrence`] fit of c_1..c_N
pub fn recurrence(c: &[f64], from: usize, w: usize) -> Recurrence {
    let n = c.len();
    let mut windows = Vec::new();
    for k0 in 3..(n + 2).saturating_sub(w) {
        let ks = k0..k0 + w;
        // columns c_(k-1), c_(k-2) and right-hand side c_k, scaled by their largest entry
        let sc = ks
            .clone()
            .flat_map(|k| [c[k - 2].abs(), c[k - 3].abs()])
            .fold(0.0, nan_max);
        if sc == 0.0 || !sc.is_finite() {
            continue;
        }
        let a1: Vec<f64> = ks.clone().map(|k| c[k - 2] / sc).collect();
        let a2: Vec<f64> = ks.clone().map(|k| c[k - 3] / sc).collect();
        let b: Vec<f64> = ks.map(|k| c[k - 1] / sc).collect();
        // least squares by modified Gram-Schmidt on the two columns
        let dot = |x: &[f64], y: &[f64]| x.iter().zip(y).map(|(a, b)| a * b).sum::<f64>();
        let r11 = dot(&a1, &a1).sqrt();
        if r11 == 0.0 {
            continue;
        }
        let q1: Vec<f64> = a1.iter().map(|v| v / r11).collect();
        let r12 = dot(&q1, &a2);
        let v2: Vec<f64> = a2.iter().zip(&q1).map(|(a, q)| a - r12 * q).collect();
        let r22 = dot(&v2, &v2).sqrt();
        if r22 == 0.0 {
            continue;
        }
        let q2: Vec<f64> = v2.iter().map(|v| v / r22).collect();
        let (y1, y2) = (dot(&q1, &b), dot(&q2, &b));
        let q = y2 / r22;
        let p = (y1 - r12 * q) / r11;
        // x^2 - p x - q = 0: the root of larger modulus
        let disc = p * p + 4.0 * q;
        let x = if disc >= 0.0 {
            let r = disc.sqrt();
            let (x1, x2) = ((p + r) / 2.0, (p - r) / 2.0);
            if x1.abs() >= x2.abs() {
                (x1, 0.0)
            } else {
                (x2, 0.0)
            }
        } else {
            (p / 2.0, (-disc).sqrt() / 2.0)
        };
        let m2 = x.0 * x.0 + x.1 * x.1;
        if m2 == 0.0 {
            continue;
        }
        windows.push((k0, (x.0 / m2, (x.1 / m2).abs())));
    }
    let sel: Vec<(f64, f64)> = windows.iter().filter(|(k0, _)| *k0 >= from).map(|w| w.1).collect();
    if sel.is_empty() {
        return Recurrence {
            windows,
            s: None,
            spread: f64::NAN,
        };
    }
    let s = (
        median(sel.iter().map(|v| v.0).collect()),
        median(sel.iter().map(|v| v.1).collect()),
    );
    let spread = sel.iter().map(|v| (v.0 - s.0).hypot(v.1 - s.1)).fold(0.0, nan_max) / s.0.hypot(s.1);
    Recurrence {
        windows,
        s: Some(s),
        spread,
    }
}

/// c_1..c_N of unknown i of a branch's series: the scalar series the estimators read
pub fn component<P: Precision<Native>>(br: &Branch<P>, i: usize) -> Vec<P::Scalar> {
    br.z[1..].iter().map(|zk| zk[i]).collect()
}

/// A singularity's place s and exponent mu, f ~ A (1 - h/s)^mu (mu = 1/2 a square-root point, a fold of the real
/// branch; mu = -1 a simple pole): the medians over an estimator's windows or approximants, their ranges (spreads
/// over the estimator's choices, not error bars), and how many entered
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Exponent {
    pub s: f64,
    pub mu: f64,
    pub s_spread: f64,
    pub mu_spread: f64,
    pub n: usize,
}

/// the median and the range of the values (not empty)
fn median_range(mut v: Vec<f64>) -> (f64, f64) {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    let med = if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    };
    (med, v[n - 1] - v[0])
}

/// the exponent from the estimates (s, mu), None when there are none
fn exponent_of(est: Vec<(f64, f64)>) -> Option<Exponent> {
    if est.is_empty() {
        return None;
    }
    let n = est.len();
    let (s, s_spread) = median_range(est.iter().map(|e| e.0).collect());
    let (mu, mu_spread) = median_range(est.iter().map(|e| e.1).collect());
    Some(Exponent {
        s,
        mu,
        s_spread,
        mu_spread,
        n,
    })
}

/// The ratio method with a second correction: for f ~ A (1 - h/s)^mu alone, c_k / c_(k-1) = (1/s)(1 - (1 + mu)/k)
/// exactly, and the ratios over the orders k0..N are fitted by least squares as (1/s)(1 + b/k + e/k^2), so that
/// mu = -1 - b, for each first order k0 in `k0s` (C. Domb and M. F. Sykes, Proc. R. Soc. Lond. A 240 (1957) 214,
/// doi:10.1098/rspa.1957.0078). c = c_1..c_N; None without a window of four ratios.
pub fn ratio_exponent(c: &[f64], k0s: impl IntoIterator<Item = usize>) -> Option<Exponent> {
    let n = c.len();
    let mut est = Vec::new();
    for k0 in k0s {
        let ks: Vec<usize> = (k0.max(2)..=n).filter(|&k| c[k - 2] != 0.0).collect();
        if ks.len() < 4 {
            continue;
        }
        // the normal equations of the fit y = a0 + a1 / k + a2 / k^2
        let mut ata = [[0.0f64; 3]; 3];
        let mut aty = [0.0f64; 3];
        for &k in &ks {
            let x = [1.0, 1.0 / k as f64, 1.0 / (k * k) as f64];
            let y = c[k - 1] / c[k - 2];
            for i in 0..3 {
                aty[i] += x[i] * y;
                for j in 0..3 {
                    ata[i][j] += x[i] * x[j];
                }
            }
        }
        let Some(a) = solve3(ata, aty) else { continue };
        if a[0] == 0.0 || !a[0].is_finite() {
            continue;
        }
        est.push((1.0 / a[0], -1.0 - a[1] / a[0]));
    }
    exponent_of(est)
}

/// the solution of the 3 x 3 system by Gaussian elimination with partial pivoting, None when it is singular
fn solve3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    for c in 0..3 {
        let p = (c..3).max_by(|&x, &y| a[x][c].abs().total_cmp(&a[y][c].abs()))?;
        if a[p][c] == 0.0 {
            return None;
        }
        a.swap(c, p);
        b.swap(c, p);
        for r in c + 1..3 {
            let f = a[r][c] / a[c][c];
            let rc = a[c];
            for (x, y) in a[r].iter_mut().zip(rc).skip(c) {
                *x -= f * y;
            }
            b[r] -= f * b[c];
        }
    }
    let mut x = [0.0; 3];
    for c in (0..3).rev() {
        x[c] = (b[c] - (c + 1..3).map(|j| a[c][j] * x[j]).sum::<f64>()) / a[c][c];
    }
    Some(x)
}

/// The settings of [`dlog_exponent`]
#[derive(Clone, Debug)]
pub struct DlogOpts {
    /// the denominator degrees M; each with the numerator degrees M - 1, M and M + 1 that the orders allow
    pub ms: Vec<usize>,
    /// a pole is real when |Im s| is at most this fraction of |s|
    pub real: f64,
    /// a numerator zero within this fraction of |s| makes the pole s a doublet
    pub doublet: f64,
    /// a residue within this of 0 (a Froissart doublet) or of 1 (a zero of g) is skipped
    pub residue: f64,
}

impl Default for DlogOpts {
    fn default() -> Self {
        DlogOpts {
            ms: (3..10).collect(),
            real: 0.05,
            doublet: 1e-6,
            residue: 0.05,
        }
    }
}

/// The Dlog Pade estimator: g = df/dh = sum_j (j + 1) c_(j+1) h^j has the singularity (1 - h/s)^(mu - 1), a pole of
/// g'/g at s with residue mu - 1. For each [L/M] approximant of g'/g (`DlogOpts::ms`), the real pole with Re s > 0
/// nearest the origin that is neither a doublet nor of a residue near 0 or 1 (G. A. Baker, Jr. and P. Graves-Morris,
/// Pade Approximants, 2nd ed., Cambridge University Press, 1996, ch. 8). c = c_1..c_N in P; None when no approximant
/// gives such a pole.
pub fn dlog_exponent<P: Precision<Native>>(c: &[P::Scalar], opts: &DlogOpts) -> Option<Exponent> {
    if c.len() < 3 {
        return None;
    }
    let g: Vec<P::Scalar> = (0..c.len())
        .map(|j| c[j] * P::from_dd(DD::from_f64((j + 1) as f64)))
        .collect();
    if P::to_dd(g[0]).hi == 0.0 {
        return None;
    }
    let dg: Vec<P::Scalar> = (0..g.len() - 1)
        .map(|j| g[j + 1] * P::from_dd(DD::from_f64((j + 1) as f64)))
        .collect();
    // q = g'/g order by order: q_k = (dg_k - sum_(j=1..k) g_j q_(k-j)) / g_0
    let ig = recip::<P>(g[0]);
    let mut q: Vec<P::Scalar> = Vec::with_capacity(dg.len());
    for k in 0..dg.len() {
        let s = (1..=k).fold(dg[k], |s, j| s - g[j] * q[k - j]);
        q.push(s * ig);
    }
    let mut est = Vec::new();
    for &m in &opts.ms {
        for l in [m.saturating_sub(1), m, m + 1] {
            if l + m + 1 > q.len() || m == 0 {
                continue;
            }
            let Some((num, den)) = pade::<P>(&q[..l + m + 1], m) else {
                continue;
            };
            let zeros = if num.len() > 1 { roots::<P>(&num) } else { Vec::new() };
            let dden: Vec<P::Scalar> = (1..den.len())
                .map(|k| den[k] * P::from_dd(DD::from_f64(k as f64)))
                .collect();
            let mut best: Option<(f64, f64, f64)> = None;
            for z in roots::<P>(&den) {
                let (re, im) = z.to_f64();
                let az = z.abs();
                if im.abs() > opts.real * az || re <= 0.0 {
                    continue;
                }
                if zeros.iter().any(|&w| (z - w).abs() < opts.doublet * az) {
                    continue;
                }
                let res = (horner::<P>(&num, z).0 / horner::<P>(&dden, z).0).to_f64().0;
                if res.abs() < opts.residue || (res - 1.0).abs() < opts.residue {
                    continue;
                }
                if best.is_none_or(|b| az < b.0) {
                    best = Some((az, re, res));
                }
            }
            if let Some((_, re, res)) = best {
                est.push((re, res + 1.0));
            }
        }
    }
    exponent_of(est)
}

/// sqrt(d) in P by Newton's method from f64's square root, the root with Re >= 0
fn csqrt_p<P: Precision<Native>>(d: Cx<P>) -> Cx<P> {
    let (re, im) = d.to_f64();
    let r = re.hypot(im);
    if r == 0.0 {
        return Cx::real(P::ZERO);
    }
    let a = ((r + re) / 2.0).max(0.0).sqrt();
    let b = ((r - re) / 2.0).max(0.0).sqrt();
    let mut s = Cx::<P>::from_f64(a, if im < 0.0 { -b } else { b });
    let half = P::from_dd(DD::from_f64(0.5));
    for _ in 0..5 {
        s = (s + d / s).scale(half);
    }
    s
}

/// A quadratic Pade approximant (Shafer): polynomials P, Q, R with P f^2 + Q f + R = O(h^(l + m + n + 2)) for the
/// series f, p_0 = 1 (D. Shafer, SIAM J. Numer. Anal. 11 (1974) 447, doi:10.1137/0711037). Its two roots in f,
/// (-Q +- sqrt(Q^2 - 4 P R)) / (2 P), are two sheets of a function with square-root branch points, the zeros of the
/// discriminant Q^2 - 4 P R: about a fold, the branch the series follows and its mirror.
#[derive(Clone, Debug)]
pub struct Shafer<P: Precision<Native>> {
    pub p: Vec<P::Scalar>,
    pub q: Vec<P::Scalar>,
    pub r: Vec<P::Scalar>,
}

/// the product of two series truncated to n terms
fn series_mul<P: Precision<Native>>(a: &[P::Scalar], b: &[P::Scalar], n: usize) -> Vec<P::Scalar> {
    (0..n)
        .map(|k| {
            (0..=k)
                .filter(|&i| i < a.len() && k - i < b.len())
                .fold(P::ZERO, |s, i| s + a[i] * b[k - i])
        })
        .collect()
}

/// a polynomial's real coefficients (lowest first) evaluated at a complex point
fn poly_at<P: Precision<Native>>(a: &[P::Scalar], x: Cx<P>) -> Cx<P> {
    horner::<P>(a, x).0
}

impl<P: Precision<Native>> Shafer<P> {
    /// the approximant of degrees (l, m, n) of P, Q, R from f_0..f_(l+m+n+1) (f.len() at least l + m + n + 2), by
    /// Gaussian elimination with partial pivoting in P on its l + m + n + 2 conditions; None when f is too short or
    /// the system singular
    pub fn new(f: &[P::Scalar], (l, m, n): (usize, usize, usize)) -> Option<Self> {
        let k = l + m + n + 2;
        if f.len() < k {
            return None;
        }
        let f2 = series_mul::<P>(f, f, k);
        // unknowns: p_1..p_l, q_0..q_m, r_0..r_n; condition at order o: sum p_i f2_(o-i) + sum q_i f_(o-i) + r_o = 0
        let mut a: Vec<Vec<P::Scalar>> = vec![vec![P::ZERO; k]; k];
        let mut b: Vec<P::Scalar> = vec![P::ZERO; k];
        for o in 0..k {
            b[o] = -f2[o];
            for i in 1..=l {
                if i <= o {
                    a[o][i - 1] = f2[o - i];
                }
            }
            for i in 0..=m {
                if i <= o {
                    a[o][l + i] = f[o - i];
                }
            }
            if o <= n {
                a[o][l + m + 1 + o] = P::ONE;
            }
        }
        let x = solve_dense::<P>(a, b)?;
        let mut p = vec![P::ONE];
        p.extend_from_slice(&x[..l]);
        Some(Shafer {
            p,
            q: x[l..l + m + 1].to_vec(),
            r: x[l + m + 1..].to_vec(),
        })
    }

    /// the discriminant Q^2 - 4 P R, lowest coefficient first
    pub fn discriminant(&self) -> Vec<P::Scalar> {
        let d = self.q.len().max(self.p.len() + self.r.len() - 1);
        let qq = series_mul::<P>(&self.q, &self.q, 2 * self.q.len() - 1);
        let pr = series_mul::<P>(&self.p, &self.r, self.p.len() + self.r.len() - 1);
        let four = P::from_dd(DD::from_f64(4.0));
        (0..(2 * self.q.len() - 1).max(d))
            .map(|i| {
                let a = if i < qq.len() { qq[i] } else { P::ZERO };
                let b = if i < pr.len() { pr[i] } else { P::ZERO };
                a - four * b
            })
            .collect()
    }

    /// the branch points, the discriminant's roots
    pub fn branch_points(&self) -> Vec<Cx<P>> {
        roots::<P>(&self.discriminant())
    }

    /// the two roots (-Q + sqrt(D), -Q - sqrt(D)) / (2 P) at h, sqrt the root with Re >= 0
    pub fn sheets(&self, h: Cx<P>) -> (Cx<P>, Cx<P>) {
        let (pv, qv, rv) = (
            poly_at::<P>(&self.p, h),
            poly_at::<P>(&self.q, h),
            poly_at::<P>(&self.r, h),
        );
        let four = Cx::real(P::from_dd(DD::from_f64(4.0)));
        let sq = csqrt_p::<P>(qv * qv - four * pv * rv);
        let den = (pv + pv).inv();
        ((Cx::real(P::ZERO) - qv + sq) * den, (Cx::real(P::ZERO) - qv - sq) * den)
    }
}

/// the solution of the dense system a x = b by Gaussian elimination with partial pivoting in P, None when a pivot is
/// zero or not finite
fn solve_dense<P: Precision<Native>>(mut a: Vec<Vec<P::Scalar>>, mut b: Vec<P::Scalar>) -> Option<Vec<P::Scalar>> {
    let n = b.len();
    for c in 0..n {
        let piv = (c..n).max_by(|&x, &y| P::to_dd(a[x][c]).hi.abs().total_cmp(&P::to_dd(a[y][c]).hi.abs()))?;
        let pv = P::to_dd(a[piv][c]).hi;
        if pv == 0.0 || !pv.is_finite() {
            return None;
        }
        a.swap(c, piv);
        b.swap(c, piv);
        let inv = recip::<P>(a[c][c]);
        for r in c + 1..n {
            let f = a[r][c] * inv;
            let (top, bot) = a.split_at_mut(r);
            for (x, &y) in bot[0][c..].iter_mut().zip(&top[c][c..]) {
                *x = *x - f * y;
            }
            b[r] = b[r] - f * b[c];
        }
    }
    let mut x = vec![P::ZERO; n];
    for c in (0..n).rev() {
        let s = (c + 1..n).fold(b[c], |s, j| s - a[c][j] * x[j]);
        x[c] = s * recip::<P>(a[c][c]);
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Double, Quad};

    /// the roots' largest distance from the wanted ones (each matched to its nearest), the difference formed in P
    /// before it is rounded, so that errors far below f64's rounding show
    fn root_err<P: Precision<Native>>(got: &[Cx<P>], want: &[(f64, f64)]) -> f64 {
        want.iter()
            .map(|&w| {
                got.iter()
                    .map(|&g| (g - Cx::from_f64(w.0, w.1)).abs())
                    .fold(f64::INFINITY, f64::min)
            })
            .fold(0.0, nan_max)
    }

    /// the roots with the first moved by eps (in P): a result off by eps
    fn moved<P: Precision<Native>>(r: &[Cx<P>], eps: f64) -> Vec<Cx<P>> {
        let mut r = r.to_vec();
        r[0] = r[0] + Cx::from_f64(eps, 0.0);
        r
    }

    /// (x - 1)(x - 2)(x + 3)(x^2 + 1) = x^5 - 6 x^3 + 6 x^2 - 7 x + 6: its roots 1, 2, -3, +-i to 1e-25 in
    /// double-double and 1e-55 in quad-double. The controls: the roots with one moved by 1e-20 (double-double) and
    /// 1e-50 (quad-double), errors that f64 cannot hold, fail the checks, and the polynomial with one coefficient off
    /// by 1e-6 has its roots off by more than 1e-9.
    #[test]
    fn roots_of_a_known_polynomial() {
        let a = [6.0, -7.0, 6.0, -6.0, 0.0, 1.0];
        let want = [(1.0, 0.0), (2.0, 0.0), (-3.0, 0.0), (0.0, 1.0), (0.0, -1.0)];
        let d: Vec<DD> = a.iter().map(|&v| DD::from_f64(v)).collect();
        let rd = roots::<Double>(&d);
        let e = root_err::<Double>(&rd, &want);
        let q: Vec<_> = a
            .iter()
            .map(|&v| <Quad as Precision<Native>>::from_dd(DD::from_f64(v)))
            .collect();
        let rq = roots::<Quad>(&q);
        let eq = root_err::<Quad>(&rq, &want);
        assert!(e <= 1e-25 && eq <= 1e-55, "{e:e} {eq:e}");
        assert!(
            root_err::<Double>(&moved(&rd, 1e-20), &want) > 1e-25,
            "double-double control did not fire"
        );
        assert!(
            root_err::<Quad>(&moved(&rq, 1e-50), &want) > 1e-55,
            "quad-double control did not fire"
        );
        let mut bad = d.clone();
        bad[2] = DD::from_f64(6.0 + 1e-6);
        assert!(
            root_err::<Double>(&roots::<Double>(&bad), &want) > 1e-9,
            "control did not fire"
        );
    }

    /// The [1/1] approximant of 1 / (1 - a / 2) + 3 is exact: its pole is 2 to 1e-28, the difference formed in
    /// double-double. The controls: the pole moved by 1e-20, an error f64 cannot hold, fails the check, and the series
    /// of 1 / (1 - a / 2.1) gives another pole, off by more than 0.05.
    #[test]
    fn pade_of_a_rational_function() {
        let pole = |s: f64| {
            // 3 + 1 / (1 - a / s) = 4 + a / s + a^2 / s^2 + ...
            let g = [
                DD::from_f64(4.0),
                DD::ONE / DD::from_f64(s),
                DD::ONE / DD::from_f64(s * s),
            ];
            let (_, q) = pade::<Double>(&g, 1).unwrap();
            roots::<Double>(&q)
        };
        let p = pole(2.0);
        assert!(root_err::<Double>(&p, &[(2.0, 0.0)]) <= 1e-28, "{:?}", p[0].to_f64());
        assert!(
            root_err::<Double>(&moved(&p, 1e-20), &[(2.0, 0.0)]) > 1e-28,
            "control did not fire"
        );
        assert!(
            root_err::<Double>(&pole(2.1), &[(2.0, 0.0)]) > 0.05,
            "control did not fire"
        );
    }
}
