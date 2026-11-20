//! The data of the graphs in docs/graphs, printed as JSON for scripts/graphs.py: planted problems with known solutions
//! run through the crate's methods. Run with `cargo run --release --example graphs`.
//!
//! - `ladder`: max |F| per step of Newton's method in double-, quad- and octo-double on one planted system, run past
//!   each precision's floor
//! - `time`: max |F| against the seconds since the start, for Newton's method and the methods that factor less often
//!   or in a lower precision, in octo-double on one thread, five solves each
//! - `fold`: a branch with two folds followed by pseudo-arclength continuation, and continuation in p for contrast
//! - `deflation`: the iterates of successive deflated searches from one start
//! - `winding`: the adaptive samples of the argument principle on a square, a close pair of zeros among those inside

use std::cell::{Cell, RefCell};
use std::time::Instant;

use multisolver::arclength::{continuation, locate_fold, start, ArcOpts, AtParam, Bordered, FoldOpts, Path, Point};
use multisolver::deflation::{roots_from, DeflationOpts, Found};
use multisolver::multidouble::linalg::ld_for;
use multisolver::newton::{
    exact, lowered, newton, newton_with, reuse, Damping, Exact, Lower, NewtonOpts, NewtonReport,
};
use multisolver::prec::parts;
use multisolver::quasi::{newton_quasi, QuasiOpts};
use multisolver::winding::{winding, WindOpts, Winding, C64};
use multisolver::{Double, Family, Native, Octo, Precision, Problem, Quad, DD};

fn s<P: Precision<Native>>(x: f64) -> P::Scalar {
    P::from_dd(DD::from_f64(x))
}

fn hi<P: Precision<Native>>(x: P::Scalar) -> f64 {
    P::to_dd(x).hi
}

fn scalars<P: Precision<Native>>(z: &[&[f64]]) -> Vec<P::Scalar> {
    (0..z[0].len())
        .map(|i| P::from_raw(&z.iter().map(|p| p[i]).collect::<Vec<_>>()))
        .collect()
}

fn to_parts<P: Precision<Native>>(v: &[P::Scalar]) -> Vec<Vec<f64>> {
    (0..P::PARTS)
        .map(|q| v.iter().map(|&x| P::part(x, q)).collect())
        .collect()
}

fn max_abs<P: Precision<Native>>(v: &[P::Scalar]) -> f64 {
    v.iter().map(|&x| hi::<P>(x).abs()).fold(0.0, f64::max)
}

/// a column-major Jacobian of P scalars in `Problem::jacobian`'s layout
fn layout<P: Precision<Native>>(n: usize, entry: impl Fn(usize, usize) -> P::Scalar) -> (Vec<Vec<f64>>, usize) {
    let ld = ld_for(n);
    let mut c = vec![vec![0.0; ld * n]; P::PARTS];
    for j in 0..n {
        for i in 0..n {
            let v = entry(i, j);
            for (q, cq) in c.iter_mut().enumerate() {
                cq[j * ld + i] = P::part(v, q);
            }
        }
    }
    (c, ld)
}

/// numbers as a JSON array, non-finite ones as null
fn arr(v: impl IntoIterator<Item = f64>) -> String {
    let items: Vec<String> = v
        .into_iter()
        .map(|x| if x.is_finite() { format!("{x:e}") } else { "null".into() })
        .collect();
    format!("[{}]", items.join(","))
}

fn pairs(v: &[(f64, f64)]) -> String {
    let items: Vec<String> = v.iter().map(|&(a, b)| arr([a, b])).collect();
    format!("[{}]", items.join(","))
}

/// the next of a linear congruential sequence, and its top 53 bits in [0, 1)
fn lcg(u: &mut u64) -> f64 {
    *u = u.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*u >> 11) as f64 / (1u64 << 53) as f64
}

/// F(z) = A z + z^3 - b with A = 4 I plus off-diagonal entries 2 sin(1 + 3 i + 5 j) / n^(1/2), and b planted so that
/// the root is z*_i = sin(i + 1) / 2 continued by pseudo-random lower parts, so that the products of the residual
/// round in P as for any root. The expansion has octo-double's 8 parts in every precision, P taking its leading ones.
struct Cubic<P: Precision<Native>> {
    n: usize,
    a: Vec<f64>,
    root: Vec<P::Scalar>,
    b: Vec<P::Scalar>,
}

impl<P: Precision<Native>> Cubic<P> {
    fn new(n: usize) -> Self {
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = if i == j {
                    4.0
                } else {
                    2.0 * ((1 + 3 * i + 5 * j) as f64).sin() / (n as f64).sqrt()
                };
            }
        }
        let mut u = 1u64;
        let root = (0..n)
            .map(|i| {
                let mut x = ((i + 1) as f64).sin() / 2.0;
                let mut p = Vec::with_capacity(8);
                for _ in 0..8 {
                    p.push(x);
                    // the next part about half an ulp of this one, of either sign, so that no bits are left out
                    let r = lcg(&mut u);
                    x *= f64::EPSILON * (0.25 + 0.5 * r) * if r < 0.5 { -1.0 } else { 1.0 };
                }
                P::new(&p[..P::PARTS])
            })
            .collect();
        let mut pr = Cubic {
            n,
            a,
            root,
            b: vec![P::ZERO; n],
        };
        pr.b = pr.f(&pr.root.clone());
        pr
    }

    /// A z + z^3 - b
    fn f(&self, z: &[P::Scalar]) -> Vec<P::Scalar> {
        (0..self.n)
            .map(|i| {
                let az = P::dot((0..self.n).map(|j| (s::<P>(self.a[i * self.n + j]), z[j])));
                az + z[i] * z[i] * z[i] - self.b[i]
            })
            .collect()
    }

    /// the start z* + 3/4 (1, -1, 1, ...)
    fn start(&self) -> Vec<Vec<f64>> {
        let d = s::<P>(0.75);
        let z: Vec<P::Scalar> = self
            .root
            .iter()
            .enumerate()
            .map(|(i, &r)| if i % 2 == 0 { r + d } else { r - d })
            .collect();
        to_parts::<P>(&z)
    }

    /// max_i |z_i - z*_i|
    fn error(&self, z: &[Vec<f64>]) -> f64 {
        let d: Vec<P::Scalar> = scalars::<P>(&parts(z))
            .iter()
            .zip(&self.root)
            .map(|(&x, &r)| x - r)
            .collect();
        max_abs::<P>(&d)
    }
}

impl<P: Precision<Native>> Problem<P> for Cubic<P> {
    fn len(&self) -> usize {
        self.n
    }
    fn residual(&self, z: &[&[f64]], _par: bool) -> Result<Vec<P::Scalar>, String> {
        Ok(self.f(&scalars::<P>(z)))
    }
    fn jacobian(&self, z: &[&[f64]], _par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
        let z = scalars::<P>(z);
        let three = s::<P>(3.0);
        Ok(layout::<P>(self.n, |i, j| {
            let a = s::<P>(self.a[i * self.n + j]);
            if i == j {
                a + three * z[i] * z[i]
            } else {
                a
            }
        }))
    }
}

fn seq(tol: f64, max_iter: usize) -> NewtonOpts {
    NewtonOpts {
        tol,
        max_iter,
        par: false,
        ..Default::default()
    }
}

/// Newton's method on the planted system from its start in P, past the floor: no tolerance, 12 steps at most
fn ladder_of<P: Precision<Native>>(n: usize) -> String {
    let pr = Cubic::<P>::new(n);
    let mut z = pr.start();
    let mut rep = newton::<P, _>(&pr, &mut z, &seq(0.0, 12)).unwrap();
    // max |F| at the start and after each step taken; a refused correction repeats the last
    rep.fnorm.truncate(rep.scale.len() + 1);
    arr(rep.fnorm)
}

fn ladder() -> String {
    let n = 32;
    format!(
        "{{\"n\":{n},\"double\":{},\"quad\":{},\"octo\":{}}}",
        ladder_of::<Double>(n),
        ladder_of::<Quad>(n),
        ladder_of::<Octo>(n)
    )
}

/// planted truth for a solve: converged, and the iterate the root to within `tol`
fn solved<P: Precision<Native>>(pr: &Cubic<P>, rep: &NewtonReport, z: &[Vec<f64>], tol: f64) -> bool {
    rep.converged && pr.error(z) <= tol
}

/// a problem whose residuals are logged: the seconds since `t0` and max |F|
struct Timed<'a, G> {
    g: &'a G,
    t0: Instant,
    log: RefCell<Vec<(f64, f64)>>,
}

impl<P: Precision<Native>, G: Problem<P>> Problem<P> for Timed<'_, G> {
    fn len(&self) -> usize {
        self.g.len()
    }
    fn residual(&self, z: &[&[f64]], par: bool) -> Result<Vec<P::Scalar>, String> {
        let f = self.g.residual(z, par)?;
        self.log
            .borrow_mut()
            .push((self.t0.elapsed().as_secs_f64(), max_abs::<P>(&f)));
        Ok(f)
    }
    fn jacobian(&self, z: &[&[f64]], par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
        self.g.jacobian(z, par)
    }
}

/// the methods timed, by name
const METHODS: [&str; 5] = ["newton", "lowered-dd", "lowered-f64", "quasi", "quasi-dd"];

/// method `m` on `pr` from z, with the factorizations it formed
fn run<P: Precision<Native>, Pr: Problem<P>>(
    pr: &Pr,
    z: &mut Vec<Vec<f64>>,
    m: &str,
    opts: &NewtonOpts,
) -> (NewtonReport, usize) {
    let lus = Cell::new(0usize);
    let count = || lus.set(lus.get() + 1);
    let ex = |z: &[&[f64]]| {
        count();
        exact::<P, _>(pr, z, false)
    };
    let low = |to: Lower| {
        move |z: &[&[f64]]| {
            count();
            lowered::<P, _>(pr, z, to, false)
        }
    };
    let q = QuasiOpts::default();
    let rep = match m {
        "newton" => newton_with::<P, _, _>(pr, z, opts, ex),
        "lowered-dd" => newton_with::<P, _, _>(pr, z, opts, low(Lower::Double)),
        "lowered-f64" => newton_with::<P, _, _>(pr, z, opts, low(Lower::F64)),
        "quasi" => newton_quasi::<P, _, _>(pr, z, opts, &q, ex).map(|(rep, _)| rep),
        "quasi-dd" => newton_quasi::<P, _, _>(pr, z, opts, &q, low(Lower::Double)).map(|(rep, _)| rep),
        "shamanskii" => newton_with::<P, _, _>(pr, z, opts, reuse(3, ex)),
        _ => unreachable!("unknown method {m}"),
    }
    .unwrap_or_else(|e| panic!("{m}: {e}"));
    (rep, lus.get())
}

fn time() -> String {
    let (n, reps, tol) = (128, 5, 1e-120);
    let pr = Cubic::<Octo>::new(n);
    let rows: Vec<String> = METHODS
        .iter()
        .map(|m| {
            let mut runs = Vec::new();
            for _ in 0..reps {
                let tp = Timed {
                    g: &pr,
                    t0: Instant::now(),
                    log: RefCell::new(Vec::new()),
                };
                let mut z = pr.start();
                let (rep, lus) = run::<Octo, _>(&tp, &mut z, m, &seq(tol, 60));
                let secs = tp.t0.elapsed().as_secs_f64();
                assert!(
                    solved(&pr, &rep, &z, tol),
                    "{m}: converged {}, error {:e}",
                    rep.converged,
                    pr.error(&z)
                );
                runs.push((secs, rep.scale.len(), lus, tp.log.into_inner()));
            }
            let mut order: Vec<usize> = (0..reps).collect();
            order.sort_by(|&a, &b| runs[a].0.total_cmp(&runs[b].0));
            let (_, steps, lus, log) = &runs[order[reps / 2]];
            format!(
                "\"{m}\":{{\"secs\":{},\"steps\":{steps},\"lus\":{lus},\"trace\":{}}}",
                arr(runs.iter().map(|r| r.0)),
                pairs(log)
            )
        })
        .collect();
    format!(
        "{{\"n\":{n},\"reps\":{reps},\"tol\":{tol:e},\"methods\":{{{}}}}}",
        rows.join(",")
    )
}

/// F(z, p) = B G(z, p) with G_1 = z_1^3 - 3 z_1 - p, G_2 = z_2 - z_1^2 / 2 - p / 4 and B = [[2, 0.3], [-0.4, 2]]: the
/// branch p = z_1^3 - 3 z_1 has folds at z_1 = -1, p = 2 and z_1 = 1, p = -2
struct S;

const B: [[f64; 2]; 2] = [[2.0, 0.3], [-0.4, 2.0]];

fn mix<P: Precision<Native>>(g: [P::Scalar; 2]) -> Vec<P::Scalar> {
    B.iter()
        .map(|row| s::<P>(row[0]) * g[0] + s::<P>(row[1]) * g[1])
        .collect()
}

impl<P: Precision<Native>> Family<P> for S {
    fn len(&self) -> usize {
        2
    }
    fn residual_at(&self, z: &[&[f64]], p: P::Scalar, _par: bool) -> Result<Vec<P::Scalar>, String> {
        let z = scalars::<P>(z);
        let g1 = z[0] * z[0] * z[0] - s::<P>(3.0) * z[0] - p;
        let g2 = z[1] - z[0] * z[0] * s::<P>(0.5) - p * s::<P>(0.25);
        Ok(mix::<P>([g1, g2]))
    }
    fn jacobian_at(&self, z: &[&[f64]], _p: P::Scalar, _par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
        let z = scalars::<P>(z);
        // dG/dz by columns
        let cols = [
            mix::<P>([s::<P>(3.0) * z[0] * z[0] - s::<P>(3.0), -z[0]]),
            mix::<P>([P::ZERO, P::ONE]),
        ];
        Ok(layout::<P>(2, |i, j| cols[j][i]))
    }
    fn dp_at(&self, _z: &[&[f64]], _p: P::Scalar, _par: bool) -> Result<Vec<P::Scalar>, String> {
        Ok(mix::<P>([-P::ONE, -s::<P>(0.25)]))
    }
}

fn ex(b: &Bordered<'_, Double, S>, w: &[&[f64]]) -> Result<Exact<Double>, String> {
    exact::<Double, _>(b, w, false)
}

/// the branch's start: z_1 = -9/4, exact, and so are p and z_2 from it
fn fold_start() -> (Vec<Vec<f64>>, DD) {
    let z1 = DD::from_f64(-2.25);
    let p = z1 * z1 * z1 - DD::from_f64(3.0) * z1;
    let z2 = z1 * z1 * DD::from_f64(0.5) + p * DD::from_f64(0.25);
    (to_parts::<Double>(&[z1, z2]), p)
}

fn arc_opts() -> ArcOpts {
    ArcOpts {
        ds: 0.25,
        ds_max: 0.6,
        max_points: 200,
        newton: seq(1e-30, 20),
        ..Default::default()
    }
}

/// the branch from the start along increasing p until z_1 > 9/4, past both folds
fn branch() -> Path<Double> {
    let (z0, p) = fold_start();
    let a = start::<Double, _, _>(&S, &parts(&z0), p, 1.0, false, ex).unwrap();
    continuation::<Double, _, _>(&S, a, &arc_opts(), ex, |pt| pt.z[0][0] > 2.25).unwrap()
}

/// the fold between point i of the path and the next
fn locate(path: &Path<Double>, i: usize) -> Point<Double> {
    let opts = arc_opts();
    locate_fold::<Double, _, _>(&S, &path.points[i], path.ds[i], &FoldOpts::default(), &opts.newton, ex)
        .unwrap()
        .0
}

/// |p - p*| and |z_1 - z_1*| of a located fold from the nearer planted fold, (2, -1) or (-2, 1), in DD
fn fold_errors(pt: &Point<Double>) -> (f64, f64) {
    let z1 = DD::new(pt.z[0][0], pt.z[1][0]);
    let sign = if pt.p.hi > 0.0 { 1.0 } else { -1.0 };
    let ep = (pt.p - DD::from_f64(2.0 * sign)).hi.abs();
    let ez = (z1 + DD::from_f64(sign)).hi.abs();
    (ep, ez)
}

/// planted truth for a fold: |t_p| <= 1e-15 leaves z_1 within about 1e-15 of it and p within about 1e-30
fn fold_ok((ep, ez): (f64, f64)) -> bool {
    ep < 1e-28 && ez < 1e-13
}

/// planted truth for the branch: followed past both folds to z_1 > 9/4, with one fold located at each, p = 2 and
/// p = -2, in order
fn branch_ok(path: &Path<Double>, folds: &[Point<Double>]) -> bool {
    let last = &path.points[path.points.len() - 1];
    let signs: Vec<bool> = folds.iter().map(|pt| pt.p.hi > 0.0).collect();
    !path.stalled && last.z[0][0] > 2.25 && signs == [true, false]
}

fn fold() -> String {
    let (z0, p) = fold_start();
    let path = branch();
    let pts_at: Vec<Point<Double>> = path.folds.iter().map(|&i| locate(&path, i)).collect();
    assert!(
        branch_ok(&path, &pts_at),
        "{} points, folds {:?}",
        path.points.len(),
        path.folds
    );
    let pts: Vec<(f64, f64)> = path.points.iter().map(|pt| (pt.p.hi, pt.z[0][0])).collect();
    // each fold's (p, z_1) and its errors
    let located: Vec<((f64, f64), (f64, f64))> = pts_at
        .iter()
        .map(|pt| {
            let e = fold_errors(pt);
            assert!(fold_ok(e), "fold at {:?} {:?}", pt.p, pt.z);
            ((pt.p.hi, pt.z[0][0]), e)
        })
        .collect();
    let folds: Vec<(f64, f64)> = located.iter().map(|l| l.0).collect();
    let errs: Vec<(f64, f64)> = located.iter().map(|l| l.1).collect();
    // continuation in p: Newton at p = p0 + 0.4 k from the last solution
    let mut natural = Vec::new();
    let mut z = z0.clone();
    for k in 0..24 {
        let pk = p + DD::from_f64(0.4 * k as f64);
        let at = AtParam::<Double, S> { fam: &S, p: pk };
        let mut zk = z.clone();
        match newton::<Double, _>(&at, &mut zk, &seq(1e-30, 40)) {
            Ok(rep) if rep.converged => {
                natural.push((pk.hi, zk[0][0]));
                z = zk;
            }
            _ => natural.push((pk.hi, f64::NAN)),
        }
    }
    format!(
        "{{\"points\":{},\"ds\":{},\"folds\":{},\"fold_errors\":{},\"refused\":{},\"natural\":{}}}",
        pairs(&pts),
        arr(path.ds.iter().copied()),
        pairs(&folds),
        pairs(&errs),
        path.refused,
        pairs(&natural)
    )
}

/// z^5 = 1 in the real unknowns (x, y) of z = x + i y
struct Roots5;

impl<P: Precision<Native>> Problem<P> for Roots5 {
    fn len(&self) -> usize {
        2
    }
    fn residual(&self, z: &[&[f64]], _par: bool) -> Result<Vec<P::Scalar>, String> {
        let z = scalars::<P>(z);
        let (mut re, mut im) = (P::ONE, P::ZERO);
        for _ in 0..5 {
            (re, im) = (re * z[0] - im * z[1], re * z[1] + im * z[0]);
        }
        Ok(vec![re - P::ONE, im])
    }
    fn jacobian(&self, z: &[&[f64]], _par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
        let z = scalars::<P>(z);
        // 5 z^4 = a + i b; by Cauchy-Riemann J = [[a, -b], [b, a]]
        let (mut re, mut im) = (s::<P>(5.0), P::ZERO);
        for _ in 0..4 {
            (re, im) = (re * z[0] - im * z[1], re * z[1] + im * z[0]);
        }
        Ok(layout::<P>(2, |i, j| match (i, j) {
            (0, 1) => -im,
            (1, 0) => im,
            _ => re,
        }))
    }
}

/// a problem whose Jacobian evaluations, one at each Newton iterate, are logged
struct Logged<'a, G> {
    g: &'a G,
    at: RefCell<Vec<(f64, f64)>>,
}

impl<P: Precision<Native>, G: Problem<P>> Problem<P> for Logged<'_, G> {
    fn len(&self) -> usize {
        self.g.len()
    }
    fn residual(&self, z: &[&[f64]], par: bool) -> Result<Vec<P::Scalar>, String> {
        self.g.residual(z, par)
    }
    fn jacobian(&self, z: &[&[f64]], par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
        self.at.borrow_mut().push((z[0][0], z[0][1]));
        self.g.jacobian(z, par)
    }
}

/// The error of the root (x, y) from the nearest fifth root of unity e^(2 pi i k / 5), in DD, k chosen by the angle of
/// (x, y): cos 72 deg = (5^(1/2) - 1) / 4, sin 72 deg = ((5 + 5^(1/2)) / 8)^(1/2), cos 144 deg = -(5^(1/2) + 1) / 4
/// and sin 144 deg = ((5 - 5^(1/2)) / 8)^(1/2)
fn root_error(x: DD, y: DD) -> f64 {
    let k = (y.hi.atan2(x.hi) / (std::f64::consts::TAU / 5.0)).round() as i32;
    let r5 = DD::from_f64(5.0).sqrt();
    let quarter = DD::from_f64(0.25);
    let eighth = DD::from_f64(0.125);
    let (c, s) = match k.abs() {
        0 => (DD::ONE, DD::ZERO),
        1 => ((r5 - DD::ONE) * quarter, ((DD::from_f64(5.0) + r5) * eighth).sqrt()),
        _ => (-(r5 + DD::ONE) * quarter, ((DD::from_f64(5.0) - r5) * eighth).sqrt()),
    };
    let s = if k < 0 { -s } else { s };
    (x - c).hi.abs().max((y - s).hi.abs())
}

/// planted truth for a root: |F| <= 1e-30 with |F'| = 5 leaves it within about 2e-31, the references within 1e-32
fn root_ok(e: f64) -> bool {
    e < 1e-30
}

/// the deflated searches from `START`: the roots found, and each search's iterates
fn searches() -> (Found, Vec<Vec<(f64, f64)>>) {
    let z0 = vec![vec![START.0, START.1], vec![0.0, 0.0]];
    // full steps, which leave the first root's basin (the deflation module's documentation)
    let opts = NewtonOpts {
        damping: Damping::Residual,
        max_halvings: 0,
        ..seq(1e-30, 60)
    };
    let mut paths = Vec::new();
    let found = roots_from::<Double, _>(&Roots5, &z0, 5, &DeflationOpts::default(), |g, z| {
        let lg = Logged {
            g,
            at: RefCell::new(Vec::new()),
        };
        let rep = newton::<Double, _>(&lg, z, &opts);
        let mut at = lg.at.into_inner();
        at.push((z[0][0], z[0][1]));
        paths.push(at);
        rep
    })
    .unwrap();
    (found, paths)
}

/// the start of every search
const START: C64 = (-0.487, -0.393);

fn deflation() -> String {
    let (found, paths) = searches();
    assert!(found.ended.is_none(), "{:?}", found.ended);
    let mut out = Vec::new();
    for (r, path) in found.roots.iter().zip(&paths) {
        let (x, y) = (DD::new(r[0][0], r[1][0]), DD::new(r[0][1], r[1][1]));
        assert!(root_ok(root_error(x, y)), "root {x:?} {y:?}");
        out.push(format!("{{\"path\":{},\"root\":{}}}", pairs(path), arr([x.hi, y.hi])));
    }
    format!(
        "{{\"start\":{},\"searches\":[{}]}}",
        arr([START.0, START.1]),
        out.join(",")
    )
}

/// the zeros of g: four inside the square [-1, 1]^2, a close pair near its right edge among them, and two outside, one
/// near the same edge
const ZEROS: [C64; 6] = [
    (0.86, 0.3),
    (0.86, 0.33),
    (-0.45, -0.5),
    (-0.3, 0.55),
    (1.1, -0.55),
    (-0.35, 1.4),
];

const SQUARE: [C64; 4] = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];

/// the winding of g(lambda) = prod (lambda - z_k) over `zeros` along the square
fn wind(zeros: &[C64]) -> Winding {
    let g = |l: C64| -> Result<(f64, f64), String> {
        let (mut ln, mut arg) = (0.0, 0.0);
        for &z in zeros {
            let d = (l.0 - z.0, l.1 - z.1);
            ln += d.0.hypot(d.1).ln();
            arg += d.1.atan2(d.0);
        }
        Ok((ln, multisolver::winding::wrap(arg)))
    };
    winding(g, &SQUARE, &WindOpts::default()).unwrap()
}

/// planted truth for the count: resolved, and the four zeros inside, the close pair among them
fn count_ok(w: &Winding) -> bool {
    w.resolved && w.count() == 4
}

fn winding_data() -> String {
    let w = wind(&ZEROS);
    assert!(count_ok(&w), "{} turns", w.turns);
    let edges: Vec<String> = w
        .edges
        .iter()
        .map(|e| {
            let pts: Vec<(f64, f64)> =
                e.t.iter()
                    .map(|&t| (e.from.0 + t * (e.to.0 - e.from.0), e.from.1 + t * (e.to.1 - e.from.1)))
                    .collect();
            format!("{{\"pts\":{},\"arg\":{}}}", pairs(&pts), arr(e.val.iter().map(|v| v.1)))
        })
        .collect();
    let m = w.moment();
    format!(
        "{{\"vertices\":{},\"zeros\":{},\"edges\":[{}],\"turns\":{:e},\"evals\":{},\"moment\":{}}}",
        pairs(&SQUARE),
        pairs(&ZEROS),
        edges.join(","),
        w.turns,
        w.evals,
        arr([m.0, m.1])
    )
}

fn main() {
    println!(
        "{{\"ladder\":{},\"time\":{},\"fold\":{},\"deflation\":{},\"winding\":{}}}",
        ladder(),
        time(),
        fold(),
        deflation(),
        winding_data()
    );
}

/// The controls: each planted-truth check passes the example's result and fails it moved
#[cfg(test)]
mod tests {
    use super::*;

    /// Pass, written before the first run: Newton's method in double-double on the planted system of n = 8 passes the
    /// solve check with tolerance 1e-30, and fails it with the first unknown moved by 1e-12.
    #[test]
    fn solve_check_fails_a_moved_iterate() {
        let pr = Cubic::<Double>::new(8);
        let mut z = pr.start();
        let rep = newton::<Double, _>(&pr, &mut z, &seq(1e-30, 40)).unwrap();
        assert!(solved(&pr, &rep, &z, 1e-30), "error {:e}", pr.error(&z));
        z[0][0] += 1e-12;
        assert!(!solved(&pr, &rep, &z, 1e-30));
    }

    /// Pass, written before the first run: both folds the branch finds pass the fold check, and each fails it with p
    /// moved by 1e-14.
    #[test]
    fn fold_check_fails_a_moved_fold() {
        let path = branch();
        assert_eq!(path.folds.len(), 2);
        for &i in &path.folds {
            let pt = locate(&path, i);
            assert!(fold_ok(fold_errors(&pt)), "{:?}", fold_errors(&pt));
            let mut moved = pt.clone();
            moved.p = moved.p + DD::from_f64(1e-14);
            assert!(!fold_ok(fold_errors(&moved)));
        }
    }

    /// Pass, written before the first run: the branch with its two located folds passes the branch check, and fails it
    /// cut after its first fold, with that fold alone, and with the folds in the other order.
    #[test]
    fn branch_check_fails_a_cut_branch() {
        let path = branch();
        let folds: Vec<Point<Double>> = path.folds.iter().map(|&i| locate(&path, i)).collect();
        assert!(branch_ok(&path, &folds));
        let mut cut = path.clone();
        cut.points.truncate(path.folds[1] + 1);
        assert!(!branch_ok(&cut, &folds[..1]));
        assert!(!branch_ok(&path, &[folds[1].clone(), folds[0].clone()]));
    }

    /// Pass, written before the first run: the five roots the searches find pass the root check, and each fails it
    /// with its real part moved by 1e-20, in its low part alone.
    #[test]
    fn root_check_fails_a_moved_root() {
        let (found, _) = searches();
        assert_eq!(found.roots.len(), 5);
        for r in &found.roots {
            let (x, y) = (DD::new(r[0][0], r[1][0]), DD::new(r[0][1], r[1][1]));
            assert!(root_ok(root_error(x, y)));
            let moved = x + DD::from_f64(1e-20);
            assert!(moved.hi == x.hi && !root_ok(root_error(moved, y)));
        }
    }

    /// Pass, written before the first run: the zeros pass the count check, and fail it with one of the close pair moved
    /// outside the square, to 1.86 + 0.3i.
    #[test]
    fn count_check_fails_a_zero_moved_outside() {
        assert!(count_ok(&wind(&ZEROS)));
        let mut moved = ZEROS;
        moved[0] = (1.86, 0.3);
        assert!(!count_ok(&wind(&moved)));
    }
}
