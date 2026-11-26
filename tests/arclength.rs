//! Pseudo-arclength continuation on a planted fold: a branch of five unknowns whose first two equations are a circle
//! in (z_1, p), so that the branch turns back at p = 1, with the residual's rows mixed by a dense matrix.

use multisolver::arclength::{
    continuation, locate_fold, start, tangent, ArcOpts, AtParam, Bordered, FoldOpts, Path, Point,
};
use multisolver::multidouble::linalg::ld_for;
use multisolver::newton::{exact, lowered, newton, Damping, Factored, Lower, NewtonOpts};
use multisolver::prec::{nan_max, parts};
use multisolver::{Double, Family, Native, Precision, Quad, DD};

const N: usize = 5;

fn s<P: Precision<Native>>(x: f64) -> P::Scalar {
    P::from_dd(DD::from_f64(x))
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

/// F(z, p) = B G(z, p) with G_1 = z_1^2 + p^2 - 1 and G_i = z_i - (z_(i-1)^2 + p) / 2 for i = 2..5, and B the dense
/// matrix 2 I plus off-diagonal entries 0.3 sin(1 + 3 i + 5 j), diagonally dominant. The branch through (z_1, p) =
/// (1, 0) is z_1 = +-(1 - p^2)^(1/2), z_i = (z_(i-1)^2 + p) / 2: a circle in (z_1, p) with simple folds at p = +-1,
/// z_1 = 0, where z = (0, 0.5, 0.625, 0.6953125, 0.741729736328125). Row 1 of dF/dp is scaled by 1 + `dp_err`, 0
/// for the exact one.
struct Fold<P: Precision<Native>> {
    b: Vec<Vec<P::Scalar>>,
    dp_err: f64,
}

impl<P: Precision<Native>> Fold<P> {
    fn new() -> Self {
        let b = (0..N)
            .map(|i| {
                (0..N)
                    .map(|j| {
                        s::<P>(if i == j {
                            2.0
                        } else {
                            0.3 * ((1 + 3 * i + 5 * j) as f64).sin()
                        })
                    })
                    .collect()
            })
            .collect();
        Fold { b, dp_err: 0.0 }
    }

    /// G(z, p), the planted equations
    fn g(&self, z: &[P::Scalar], p: P::Scalar) -> Vec<P::Scalar> {
        let half = s::<P>(0.5);
        (0..N)
            .map(|i| {
                if i == 0 {
                    z[0] * z[0] + p * p - P::ONE
                } else {
                    z[i] - (z[i - 1] * z[i - 1] + p) * half
                }
            })
            .collect()
    }

    /// B v
    fn mix(&self, v: &[P::Scalar]) -> Vec<P::Scalar> {
        self.b
            .iter()
            .map(|row| P::dot(row.iter().copied().zip(v.iter().copied())))
            .collect()
    }
}

impl<P: Precision<Native>> Family<P> for Fold<P> {
    fn len(&self) -> usize {
        N
    }
    fn residual_at(&self, z: &[&[f64]], p: P::Scalar, _par: bool) -> Result<Vec<P::Scalar>, String> {
        Ok(self.mix(&self.g(&scalars::<P>(z), p)))
    }
    fn jacobian_at(&self, z: &[&[f64]], _p: P::Scalar, _par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
        let z = scalars::<P>(z);
        // dG/dz by columns: column 0 has 2 z_1 in row 0 and -z_1 in row 1; column j >= 1 has 1 in row j and -z_j in
        // row j + 1
        let mut gz = vec![vec![P::ZERO; N]; N];
        for j in 0..N {
            gz[j][j] = if j == 0 { s::<P>(2.0) * z[0] } else { P::ONE };
            if j + 1 < N {
                gz[j][j + 1] = -z[j];
            }
        }
        let ld = ld_for(N);
        let mut c = vec![vec![0.0; ld * N]; P::PARTS];
        for (j, col) in gz.iter().enumerate() {
            for (i, &v) in self.mix(col).iter().enumerate() {
                for (q, cq) in c.iter_mut().enumerate() {
                    cq[j * ld + i] = P::part(v, q);
                }
            }
        }
        Ok((c, ld))
    }
    fn dp_at(&self, _z: &[&[f64]], p: P::Scalar, _par: bool) -> Result<Vec<P::Scalar>, String> {
        let gp: Vec<P::Scalar> = (0..N)
            .map(|i| if i == 0 { s::<P>(2.0) * p } else { -s::<P>(0.5) })
            .collect();
        let mut fp = self.mix(&gp);
        fp[0] = fp[0] * P::from_dd(DD::new(1.0, self.dp_err));
        Ok(fp)
    }
}

/// the start of the branch: z_1 = 1 at p = 0, so z = (1, 0.5, 0.125, 0.0078125, 0.000030517578125)
fn z_start<P: Precision<Native>>() -> Vec<Vec<f64>> {
    let mut z = vec![P::ONE];
    for i in 1..N {
        z.push(z[i - 1] * z[i - 1] * s::<P>(0.5));
    }
    to_parts::<P>(&z)
}

/// the exact factors of a bordered matrix
fn ex<P: Precision<Native>>(
    b: &Bordered<'_, P, Fold<P>>,
    w: &[&[f64]],
) -> Result<multisolver::newton::Exact<P>, String> {
    exact::<P, _>(b, w, false)
}

/// the corrector's settings: tolerance `tol`, sequential
fn corrector(tol: f64) -> NewtonOpts {
    NewtonOpts {
        tol,
        par: false,
        ..Default::default()
    }
}

/// The branch from the start along increasing p, until a point with z_1 < -1/2 (past the fold, on the circle's other
/// half) or 200 points
fn branch<P: Precision<Native>>(fam: &Fold<P>, tol: f64) -> Path<P> {
    let z0 = z_start::<P>();
    let a = start::<P, _, _>(fam, &parts(&z0), P::ZERO, 1.0, false, ex::<P>).unwrap();
    let opts = ArcOpts {
        max_points: 200,
        newton: corrector(tol),
        ..Default::default()
    };
    continuation::<P, _, _>(fam, a, &opts, ex::<P>, |pt| pt.z[0][0] < -0.5).unwrap()
}

/// max |G| at a point, the planted equations' residual
fn g_err<P: Precision<Native>>(fam: &Fold<P>, z: &[Vec<f64>], p: P::Scalar) -> f64 {
    fam.g(&scalars::<P>(&parts(z)), p)
        .iter()
        .map(|&v| P::to_dd(v).hi.abs())
        .fold(0.0, nan_max)
}

/// the checks of a path: [the last point is past the fold (z_1 < -1/2), one fold found between a point with z_1 > 0
/// and the next with z_1 < 0, every point on the planted branch to max |G| <= gtol]
fn path_checks<P: Precision<Native>>(fam: &Fold<P>, path: &Path<P>, gtol: f64) -> [bool; 3] {
    let z1 = |i: usize| path.points[i].z[0][0];
    let last = path.points.len() - 1;
    let fold_ok = path.folds.len() == 1 && z1(path.folds[0]) > 0.0 && z1(path.folds[0] + 1) < 0.0;
    let on = path
        .points
        .iter()
        .map(|pt| g_err::<P>(fam, &pt.z, pt.p))
        .fold(0.0, nan_max);
    [z1(last) < -0.5, fold_ok, on <= gtol]
}

/// Pass, written before the first run: in double-double, with the corrector's tolerance max |F| <= 1e-30, the
/// continuation from (z_1, p) = (1, 0) along increasing p passes the fold at p = 1 and reaches z_1 < -1/2 within 200
/// points, finds exactly one fold, between a point with z_1 > 0 and the next with z_1 < 0, and keeps every point on
/// the planted branch to max |G| <= 1e-27 (B^-1 has norm at most 1.25). In quad-double the same with the tolerance
/// 1e-62 and max |G| <= 1e-59. The controls: continuation in p (Newton at fixed p = 0.1 k from the last solution)
/// stops at the latest at p = 1.1, where the circle has no point, so it never reaches z_1 < -1/2; the path cut before its fold has no fold and fails the first two checks; and the path with one
/// point's z_3 moved by 1e-20 fails the third.
#[test]
fn continuation_passes_a_fold() {
    let fam = Fold::<Double>::new();
    let path = branch::<Double>(&fam, 1e-30);
    let last = &path.points[path.points.len() - 1];
    println!(
        "double-double: {} points, {} refused, folds after {:?}, last (z_1, p) = ({:.6}, {:.6})",
        path.points.len(),
        path.refused,
        path.folds,
        last.z[0][0],
        last.p.hi
    );
    assert_eq!(path_checks(&fam, &path, 1e-27), [true; 3]);
    let famq = Fold::<Quad>::new();
    let pq = branch::<Quad>(&famq, 1e-62);
    println!("quad-double: {} points, folds after {:?}", pq.points.len(), pq.folds);
    assert_eq!(path_checks(&famq, &pq, 1e-59), [true; 3]);

    // the controls
    let mut z = z_start::<Double>();
    let mut reached = f64::INFINITY;
    let mut stopped_at = None;
    for k in 1..=15 {
        let p = 0.1 * k as f64;
        let pr = AtParam {
            fam: &fam,
            p: DD::from_f64(p),
        };
        let mut zt = z.clone();
        let ok = newton::<Double, _>(&pr, &mut zt, &corrector(1e-30)).is_ok_and(|r| r.converged);
        if !ok {
            stopped_at = Some(p);
            break;
        }
        z = zt;
        reached = z[0][0];
    }
    println!("control, continuation in p: stopped at p = {stopped_at:?}, last z_1 {reached:.6}");
    assert!(
        stopped_at.is_some_and(|p| p <= 1.1 + 1e-12) && reached >= -0.5,
        "control did not fire"
    );
    let mut cut = path.clone();
    cut.points.truncate(path.folds[0] + 1);
    cut.folds.clear();
    assert_eq!(
        path_checks(&fam, &cut, 1e-27),
        [false, false, true],
        "control, cut path"
    );
    let mut moved = path.clone();
    let i = moved.points.len() / 2;
    let z3 = DD::new(moved.points[i].z[0][2], moved.points[i].z[1][2]) + DD::from_f64(1e-20);
    (moved.points[i].z[0][2], moved.points[i].z[1][2]) = (z3.hi, z3.lo);
    assert_eq!(
        path_checks(&fam, &moved, 1e-27),
        [true, true, false],
        "control, moved point"
    );
}

/// the checks of a located fold: [|p - 1| <= ptol, |z_1| <= ztol, z_2..z_5 their values at the fold within ptol]
fn fold_checks<P: Precision<Native>>(z: &[Vec<f64>], p: P::Scalar, ptol: f64, ztol: f64) -> [bool; 3] {
    let zs = scalars::<P>(&parts(z));
    let want = [0.5, 0.625, 0.6953125, 0.741729736328125];
    let rest = (1..N)
        .map(|i| P::to_dd(zs[i] - s::<P>(want[i - 1])).hi.abs())
        .fold(0.0, nan_max);
    [
        P::to_dd(p - P::ONE).hi.abs() <= ptol,
        P::to_dd(zs[0]).hi.abs() <= ztol,
        rest <= ptol,
    ]
}

/// the fold of the branch located in P, with the corrector's tolerance `tol` and |t_p| <= `tp_tol`: (the point, the
/// corrections, the first point of the fold's step)
fn located<P: Precision<Native>>(tol: f64, tp_tol: f64) -> (Point<P>, usize, Point<P>) {
    let fam = Fold::<P>::new();
    let path = branch::<P>(&fam, tol);
    let i = path.folds[0];
    let opts = FoldOpts {
        tol: tp_tol,
        ..Default::default()
    };
    let (pt, it) = locate_fold::<P, _, _>(&fam, &path.points[i], path.ds[i], &opts, &corrector(tol), ex::<P>).unwrap();
    (pt, it, path.points[i].clone())
}

/// Pass, written before the first run: between the two points that bracket the fold, regula falsi on the step
/// length locates it in double-double, with the corrector's tolerance 1e-30 and |t_p| <= 1e-15, to |p - 1| <= 1e-29,
/// |z_1| <= 1e-14 and z_2..z_5 within 1e-29 of their values at the fold (p and z_2..z_5 move by about z_1^2 there);
/// in quad-double, with 1e-62 and |t_p| <= 1e-31, to 1e-61, 1e-30 and 1e-61. The control: the bracket's first point,
/// short of the fold, fails the checks of p and z_1.
#[test]
fn fold_is_located_to_the_precision() {
    let (pt, it, a) = located::<Double>(1e-30, 1e-15);
    let (z, p, za, pa) = (pt.z, pt.p, a.z, a.p);
    println!(
        "double-double: {it} corrections, p - 1 = {:.3e}, z_1 = {:.3e}",
        (p - DD::ONE).hi,
        z[0][0]
    );
    assert_eq!(fold_checks::<Double>(&z, p, 1e-29, 1e-14), [true; 3]);
    let fa = fold_checks::<Double>(&za, pa, 1e-29, 1e-14);
    println!(
        "control, the bracket's first point: p = {:.6}, z_1 = {:.6}",
        pa.hi, za[0][0]
    );
    assert!(!fa[0] && !fa[1], "control did not fire");
    let (ptq, itq, _) = located::<Quad>(1e-62, 1e-31);
    let (zq, pq) = (ptq.z, ptq.p);
    let dq = <Quad as Precision<Native>>::to_dd(pq - <Quad as Precision<Native>>::ONE).hi;
    println!(
        "quad-double: {itq} corrections, p - 1 = {dq:.3e}, z_1 = {:.3e}",
        zq[0][0]
    );
    assert_eq!(fold_checks::<Quad>(&zq, pq, 1e-61, 1e-30), [true; 3]);
}

/// t_z / t_p against dz/dp at the start, max over the entries, and t_p, |t| - 1
fn tangent_errors<P: Precision<Native>>(fam: &Fold<P>, dir: f64) -> (f64, f64, f64) {
    let z0 = z_start::<P>();
    let n1 = N + 1;
    let mut d = vec![P::ZERO; n1];
    d[N] = s::<P>(dir);
    let t = tangent::<P, _, _>(fam, &parts(&z0), P::ZERO, &d, false, ex::<P>).unwrap();
    errors_of::<P>(&t)
}

/// t_z / t_p of the tangent t at the start against dz/dp, max over the entries, and t_p, |t| - 1
fn errors_of<P: Precision<Native>>(t: &[P::Scalar]) -> (f64, f64, f64) {
    let z0 = z_start::<P>();
    // dz/dp at p = 0 on the branch: dz_1/dp = -p / z_1 = 0, dz_i/dp = (2 z_(i-1) dz_(i-1)/dp + 1) / 2
    let zs = scalars::<P>(&parts(&z0));
    let mut dz = vec![P::ZERO];
    for i in 1..N {
        dz.push((s::<P>(2.0) * zs[i - 1] * dz[i - 1] + P::ONE) * s::<P>(0.5));
    }
    let inv = P::from_dd(DD::ONE / P::to_dd(t[N]));
    // t_z / t_p in P: one Newton step on the f64-accurate reciprocal
    let inv = inv + inv * (P::ONE - t[N] * inv);
    let inv = inv + inv * (P::ONE - t[N] * inv);
    let err = (0..N).map(|i| P::to_dd(t[i] * inv - dz[i]).hi.abs()).fold(0.0, nan_max);
    let norm = t.iter().map(|&v| P::to_dd(v).hi.powi(2)).sum::<f64>().sqrt();
    (err, P::to_dd(t[N]).hi, norm - 1.0)
}

/// Pass, written before the first run: the tangent at the start has the direction (dz/dp, 1) of the branch, t_z / t_p
/// within 1e-30 of dz/dp in double-double and 1e-62 in quad-double, unit length to 1e-15, and t_p > 0 oriented by +1,
/// t_p < 0 by -1. The control: with row 1 of dF/dp off by 1e-20 relative, t_z / t_p misses by more than 1e-30.
#[test]
fn tangent_is_the_branch_direction() {
    let fam = Fold::<Double>::new();
    let (e, tp, dn) = tangent_errors::<Double>(&fam, 1.0);
    println!("double-double: t_z / t_p off by {e:.3e}, t_p {tp:.6}, |t| - 1 = {dn:.3e}");
    assert!(e <= 1e-30 && tp > 0.0 && dn.abs() <= 1e-15);
    let (_, tm, _) = tangent_errors::<Double>(&fam, -1.0);
    assert!(tm < 0.0, "oriented by -1: t_p {tm}");
    let (eq, tq, dq) = tangent_errors::<Quad>(&Fold::<Quad>::new(), 1.0);
    println!("quad-double: t_z / t_p off by {eq:.3e}, t_p {tq:.6}, |t| - 1 = {dq:.3e}");
    assert!(eq <= 1e-62 && tq > 0.0 && dq.abs() <= 1e-15);
    let bad = Fold::<Double> {
        dp_err: 1e-20,
        ..Fold::<Double>::new()
    };
    let (eb, _, _) = tangent_errors::<Double>(&bad, 1.0);
    println!("control, dF/dp off by 1e-20: t_z / t_p off by {eb:.3e}");
    assert!(eb > 1e-30, "control did not fire");
}

/// the start's w = (z, 0) as part arrays
fn w_start<P: Precision<Native>>() -> Vec<Vec<f64>> {
    z_start::<P>()
        .into_iter()
        .map(|q| q.into_iter().chain([0.0]).collect())
        .collect()
}

/// the tangent at the start oriented along increasing p, the bordered matrix factored in f64
fn tangent_f64<P: Precision<Native>>(fam: &Fold<P>) -> Vec<P::Scalar> {
    let mut d = vec![P::ZERO; N + 1];
    d[N] = P::ONE;
    let f64_factors = |b: &Bordered<'_, P, Fold<P>>, w: &[&[f64]]| lowered::<P, _>(b, w, Lower::F64, false);
    tangent::<P, _, _>(fam, &parts(&z_start::<P>()), P::ZERO, &d, false, f64_factors).unwrap()
}

/// Pass, written before the first run: with the bordered matrix factored in f64, whose one solve leaves t_z / t_p off
/// by about 1e-16, the tangent at the start, refined in P, has t_z / t_p within 1e-30 of dz/dp in double-double and
/// 1e-62 in quad-double, as with exact factors, unit length to 1e-15 and t_p > 0. The control: the one solve with the
/// same factors, unrefined, misses by more than 1e-30.
#[test]
fn tangent_from_f64_factors_is_exact_to_the_precision() {
    let fam = Fold::<Double>::new();
    let (e, tp, dn) = errors_of::<Double>(&tangent_f64::<Double>(&fam));
    println!("double-double, f64 factors: t_z / t_p off by {e:.3e}, t_p {tp:.6}, |t| - 1 = {dn:.3e}");
    assert!(e <= 1e-30 && tp > 0.0 && dn.abs() <= 1e-15);
    let famq = Fold::<Quad>::new();
    let (eq, tq, dq) = errors_of::<Quad>(&tangent_f64::<Quad>(&famq));
    println!("quad-double, f64 factors: t_z / t_p off by {eq:.3e}, t_p {tq:.6}, |t| - 1 = {dq:.3e}");
    assert!(eq <= 1e-62 && tq > 0.0 && dq.abs() <= 1e-15);

    // the control: -M^-1 of -e_(n+1) with the f64 factors, unrefined
    let mut d = vec![DD::ZERO; N + 1];
    d[N] = DD::ONE;
    let b = Bordered {
        fam: &fam,
        t: d,
        pred: vec![DD::ZERO; N + 1],
    };
    let lu = lowered::<Double, _>(&b, &parts(&w_start::<Double>()), Lower::F64, false).unwrap();
    let mut e1 = vec![DD::ZERO; N + 1];
    e1[N] = -DD::ONE;
    let t1 = Factored::<Double>::solve_neg(&lu, &e1);
    let (ec, _, _) = errors_of::<Double>(&t1);
    println!("control, one solve with f64 factors: t_z / t_p off by {ec:.3e}");
    assert!(ec > 1e-30, "control did not fire");
}

/// F(z, p) = (z_1 + z_2, z_1 + z_2 + z_1^2 + p): the branch z_2 = -z_1, p = -z_1^2 has a simple fold at z = 0,
/// p = 0, where J = [[1, 1], [1 + 2 z_1, 1]] is singular. Rounded to f64, J is singular for every |z_1| below about
/// 1e-16, so a tangent from f64 factors alone has t_p = 0 there.
struct Pinch;

impl<P: Precision<Native>> Family<P> for Pinch {
    fn len(&self) -> usize {
        2
    }
    fn residual_at(&self, z: &[&[f64]], p: P::Scalar, _par: bool) -> Result<Vec<P::Scalar>, String> {
        let z = scalars::<P>(z);
        let u = z[0] + z[1];
        Ok(vec![u, u + z[0] * z[0] + p])
    }
    fn jacobian_at(&self, z: &[&[f64]], _p: P::Scalar, _par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
        let z = scalars::<P>(z);
        let ld = ld_for(2);
        let mut c = vec![vec![0.0; ld * 2]; P::PARTS];
        let d = P::ONE + s::<P>(2.0) * z[0];
        for (q, cq) in c.iter_mut().enumerate() {
            let one = if q == 0 { 1.0 } else { 0.0 };
            (cq[0], cq[1], cq[ld], cq[ld + 1]) = (one, P::part(d, q), one, one);
        }
        Ok((c, ld))
    }
    fn dp_at(&self, _z: &[&[f64]], _p: P::Scalar, _par: bool) -> Result<Vec<P::Scalar>, String> {
        Ok(vec![P::ZERO, P::ONE])
    }
}

/// Pinch's fold located in P from its branch, the continuation from (1, -1, -1) along increasing p with exact
/// factors to z_1 < -1/2, the fold's corrections and tangents factored in f64, the corrector's tolerance `tol` and
/// |t_p| <= `tol`: (z_1, z_1 + z_2, p) at the located point
fn pinch_fold<P: Precision<Native>>(tol: f64) -> (P::Scalar, P::Scalar, P::Scalar) {
    let ex = |b: &Bordered<'_, P, Pinch>, w: &[&[f64]]| exact::<P, _>(b, w, false);
    let f64_factors = |b: &Bordered<'_, P, Pinch>, w: &[&[f64]]| lowered::<P, _>(b, w, Lower::F64, false);
    let z0 = to_parts::<P>(&[P::ONE, -P::ONE]);
    let a = start::<P, _, _>(&Pinch, &parts(&z0), -P::ONE, 1.0, false, ex).unwrap();
    let opts = ArcOpts {
        max_points: 200,
        newton: corrector(tol),
        ..Default::default()
    };
    let path = continuation::<P, _, _>(&Pinch, a, &opts, ex, |pt| pt.z[0][0] < -0.5).unwrap();
    let i = path.folds[0];
    let fopts = FoldOpts {
        tol,
        ..Default::default()
    };
    let (pt, _) = locate_fold::<P, _, _>(
        &Pinch,
        &path.points[i],
        path.ds[i],
        &fopts,
        &corrector(tol),
        f64_factors,
    )
    .unwrap();
    let z = scalars::<P>(&parts(&pt.z));
    (z[0], z[0] + z[1], pt.p)
}

/// Pass, written before the first run: on Pinch, the fold located with its corrections and tangents factored in f64
/// lies at z_1, z_1 + z_2 and p within 1e-29 of 0 in double-double (|t_p| <= 1e-30) and within 1e-61 in quad-double
/// (1e-62). The control: z_1 = 1e-17, where the f64 factors see a fold, fails the check of z_1.
#[test]
fn fold_is_located_with_f64_factors() {
    let check = |v: [f64; 3], tol: f64| v.map(|x| x.abs() <= tol);
    let (z1, u, p) = pinch_fold::<Double>(1e-30);
    let v = [z1.hi, u.hi, p.hi];
    println!(
        "double-double: z_1 {:.3e}, z_1 + z_2 {:.3e}, p {:.3e}",
        v[0], v[1], v[2]
    );
    assert_eq!(check(v, 1e-29), [true; 3]);
    let (z1, u, p) = pinch_fold::<Quad>(1e-62);
    let hi = |x| <Quad as Precision<Native>>::to_dd(x).hi;
    let vq = [hi(z1), hi(u), hi(p)];
    println!(
        "quad-double: z_1 {:.3e}, z_1 + z_2 {:.3e}, p {:.3e}",
        vq[0], vq[1], vq[2]
    );
    assert_eq!(check(vq, 1e-61), [true; 3]);
    assert!(!check([1e-17, 0.0, 0.0], 1e-29)[0], "control did not fire");
}

/// F(z, p) = (1 - p) z - p^2, one unknown: the branch through (0, 0) is z = p^2 / (1 - p), whose tangent there is
/// (0, 1), so that the prediction a step of 1 along it is (0, 1), where dF/dz = 1 - p and the bordered matrix
/// [[0, -2], [0, 1]] are singular. The residual fails for p above `fail_above`, an error of the problem's own.
struct Bend {
    fail_above: f64,
}

impl Family<Double> for Bend {
    fn len(&self) -> usize {
        1
    }
    fn residual_at(&self, z: &[&[f64]], p: DD, _par: bool) -> Result<Vec<DD>, String> {
        if p.hi > self.fail_above {
            return Err(format!("Bend: no residual at p = {}", p.hi));
        }
        let z = DD::new(z[0][0], z[1][0]);
        Ok(vec![(DD::ONE - p) * z - p * p])
    }
    fn jacobian_at(&self, _z: &[&[f64]], p: DD, _par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
        let ld = ld_for(1);
        let d = DD::ONE - p;
        let mut c = vec![vec![0.0; ld]; 2];
        (c[0][0], c[1][0]) = (d.hi, d.lo);
        Ok((c, ld))
    }
    fn dp_at(&self, z: &[&[f64]], p: DD, _par: bool) -> Result<Vec<DD>, String> {
        Ok(vec![-DD::new(z[0][0], z[1][0]) - p.mul_f(2.0)])
    }
}

/// |F| at a point of Bend
fn bend_err(z: &[Vec<f64>], p: DD) -> f64 {
    let z = DD::new(z[0][0], z[1][0]);
    ((DD::ONE - p) * z - p * p).hi.abs()
}

/// Bend's branch from (0, 0) with a first step of 1, `max_points` points
fn bend(fam: &Bend, max_points: usize) -> Result<Path<Double>, String> {
    let ex = |b: &Bordered<'_, Double, Bend>, w: &[&[f64]]| exact::<Double, _>(b, w, false);
    let a = start::<Double, _, _>(fam, &[&[0.0][..], &[0.0][..]], DD::ZERO, 1.0, false, ex)?;
    let opts = ArcOpts {
        ds: 1.0,
        max_points,
        newton: corrector(1e-30),
        ..Default::default()
    };
    continuation::<Double, _, _>(fam, a, &opts, ex, |_| false)
}

/// Pass, written before the first run: on Bend from (0, 0) with a first step of 1, whose prediction has a singular
/// bordered matrix, the continuation refuses that step, halves it and goes on: it returns 4 points with at least one
/// step refused, the second at p = 1/2, z = 1/2 to 1e-30, and every point on the branch to |F| <= 1e-30. The
/// controls: the second point with z moved by 1e-20 fails the check of |F|; and a residual that fails for p > 1/4,
/// an error of the problem's own, still ends the continuation with that error.
#[test]
fn continuation_refuses_a_singular_prediction() {
    let path = bend(
        &Bend {
            fail_above: f64::INFINITY,
        },
        4,
    )
    .unwrap();
    let second = &path.points[1];
    let (z1, p1) = (DD::new(second.z[0][0], second.z[1][0]), second.p);
    let worst = path.points.iter().map(|pt| bend_err(&pt.z, pt.p)).fold(0.0, nan_max);
    println!(
        "{} points, {} refused, second (z, p) = ({:.3e}, {:.3e}) off (1/2, 1/2), max |F| {worst:.3e}",
        path.points.len(),
        path.refused,
        (z1 - DD::from_f64(0.5)).hi,
        (p1 - DD::from_f64(0.5)).hi
    );
    assert!(path.points.len() == 4 && path.refused >= 1 && worst <= 1e-30);
    assert!((z1 - DD::from_f64(0.5)).hi.abs() <= 1e-30 && (p1 - DD::from_f64(0.5)).hi.abs() <= 1e-30);

    // the controls
    let mut z = second.z.clone();
    let zm = DD::new(z[0][0], z[1][0]) + DD::from_f64(1e-20);
    (z[0][0], z[1][0]) = (zm.hi, zm.lo);
    assert!(bend_err(&z, p1) > 1e-30, "control, moved point");
    let e = bend(&Bend { fail_above: 0.25 }, 4).unwrap_err();
    println!("control, a failing residual: {e}");
    assert!(e.starts_with("Bend: no residual"), "control, the problem's error");
}

/// The branch's settings from the start of Fold in double-double: steps from `ds` of at least `ds_min`, aiming at the
/// first contraction `theta_target`, `max_points` points
fn fold_path(
    ds: f64,
    ds_min: f64,
    theta_target: f64,
    max_points: usize,
    damping: Damping,
) -> Result<Path<Double>, String> {
    let fam = Fold::<Double>::new();
    let z0 = z_start::<Double>();
    let a = start::<Double, _, _>(&fam, &parts(&z0), DD::ZERO, 1.0, false, ex::<Double>)?;
    let opts = ArcOpts {
        ds,
        ds_min,
        theta_target,
        max_points,
        newton: NewtonOpts {
            damping,
            ..corrector(1e-30)
        },
        ..Default::default()
    };
    continuation::<Double, _, _>(&fam, a, &opts, ex::<Double>, |_| false)
}

/// Pass, written before the first run: with ds = ds_min = 0.05 and theta_target 1e-12, far below any first
/// contraction, each step taken asks for half its length, and the steps stay at ds_min: 6 points, every step of
/// length at least 0.05. The control: the same run with ds_min 0.025 halves its steps to 0.025 and fails the check.
#[test]
fn continuation_keeps_steps_at_least_ds_min() {
    let check = |path: &Path<Double>| path.points.len() == 6 && path.ds.iter().all(|&d| d >= 0.05);
    let path = fold_path(0.05, 0.05, 1e-12, 6, Damping::Correction).unwrap();
    println!("ds_min 0.05: steps {:?}", path.ds);
    assert!(check(&path));
    let c = fold_path(0.05, 0.025, 1e-12, 6, Damping::Correction).unwrap();
    println!("control, ds_min 0.025: steps {:?}", c.ds);
    assert!(!check(&c), "control did not fire");
}

/// Pass, written before the first run: the step control needs the correction rule's first contraction, which the
/// residual rule does not form, so continuation with the residual rule is an error. The control: with the correction
/// rule the same continuation runs.
#[test]
fn continuation_refuses_the_residual_rule() {
    let e = fold_path(0.1, 1e-8, 0.125, 4, Damping::Residual);
    println!("residual rule: {:?}", e.as_ref().map(|p| p.points.len()));
    assert!(e.is_err());
    assert!(
        fold_path(0.1, 1e-8, 0.125, 4, Damping::Correction).is_ok(),
        "control did not fire"
    );
}

/// Pass, written before the first run: a start for which `stop` returns true ends the continuation there, with one
/// point and no step. The control: a `stop` true for no point, with max_points 2, takes a step.
#[test]
fn continuation_stops_at_a_start_that_meets_stop() {
    let fam = Fold::<Double>::new();
    let z0 = z_start::<Double>();
    let a = start::<Double, _, _>(&fam, &parts(&z0), DD::ZERO, 1.0, false, ex::<Double>).unwrap();
    let opts = ArcOpts {
        max_points: 2,
        newton: corrector(1e-30),
        ..Default::default()
    };
    let path = continuation::<Double, _, _>(&fam, a.clone(), &opts, ex::<Double>, |_| true).unwrap();
    println!(
        "stop at the start: {} points, {} steps",
        path.points.len(),
        path.ds.len()
    );
    assert!(path.points.len() == 1 && path.ds.is_empty());
    let c = continuation::<Double, _, _>(&fam, a, &opts, ex::<Double>, |_| false).unwrap();
    assert!(!(c.points.len() == 1 && c.ds.is_empty()), "control did not fire");
}

/// the branch's exact point with z_1 = (1 - p^2)^(1/2) at p, as w = (z, p) in P
fn on_branch<P: Precision<Native>>(p: f64) -> Vec<P::Scalar> {
    let mut z = vec![(1.0 - p * p).sqrt()];
    for i in 1..N {
        z.push((z[i - 1] * z[i - 1] + p) / 2.0);
    }
    z.into_iter().chain([p]).map(|v| s::<P>(v)).collect()
}

/// Pass, written before the first run: from the start, `step_from` corrects a prediction 1e-6 off the branch's point at
/// p = 0.3 (each unknown moved by 1e-6) to max |G| <= 1e-28 with a first contraction below 1e-4, and with `end_tangent`
/// false the point keeps the start's tangent bit for bit while with it the tangent is the end's. The control: a
/// prediction 0.5 off must not reach the branch with so small a contraction.
#[test]
fn step_from_corrects_a_given_prediction() {
    let fam = Fold::<Double>::new();
    let z0 = z_start::<Double>();
    let a = start::<Double, _, _>(&fam, &parts(&z0), DD::ZERO, 1.0, false, ex::<Double>).unwrap();
    let run = |off: f64, end_tangent: bool| {
        let pred: Vec<DD> = on_branch::<Double>(0.3)
            .iter()
            .map(|&v| v + DD::from_f64(off))
            .collect();
        multisolver::arclength::step_from::<Double, _, _>(
            &fam,
            &a,
            &pred,
            end_tangent,
            &corrector(1e-28),
            &ex::<Double>,
        )
        .unwrap()
    };
    let (rep, pt) = run(1e-6, false);
    let pt = pt.expect("the corrector converged");
    let theta0 = rep.theta[0];
    let g = g_err(&fam, &pt.z, pt.p);
    println!(
        "step_from: {} steps, theta_0 {theta0:.3e}, max |G| {g:.3e}",
        rep.scale.len()
    );
    assert!(g <= 1e-28 && theta0 < 1e-4, "{g:e} {theta0:e}");
    assert!(pt.t.iter().zip(&a.t).all(|(x, y)| x.hi == y.hi && x.lo == y.lo));
    let (_, pe) = run(1e-6, true);
    assert!(pe.unwrap().t.iter().zip(&a.t).any(|(x, y)| x.hi != y.hi));
    let (c, cp) = run(0.5, true);
    let c0 = c.theta.first().copied().unwrap_or(f64::INFINITY);
    println!("control, 0.5 off: converged {}, theta_0 {c0:.3e}", cp.is_some());
    assert!(cp.is_none() || c0 >= 1e-4, "control did not fire");
}

/// the branch from the start with every correction starting damped (lambda0 0.5, as a corrector that damps its first
/// step from a prediction) and `accept_damped`: (path, the accepted steps whose corrector damped)
fn damped_start(accept_damped: bool) -> (Path<Double>, usize) {
    let fam = Fold::<Double>::new();
    let z0 = z_start::<Double>();
    let a = start::<Double, _, _>(&fam, &parts(&z0), DD::ZERO, 1.0, false, ex::<Double>).unwrap();
    let opts = ArcOpts {
        max_points: 200,
        accept_damped,
        newton: NewtonOpts {
            lambda0: 0.5,
            ..corrector(1e-28)
        },
        ..Default::default()
    };
    let path = continuation::<Double, _, _>(&fam, a, &opts, ex::<Double>, |pt| pt.z[0][0] < -0.5).unwrap();
    let damped = path.reports.iter().filter(|r| r.scale.iter().any(|&l| l < 1.0)).count();
    (path, damped)
}

/// Pass, written before the first run: with every correction starting at lambda 0.5 and `accept_damped`, the
/// continuation takes damped steps (every one) and still passes the fold with every point on the branch
/// (`path_checks`, 1e-27). The control: without it every step is refused, and the continuation stalls at its start.
#[test]
fn continuation_accepts_damped_corrections() {
    let fam = Fold::<Double>::new();
    let (path, damped) = damped_start(true);
    let checks = path_checks(&fam, &path, 1e-27);
    let (plain, _) = damped_start(false);
    println!(
        "accept_damped: {} points, {} refused, {damped} damped steps taken, checks {checks:?}; without: {} points, \
         stalled {}",
        path.points.len(),
        path.refused,
        plain.points.len(),
        plain.stalled
    );
    assert!(
        damped >= 1 && damped == path.reports.len() && checks.iter().all(|&c| c),
        "{damped} {checks:?}"
    );
    assert!(plain.points.len() == 1 && plain.stalled, "control did not fire");
}
