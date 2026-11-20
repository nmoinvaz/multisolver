//! The correction rule's scaled norms (`Scaling::Iterates`, NLEQ-ERR's weights): a problem and its copy with the
//! unknowns rescaled take the same steps, in `newton`, `newton_quasi` and the arclength corrector, and a problem
//! whose small unknown hides a strong nonlinearity from the 2-norm is damped where it needs to be.

use multisolver::arclength::{continuation, start, ArcOpts, Bordered, Path};
use multisolver::multidouble::linalg::ld_for;
use multisolver::newton::{exact, newton, NewtonOpts, NewtonReport, Scaling};
use multisolver::prec::{nan_max, parts};
use multisolver::quasi::{newton_quasi, QuasiInfo, QuasiOpts};
use multisolver::{Double, Family, Native, Precision, Problem, Quad, DD};

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

/// 1 / a in P: the double-double reciprocal and two Newton steps
fn recip<P: Precision<Native>>(a: P::Scalar) -> P::Scalar {
    let x = P::from_dd(DD::ONE / P::to_dd(a));
    let x = x + x * (P::ONE - a * x);
    x + x * (P::ONE - a * x)
}

/// the scaled norms with a floor below every unknown of the tests
const SCALED: Scaling = Scaling::Iterates { floor: 1e-30 };

const N: usize = 4;

/// the units of the rescaled copy, powers of two from 2^-20 (about 1e-6) to 2^10, so that z' = D z is exact
const D: [f64; N] = [9.5367431640625e-7, 0.0078125, 8.0, 1024.0];

/// F_i = z_i^3 + z_i + c z_(i+1) - b_i, the index cyclic, at z = D^-1 z' in the units `d` (ones for the original),
/// with b from the root z*_i = 0.5 + 0.1 sin(i). The Jacobian in z' is dF/dz D^-1.
struct Cubic<P: Precision<Native>> {
    c: P::Scalar,
    b: Vec<P::Scalar>,
    d: [f64; N],
}

impl<P: Precision<Native>> Cubic<P> {
    fn new(d: [f64; N]) -> Self {
        let c = s::<P>(0.3);
        let zs = Self::root();
        let b = (0..N)
            .map(|i| zs[i] * zs[i] * zs[i] + zs[i] + c * zs[(i + 1) % N])
            .collect();
        Cubic { c, b, d }
    }

    fn root() -> Vec<P::Scalar> {
        (0..N).map(|i| s::<P>(0.5 + 0.1 * (i as f64).sin())).collect()
    }

    /// z in the original units
    fn unscale(&self, zp: &[&[f64]]) -> Vec<P::Scalar> {
        scalars::<P>(zp)
            .iter()
            .zip(&self.d)
            .map(|(&v, &d)| v * s::<P>(1.0 / d))
            .collect()
    }

    /// the start z_i = 3 in the units d
    fn start(&self) -> Vec<Vec<f64>> {
        to_parts::<P>(&self.d.map(|d| s::<P>(3.0 * d)))
    }
}

impl<P: Precision<Native>> Problem<P> for Cubic<P> {
    fn len(&self) -> usize {
        N
    }
    fn residual(&self, zp: &[&[f64]], _par: bool) -> Result<Vec<P::Scalar>, String> {
        let z = self.unscale(zp);
        Ok((0..N)
            .map(|i| z[i] * z[i] * z[i] + z[i] + self.c * z[(i + 1) % N] - self.b[i])
            .collect())
    }
    fn jacobian(&self, zp: &[&[f64]], _par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
        let z = self.unscale(zp);
        let ld = ld_for(N);
        let mut c = vec![vec![0.0; ld * N]; P::PARTS];
        for i in 0..N {
            let j = (i + 1) % N;
            let diag = (s::<P>(3.0) * z[i] * z[i] + P::ONE) * s::<P>(1.0 / self.d[i]);
            let off = self.c * s::<P>(1.0 / self.d[j]);
            for (q, cq) in c.iter_mut().enumerate() {
                cq[i * ld + i] = P::part(diag, q);
                cq[j * ld + i] = P::part(off, q);
            }
        }
        Ok((c, ld))
    }
}

/// max over i of |z_i - z*_i| at z' in the units of `pr`
fn root_err<P: Precision<Native>>(pr: &Cubic<P>, z: &[Vec<f64>]) -> f64 {
    pr.unscale(&parts(z))
        .iter()
        .zip(Cubic::<P>::root())
        .map(|(&a, b)| P::to_dd(a - b).hi.abs())
        .fold(0.0, nan_max)
}

/// The control of `root_err`'s bound: the root moved by `delta` in every unknown, in the units d, is off by more than
/// `etol`
fn moved_root_fails<P: Precision<Native>>(d: [f64; N], delta: f64, etol: f64) -> bool {
    let pr = Cubic::<P>::new(d);
    let z: Vec<P::Scalar> = Cubic::<P>::root()
        .iter()
        .zip(&d)
        .map(|(&r, &di)| (r + s::<P>(delta)) * s::<P>(di))
        .collect();
    root_err(&pr, &to_parts::<P>(&z)) > etol
}

/// whether two reports took the same steps: equal step counts, lambda and theta within 1e-12 relative
fn same_steps(a: &NewtonReport, b: &NewtonReport) -> bool {
    let close = |x: &[f64], y: &[f64]| {
        x.len() == y.len()
            && x.iter()
                .zip(y)
                .all(|(&u, &v)| (u - v).abs() <= 1e-12 * u.abs().max(v.abs()))
    };
    a.converged && b.converged && close(&a.scale, &b.scale) && close(&a.theta, &b.theta)
}

/// `newton` from z_i = 3 with lambda0 = 1/4 on the original and the rescaled copy, the norms `scaling`: the reports and
/// the copies' errors from the root
fn pair<P: Precision<Native>>(scaling: Scaling, tol: f64) -> ((NewtonReport, f64), (NewtonReport, f64)) {
    let opts = NewtonOpts {
        tol,
        lambda0: 0.25,
        scaling,
        par: false,
        ..Default::default()
    };
    let run = |d: [f64; N]| {
        let pr = Cubic::<P>::new(d);
        let mut z = pr.start();
        let rep = newton::<P, _>(&pr, &mut z, &opts).unwrap();
        let e = root_err(&pr, &z);
        (rep, e)
    };
    (run([1.0; N]), run(D))
}

/// Pass, written before the first run: with the scaled norms, `newton` on the cubic and on its copy with the unknowns
/// in units from 2^-20 to 2^10 takes the same steps (equal step counts, lambda and theta within 1e-12 relative), and
/// both reach the root to 1e-27 in double-double (tolerance 1e-28) and 1e-59 in quad-double (1e-60). The scales are
/// powers of two, so the copies' arithmetic is the same up to exact scalings and the steps are expected bitwise
/// equal. The controls: in the 2-norm, which the copy's largest unknown dominates, the two take different steps; and
/// the root moved by 1e-20 (double-double) or 1e-40 (quad-double) fails the bound on the error.
///
/// The first version also required a damped first step from lambda0 = 1/4. It failed: the cubic needs no damping from
/// z_i = 3, and the rule's one raise per step lifts lambda to 1 there. The damped case is
/// `scaled_damping_does_not_depend_on_the_units`.
#[test]
fn scaled_steps_do_not_depend_on_the_units() {
    for (name, ((ra, ea), (rb, eb)), etol) in [
        ("double-double", pair::<Double>(SCALED, 1e-28), 1e-27),
        ("quad-double", pair::<Quad>(SCALED, 1e-60), 1e-59),
    ] {
        let bitwise = ra.scale == rb.scale && ra.theta == rb.theta;
        println!(
            "{name}: {} and {} steps, lambda {:?}, errors {ea:.3e} and {eb:.3e}, bitwise {bitwise}",
            ra.scale.len(),
            rb.scale.len(),
            ra.scale
        );
        assert!(same_steps(&ra, &rb) && ea <= etol && eb <= etol, "{name}");
    }
    assert!(
        moved_root_fails::<Double>(D, 1e-20, 1e-27),
        "control, double-double error"
    );
    assert!(moved_root_fails::<Quad>(D, 1e-40, 1e-59), "control, quad-double error");
    let ((ra, _), (rb, _)) = pair::<Double>(Scaling::None, 1e-28);
    println!(
        "control, 2-norm: {} and {} steps, theta {:?} and {:?}",
        ra.scale.len(),
        rb.scale.len(),
        ra.theta,
        rb.theta
    );
    assert!(!same_steps(&ra, &rb), "control did not fire");
}

/// `newton_quasi` with exact factors from z_i = 3 on the original and the rescaled copy, the norms `scaling`
fn quasi_pair(scaling: Scaling) -> [(NewtonReport, QuasiInfo, f64); 2] {
    let opts = NewtonOpts {
        tol: 1e-28,
        max_iter: 40,
        scaling,
        par: false,
        ..Default::default()
    };
    [[1.0; N], D].map(|d| {
        let pr = Cubic::<Double>::new(d);
        let mut z = pr.start();
        let (rep, info) = newton_quasi::<Double, _, _>(&pr, &mut z, &opts, &QuasiOpts::default(), |z| {
            exact::<Double, _>(&pr, z, false)
        })
        .unwrap();
        let e = root_err(&pr, &z);
        (rep, info, e)
    })
}

/// Pass, written before the first run: with the scaled norms and inner products, `newton_quasi` on the cubic and its
/// rescaled copy takes the same steps (as `same_steps`), with the same factorizations and updates' steps (at least
/// one), both to the root within 1e-27. The controls: in the 2-norm the two take different steps; and the root moved
/// by 1e-20 fails the bound on the error.
#[test]
fn scaled_quasi_newton_does_not_depend_on_the_units() {
    let [(ra, ia, ea), (rb, ib, eb)] = quasi_pair(SCALED);
    println!(
        "scaled: {ia:?} and {ib:?}, errors {ea:.3e} and {eb:.3e}, theta {:?}",
        ra.theta
    );
    assert!(same_steps(&ra, &rb) && ia == ib && ia.quasi_steps >= 1 && ea <= 1e-27 && eb <= 1e-27);
    assert!(moved_root_fails::<Double>([1.0; N], 1e-20, 1e-27), "control, error");
    let [(ra, ia, _), (rb, ib, _)] = quasi_pair(Scaling::None);
    println!("control, 2-norm: {ia:?} and {ib:?}");
    assert!(!same_steps(&ra, &rb), "control did not fire");
}

/// F_1 = (z_1^2 - 10^6) / 2000, F_2 = g(u) with u = (z_2 - c) 10^6, c = 2e-6, and g(u) = u / (1 + |u|), at
/// z = D^-1 z' in the units `d` (ones for the original): the root (1000, 2e-6). Newton's map of g is u -> -u |u|,
/// which diverges from |u| > 1, so the small unknown needs damping while its corrections are 10^-6 of z_1's.
struct Kink {
    d: [f64; 2],
}

/// Kink's units in which both unknowns are of order 1, powers of two
const KD: [f64; 2] = [0.0009765625, 524288.0];

impl Kink {
    /// z in the original units
    fn unscale<P: Precision<Native>>(&self, zp: &[&[f64]]) -> Vec<P::Scalar> {
        scalars::<P>(zp)
            .iter()
            .zip(&self.d)
            .map(|(&v, &d)| v * s::<P>(1.0 / d))
            .collect()
    }
}

impl<P: Precision<Native>> Problem<P> for Kink {
    fn len(&self) -> usize {
        2
    }
    fn residual(&self, z: &[&[f64]], _par: bool) -> Result<Vec<P::Scalar>, String> {
        let z = self.unscale::<P>(z);
        let u = (z[1] - s::<P>(2e-6)) * s::<P>(1e6);
        let au = if P::to_dd(u).hi < 0.0 { -u } else { u };
        Ok(vec![
            (z[0] * z[0] - s::<P>(1e6)) * s::<P>(0.0005),
            u * recip::<P>(P::ONE + au),
        ])
    }
    fn jacobian(&self, z: &[&[f64]], _par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
        let z = self.unscale::<P>(z);
        let u = (z[1] - s::<P>(2e-6)) * s::<P>(1e6);
        let au = if P::to_dd(u).hi < 0.0 { -u } else { u };
        let r = recip::<P>(P::ONE + au);
        let j11 = z[0] * s::<P>(0.001) * s::<P>(1.0 / self.d[0]);
        let j22 = s::<P>(1e6) * r * r * s::<P>(1.0 / self.d[1]);
        let ld = ld_for(2);
        let mut c = vec![vec![0.0; ld * 2]; P::PARTS];
        for (q, cq) in c.iter_mut().enumerate() {
            (cq[0], cq[ld + 1]) = (P::part(j11, q), P::part(j22, q));
        }
        Ok((c, ld))
    }
}

/// `newton` on Kink in the units d from (1100, 4e-6) (u = 2), the norms `scaling`: (report, |z_1 - 1000| / 1000,
/// |z_2 / c - 1|)
fn kink_in<P: Precision<Native>>(d: [f64; 2], scaling: Scaling, tol: f64) -> (NewtonReport, f64, f64) {
    let pr = Kink { d };
    let mut z = to_parts::<P>(&[s::<P>(1100.0 * d[0]), s::<P>(4e-6) * s::<P>(d[1])]);
    let opts = NewtonOpts {
        tol,
        scaling,
        par: false,
        ..Default::default()
    };
    let rep = newton::<P, _>(&pr, &mut z, &opts).unwrap();
    let (e1, e2) = kink_errors::<P>(&pr, &z);
    (rep, e1, e2)
}

/// |z_1 - 1000| / 1000 and |z_2 / c - 1| at z' in the units of `pr`
fn kink_errors<P: Precision<Native>>(pr: &Kink, z: &[Vec<f64>]) -> (f64, f64) {
    let z = pr.unscale::<P>(&parts(z));
    let e1 = P::to_dd(z[0] - s::<P>(1000.0)).hi.abs() / 1000.0;
    let e2 = P::to_dd(z[1] * recip::<P>(s::<P>(2e-6)) - P::ONE).hi.abs();
    (e1, e2)
}

/// The control of `kink_errors`' bounds: the root moved by the relative `delta` in both unknowns, in the units d,
/// is off by more than `t1` and `t2`
fn moved_kink_fails<P: Precision<Native>>(d: [f64; 2], delta: f64, t1: f64, t2: f64) -> bool {
    let pr = Kink { d };
    let up = P::ONE + s::<P>(delta);
    let z = to_parts::<P>(&[s::<P>(1000.0) * up * s::<P>(d[0]), s::<P>(2e-6) * up * s::<P>(d[1])]);
    let (e1, e2) = kink_errors::<P>(&pr, &z);
    e1 > t1 && e2 > t2
}

/// `kink_in` in the original units
fn kink<P: Precision<Native>>(scaling: Scaling, tol: f64) -> (NewtonReport, f64, f64) {
    kink_in::<P>([1.0; 2], scaling, tol)
}

/// Pass, written before the first run: with the scaled norms, `newton` on Kink and on its copy in the units
/// (2^-10, 2^19), where both unknowns are of order 1, takes the same steps (as `same_steps`), the first damped
/// (lambda < 1), both to z_1 and z_2 within 1e-28 and 1e-27 relative in double-double. The controls: in the 2-norm
/// the two take different steps; and the root moved by 1e-20 relative fails both bounds on the errors.
#[test]
fn scaled_damping_does_not_depend_on_the_units() {
    let (ra, a1, a2) = kink_in::<Double>([1.0; 2], SCALED, 1e-28);
    let (rb, b1, b2) = kink_in::<Double>(KD, SCALED, 1e-28);
    let bitwise = ra.scale == rb.scale && ra.theta == rb.theta;
    println!(
        "scaled: {} and {} steps, lambda {:?}, errors {a1:.3e} {a2:.3e} and {b1:.3e} {b2:.3e}, bitwise {bitwise}",
        ra.scale.len(),
        rb.scale.len(),
        ra.scale
    );
    assert!(same_steps(&ra, &rb) && ra.scale[0] < 1.0);
    assert!(a1 <= 1e-28 && a2 <= 1e-27 && b1 <= 1e-28 && b2 <= 1e-27);
    assert!(moved_kink_fails::<Double>(KD, 1e-20, 1e-28, 1e-27), "control, errors");
    let (ra, _, _) = kink_in::<Double>([1.0; 2], Scaling::None, 1e-28);
    let (rb, _, _) = kink_in::<Double>(KD, Scaling::None, 1e-28);
    println!(
        "control, 2-norm: converged {} in {} steps and {} in {} steps",
        ra.converged,
        ra.scale.len(),
        rb.converged,
        rb.scale.len()
    );
    assert!(!same_steps(&ra, &rb), "control did not fire");
}

/// Pass, written before the first run: on Kink, whose small unknown z_2 diverges under full steps while z_1's
/// corrections, 10^6 times larger, set the 2-norm, the scaled norms damp the first step (lambda < 1) and `newton`
/// converges within 20 steps to max |F| <= 1e-28 with z_1 and z_2 within 1e-28 and 1e-27 relative in double-double,
/// and to 1e-60 with 1e-60 and 1e-59 in quad-double. The controls: in the 2-norm the first steps are full, u runs to
/// 2, -4, 16, -256, and the method stops without converging; and the root moved by 1e-20 (double-double) or 1e-40
/// (quad-double) relative fails both bounds on the errors.
#[test]
fn scaled_norms_damp_a_small_unknown() {
    for (name, (rep, e1, e2), t1, t2) in [
        ("double-double", kink::<Double>(SCALED, 1e-28), 1e-28, 1e-27),
        ("quad-double", kink::<Quad>(SCALED, 1e-60), 1e-60, 1e-59),
    ] {
        println!(
            "{name}: converged {} in {} steps, lambda {:?}, errors {e1:.3e} {e2:.3e}",
            rep.converged,
            rep.scale.len(),
            rep.scale
        );
        assert!(rep.converged && rep.scale.len() <= 20 && e1 <= t1 && e2 <= t2, "{name}");
        assert!(rep.scale[0] < 1.0, "{name}: the first step is not damped");
    }
    assert!(
        moved_kink_fails::<Double>([1.0; 2], 1e-20, 1e-28, 1e-27),
        "control, double-double errors"
    );
    assert!(
        moved_kink_fails::<Quad>([1.0; 2], 1e-40, 1e-60, 1e-59),
        "control, quad-double errors"
    );
    let (rep, _, _) = kink::<Double>(Scaling::None, 1e-28);
    println!(
        "control, 2-norm: converged {} in {} steps, lambda {:?}",
        rep.converged,
        rep.scale.len(),
        rep.scale
    );
    assert!(!rep.converged, "control did not fire");
}

/// Pass, written before the first run: a floor that is not positive and finite is an error in `newton` and in
/// `newton_quasi`. The control: the floor 1e-30 runs.
#[test]
fn scaling_refuses_a_floor_that_is_not_positive() {
    let pr = Cubic::<Double>::new([1.0; N]);
    for floor in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let opts = NewtonOpts {
            scaling: Scaling::Iterates { floor },
            ..Default::default()
        };
        let mut z = pr.start();
        assert!(
            newton::<Double, _>(&pr, &mut z, &opts).is_err(),
            "newton, floor {floor}"
        );
        let q = newton_quasi::<Double, _, _>(&pr, &mut z, &opts, &QuasiOpts::default(), |z| {
            exact::<Double, _>(&pr, z, false)
        });
        assert!(q.is_err(), "newton_quasi, floor {floor}");
    }
    let opts = NewtonOpts {
        scaling: SCALED,
        ..Default::default()
    };
    let mut z = pr.start();
    assert!(newton::<Double, _>(&pr, &mut z, &opts).is_ok(), "control did not fire");
}

/// z_1^2 + p^2 - 1 and z_2 - (z_1^2 + p) / 2 with z_2 in the units 2^-20: a circle in (z_1, p) with folds at p = +-1
struct Circle;

impl Family<Double> for Circle {
    fn len(&self) -> usize {
        2
    }
    fn residual_at(&self, z: &[&[f64]], p: DD, _par: bool) -> Result<Vec<DD>, String> {
        let z = scalars::<Double>(z);
        let z2 = z[1] * DD::from_f64(1.0 / D[0]);
        Ok(vec![
            z[0] * z[0] + p * p - DD::ONE,
            z2 - (z[0] * z[0] + p) * DD::from_f64(0.5),
        ])
    }
    fn jacobian_at(&self, z: &[&[f64]], _p: DD, _par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
        let z = scalars::<Double>(z);
        let ld = ld_for(2);
        let e = [z[0].mul_f(2.0), -z[0], DD::ZERO, DD::from_f64(1.0 / D[0])];
        let mut c = vec![vec![0.0; ld * 2]; 2];
        for (q, cq) in c.iter_mut().enumerate() {
            let part = |x: DD| if q == 0 { x.hi } else { x.lo };
            (cq[0], cq[1], cq[ld], cq[ld + 1]) = (part(e[0]), part(e[1]), part(e[2]), part(e[3]));
        }
        Ok((c, ld))
    }
    fn dp_at(&self, _z: &[&[f64]], p: DD, _par: bool) -> Result<Vec<DD>, String> {
        Ok(vec![p.mul_f(2.0), -DD::from_f64(0.5)])
    }
}

/// max |G| over a path's points, G the circle's equations in the original units
fn off_branch(path: &Path<Double>) -> f64 {
    path.points
        .iter()
        .map(|pt| {
            let f = Circle.residual_at(&parts(&pt.z), pt.p, false).unwrap();
            f.iter().map(|v| v.hi.abs()).fold(0.0, nan_max)
        })
        .fold(0.0, nan_max)
}

/// Pass, written before the first run: arclength continuation of the circle from (z_1, p) = (1, 0), its corrector in
/// the scaled norms with the floor 1e-3 (z_1 passes 0 at the fold), passes the fold at p = 1 to z_1 < -1/2 within 100
/// points, with exactly one fold, every point on the circle to max |G| <= 1e-29. The control: one point's z_1 moved by
/// 1e-20 fails the last check.
#[test]
fn arclength_corrects_in_the_scaled_norms() {
    let ex = |b: &Bordered<'_, Double, Circle>, w: &[&[f64]]| exact::<Double, _>(b, w, false);
    let z0 = to_parts::<Double>(&[DD::ONE, DD::from_f64(0.5 * D[0])]);
    let a = start::<Double, _, _>(&Circle, &parts(&z0), DD::ZERO, 1.0, false, ex).unwrap();
    let opts = ArcOpts {
        newton: NewtonOpts {
            tol: 1e-30,
            scaling: Scaling::Iterates { floor: 1e-3 },
            par: false,
            ..Default::default()
        },
        ..Default::default()
    };
    let path = continuation::<Double, _, _>(&Circle, a, &opts, ex, |pt| pt.z[0][0] < -0.5).unwrap();
    let last = &path.points[path.points.len() - 1];
    let off = off_branch(&path);
    println!(
        "{} points, {} refused, folds after {:?}, last (z_1, p) = ({:.6}, {:.6}), max |G| {off:.3e}",
        path.points.len(),
        path.refused,
        path.folds,
        last.z[0][0],
        last.p.hi
    );
    assert!(last.z[0][0] < -0.5 && path.folds.len() == 1 && off <= 1e-29);
    let mut moved = path.clone();
    let i = moved.points.len() / 2;
    let z1 = DD::new(moved.points[i].z[0][0], moved.points[i].z[1][0]) + DD::from_f64(1e-20);
    (moved.points[i].z[0][0], moved.points[i].z[1][0]) = (z1.hi, z1.lo);
    assert!(off_branch(&moved) > 1e-29, "control did not fire");
}

/// F(z) = z - 1.5e308 in one unknown, linear: from z = 1e308 the iterates' magnitudes sum past f64's largest
struct Huge;

impl Problem<Double> for Huge {
    fn len(&self) -> usize {
        1
    }
    fn residual(&self, z: &[&[f64]], _par: bool) -> Result<Vec<DD>, String> {
        Ok(vec![DD::new(z[0][0], z[1][0]) - DD::from_f64(1.5e308)])
    }
    fn jacobian(&self, _z: &[&[f64]], _par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
        let ld = ld_for(1);
        let mut c = vec![vec![0.0; ld]; 2];
        c[0][0] = 1.0;
        Ok((c, ld))
    }
}

/// Pass, written before the first run: on Huge from z = 1e308, where |z_i| + |z'_i| overflows f64 at the first step,
/// the scaled norms' weights stay finite and `newton` converges in one full step to z = 1.5e308 exactly. The control:
/// that point moved by one f64 ulp fails the check of z.
#[test]
fn scaled_norms_take_iterates_near_the_largest_f64() {
    let opts = NewtonOpts {
        tol: 1e-28,
        scaling: SCALED,
        par: false,
        ..Default::default()
    };
    let mut z = vec![vec![1e308], vec![0.0]];
    let rep = newton::<Double, _>(&Huge, &mut z, &opts).unwrap();
    let at = |z: &[Vec<f64>]| DD::new(z[0][0], z[1][0]) == DD::from_f64(1.5e308);
    println!(
        "converged {} in {} steps, lambda {:?}, z = {:e} + {:e}",
        rep.converged,
        rep.scale.len(),
        rep.scale,
        z[0][0],
        z[1][0]
    );
    assert!(rep.converged && rep.scale == [1.0] && at(&z));
    let moved = vec![vec![1.5e308f64.next_up()], vec![0.0]];
    assert!(!at(&moved), "control did not fire");
}
