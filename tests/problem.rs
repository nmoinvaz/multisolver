//! Newton's method on a problem that is not multiple shooting: a cyclic cubic system written directly as a
//! [`Problem`], with a manufactured root, in double-double and quad-double, the chord and Shamanskii variants that
//! reuse its factorization, its factors in double-double or f64 for quad- and octo-double, and Broyden's updates
//! (`newton_quasi`) on a linear system and on one whose Jacobian changes by rank one.
use multisolver::gcr::{newton_refined, GcrOpts};
use multisolver::multidouble::linalg::ld_for;
use multisolver::newton::{exact, newton, newton_chord, newton_lowered, Damping, Lower, NewtonOpts, NewtonReport};
use multisolver::prec::{max_abs, nan_max, parts};
use multisolver::quasi::{newton_quasi, QuasiOpts};
use multisolver::{Double, Jvp, Native, Octo, Precision, Problem, Quad, DD};
use std::cell::Cell;

const N: usize = 12;

/// F_i(z) = z_i^3 + z_i + c z_(i+1) - b_i, the index cyclic, with b = F's terms at the root z*_i = 0.5 + 0.1 sin(i),
/// so that F(z*) = 0 exactly in P. The Jacobian has 3 z_i^2 + 1 on the diagonal and c on the cyclic superdiagonal.
/// `diag_err` scales the Jacobian's diagonal, 1 for the exact one.
struct Cubic<P: Precision<Native>> {
    c: P::Scalar,
    b: Vec<P::Scalar>,
    diag_err: f64,
    /// scales the diagonal in the products J v, 1 for the exact ones
    jvp_err: f64,
    /// the Jacobians formed
    jacobians: Cell<usize>,
}

impl<P: Precision<Native>> Cubic<P> {
    fn new(diag_err: f64) -> (Self, Vec<Vec<f64>>) {
        let c = P::from_dd(DD::from_f64(0.3));
        let zs: Vec<P::Scalar> = (0..N)
            .map(|i| P::from_dd(DD::from_f64(0.5 + 0.1 * (i as f64).sin())))
            .collect();
        let mut pr = Cubic {
            c,
            b: vec![P::ZERO; N],
            diag_err,
            jvp_err: 1.0,
            jacobians: Cell::new(0),
        };
        pr.b = pr.terms(&zs);
        (pr, to_parts::<P>(&zs))
    }

    /// z_i^3 + z_i + c z_(i+1)
    fn terms(&self, z: &[P::Scalar]) -> Vec<P::Scalar> {
        (0..N)
            .map(|i| z[i] * z[i] * z[i] + z[i] + self.c * z[(i + 1) % N])
            .collect()
    }
}

/// the scalars of a state held as part arrays
fn scalars<P: Precision<Native>>(z: &[&[f64]]) -> Vec<P::Scalar> {
    (0..z[0].len())
        .map(|i| P::from_raw(&z.iter().map(|p| p[i]).collect::<Vec<_>>()))
        .collect()
}

/// part arrays of scalars
fn to_parts<P: Precision<Native>>(v: &[P::Scalar]) -> Vec<Vec<f64>> {
    (0..P::PARTS)
        .map(|q| v.iter().map(|&x| P::part(x, q)).collect())
        .collect()
}

impl<P: Precision<Native>> Problem<P> for Cubic<P> {
    fn len(&self) -> usize {
        N
    }
    fn residual(&self, z: &[&[f64]], _par: bool) -> Result<Vec<P::Scalar>, String> {
        let z = scalars::<P>(z);
        Ok(self.terms(&z).iter().zip(&self.b).map(|(&t, &b)| t - b).collect())
    }
    fn jacobian(&self, z: &[&[f64]], _par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
        self.jacobians.set(self.jacobians.get() + 1);
        let z = scalars::<P>(z);
        let three = P::from_dd(DD::from_f64(3.0));
        let err = P::from_dd(DD::from_f64(self.diag_err));
        let ld = ld_for(N);
        let mut c = vec![vec![0.0; ld * N]; P::PARTS];
        for i in 0..N {
            let d = (three * z[i] * z[i] + P::ONE) * err;
            for (q, cq) in c.iter_mut().enumerate() {
                cq[i * ld + i] = P::part(d, q);
                cq[((i + 1) % N) * ld + i] = P::part(self.c, q);
            }
        }
        Ok((c, ld))
    }
}

impl<P: Precision<Native>> Jvp<P> for Cubic<P> {
    fn jvp(&self, z: &[&[f64]], v: &[&[f64]], _par: bool) -> Result<Vec<P::Scalar>, String> {
        let (z, v) = (scalars::<P>(z), scalars::<P>(v));
        let three = P::from_dd(DD::from_f64(3.0));
        let err = P::from_dd(DD::from_f64(self.jvp_err));
        Ok((0..N)
            .map(|i| (three * z[i] * z[i] + P::ONE) * err * v[i] + self.c * v[(i + 1) % N])
            .collect())
    }
}

/// Newton from z* + 1e-2 wave in P: (converged, steps, max |F| at the end, max |z - z*|)
fn run<P: Precision<Native>>(diag_err: f64, tol: f64) -> (bool, usize, f64, f64) {
    let (pr, zs) = Cubic::<P>::new(diag_err);
    let mut z = start(&zs);
    let opts = NewtonOpts {
        tol,
        max_iter: 8,
        ..Default::default()
    };
    let rep = newton::<P, _>(&pr, &mut z, &opts).unwrap();
    let (zn, zt) = (scalars::<P>(&parts(&z)), scalars::<P>(&parts(&zs)));
    let err = zn
        .iter()
        .zip(&zt)
        .map(|(&a, &b)| P::to_dd(a - b).hi.abs())
        .fold(0.0, nan_max);
    let f = max_abs::<P>(&pr.residual(&parts(&z), false).unwrap());
    (rep.converged, rep.scale.len(), f, err)
}

/// Pass, written before the first run: from a start 1e-2 off the root, Newton converges within 8 steps in
/// double-double to max |F| <= 1e-28 and in quad-double to max |F| <= 1e-60, with max |z - z*| below 1e-27 and
/// 1e-59. Eight steps leave room for the quadratic rate (about five from 1e-2), not for a linear one.
#[test]
fn newton_converges_on_a_problem_that_is_not_shooting() {
    let (ok, steps, f, err) = run::<Double>(1.0, 1e-28);
    println!("double-double: converged {ok} in {steps} steps, max |F| {f:.3e}, max |z - z*| {err:.3e}");
    assert_eq!(checks((ok, f, err), 1e-28, 1e-27), [true; 3]);
    let (ok, steps, f, err) = run::<Quad>(1.0, 1e-60);
    println!("quad-double: converged {ok} in {steps} steps, max |F| {f:.3e}, max |z - z*| {err:.3e}");
    assert_eq!(checks((ok, f, err), 1e-60, 1e-59), [true; 3]);
}

/// the three checks of a run: converged, max |F| <= ftol and max |z - z*| <= etol
fn checks((ok, f, err): (bool, f64, f64), ftol: f64, etol: f64) -> [bool; 3] {
    [ok, f <= ftol, err <= etol]
}

/// The control: with the Jacobian's diagonal 10% off, Newton contracts only linearly, by about 0.1 per step, and must
/// fail every check of the run within the 8 steps, in double-double and in quad-double: it does not converge, and both
/// max |F| and max |z - z*| stay above their bounds.
#[test]
fn newton_control_wrong_jacobian_does_not_converge() {
    let (ok, steps, f, err) = run::<Double>(1.1, 1e-28);
    println!("control, diagonal 10% off: converged {ok} in {steps} steps, max |F| {f:.3e}, max |z - z*| {err:.3e}");
    assert_eq!(checks((ok, f, err), 1e-28, 1e-27), [false; 3], "control did not fire");
    let (ok, steps, f, err) = run::<Quad>(1.1, 1e-60);
    println!("control, quad-double: converged {ok} in {steps} steps, max |F| {f:.3e}, max |z - z*| {err:.3e}");
    assert_eq!(
        checks((ok, f, err), 1e-60, 1e-59),
        [false; 3],
        "quad-double control did not fire"
    );
}

/// the start of the runs: z* + 1e-2 wave
fn start(zs: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut z = zs.to_vec();
    for (i, v) in z[0].iter_mut().enumerate() {
        *v += 1e-2 * (1.0 + 0.5 * (3.0 * i as f64).cos());
    }
    z
}

/// `newton_chord` in double-double from the start, its factors formed every `every` steps (`every` 0: plain
/// `newton`): (report, final z, Jacobians formed)
fn chord(every: usize, max_iter: usize) -> (NewtonReport, Vec<Vec<f64>>, usize) {
    let (pr, zs) = Cubic::<Double>::new(1.0);
    let mut z = start(&zs);
    let opts = NewtonOpts {
        tol: 1e-28,
        max_iter,
        ..Default::default()
    };
    let rep = if every == 0 {
        newton::<Double, _>(&pr, &mut z, &opts).unwrap()
    } else {
        newton_chord::<Double, _>(&pr, &mut z, &opts, every).unwrap()
    };
    (rep, z, pr.jacobians.get())
}

/// the bits of a report and a state
fn bits(rep: &NewtonReport, z: &[Vec<f64>]) -> Vec<u64> {
    rep.fnorm
        .iter()
        .chain(&rep.scale)
        .chain(&rep.theta)
        .chain(z.iter().flatten())
        .map(|v| v.to_bits())
        .chain([rep.converged as u64])
        .collect()
}

/// the two checks that a run is plain Newton: [same report and state bits, same Jacobian count]
fn same_as_newton(every: usize) -> [bool; 2] {
    let (ra, za, ja) = chord(0, 8);
    let (rb, zb, jb) = chord(every, 8);
    [bits(&ra, &za) == bits(&rb, &zb), ja == jb]
}

/// Pass, written before the first run: `newton_chord` with its factors formed at every step is `newton` bit for bit,
/// in the report and the solution, and forms as many Jacobians. The control: formed every third step, it fails both
/// checks.
#[test]
fn chord_every_step_is_newton() {
    assert_eq!(same_as_newton(1), [true; 2], "newton_chord(1) differs from newton");
    assert_eq!(same_as_newton(3), [false; 2], "control did not fire");
}

/// the reuse check: converged, with ceil(steps / every) Jacobians, fewer than its steps
fn reuses(rep: &NewtonReport, jac: usize, every: usize) -> bool {
    let steps = rep.scale.len();
    rep.converged && jac == steps.div_ceil(every) && jac < steps
}

/// Pass, written before the first run: with the factors formed every third step, Newton converges to max |F| <= 1e-28
/// within 12 steps, and forms ceil(steps / 3) Jacobians, fewer than its steps. The control: formed at every step, it
/// forms one Jacobian per step, so the count sees the reuse, and the reuse check rejects it.
#[test]
fn chord_reuses_the_factors() {
    let (rep, _, jac) = chord(3, 12);
    let steps = rep.scale.len();
    println!(
        "every 3: converged {} in {steps} steps, {jac} Jacobians, max |F| {:.3e}",
        rep.converged,
        rep.fnorm.last().unwrap()
    );
    assert!(reuses(&rep, jac, 3));
    let (rep, _, jac) = chord(1, 12);
    println!("control, every 1: {} steps, {jac} Jacobians", rep.scale.len());
    assert_eq!(jac, rep.scale.len(), "control: every step forms its Jacobian");
    assert!(!reuses(&rep, jac, 3), "control did not fire");
}

/// `every` 0 is refused.
#[test]
fn chord_refuses_every_zero() {
    let (pr, zs) = Cubic::<Double>::new(1.0);
    let mut z = start(&zs);
    assert!(newton_chord::<Double, _>(&pr, &mut z, &NewtonOpts::default(), 0).is_err());
}

/// Newton in double-double with its steps refined by gcr on the products J v, preconditioned by the problem's
/// Jacobian with its diagonal 10% off, the products' diagonal scaled by `jvp_err`: (converged, steps, max |F|,
/// products)
fn refined(jvp_err: f64) -> (bool, usize, f64, usize) {
    let (mut pr, zs) = Cubic::<Double>::new(1.1);
    pr.jvp_err = jvp_err;
    let mut z = start(&zs);
    let opts = NewtonOpts {
        tol: 1e-28,
        max_iter: 8,
        ..Default::default()
    };
    let g = GcrOpts {
        step_tol: 1e-10,
        ..Default::default()
    };
    let (rep, products) =
        newton_refined::<Double, _, _>(&pr, &mut z, &opts, &g, |z| exact::<Double, _>(&pr, z, false)).unwrap();
    (rep.converged, rep.scale.len(), *rep.fnorm.last().unwrap(), products)
}

/// Pass, written before the first run: with the Jacobian's diagonal 10% off as the preconditioner, whose plain
/// Newton does not converge in 8 steps (`newton_control_wrong_jacobian_does_not_converge`), the steps refined by gcr
/// on the exact products converge to max |F| <= 1e-28 within the 8.
#[test]
fn refined_newton_converges_on_a_wrong_preconditioner() {
    let (ok, steps, f, products) = refined(1.0);
    println!("refined: converged {ok} in {steps} steps, max |F| {f:.3e}, {products} products");
    assert!(ok && f <= 1e-28);
}

/// The control: products J v with the same wrong diagonal make gcr solve the wrong Jacobian's system exactly, so
/// Newton again does not converge within 8 steps.
#[test]
fn refined_newton_control_wrong_products_do_not_converge() {
    let (ok, steps, f, _) = refined(1.1);
    println!("control, products 10% off: converged {ok} in {steps} steps, max |F| {f:.3e}");
    assert!(!ok, "control did not fire");
}

/// Newton in P from the start with the Jacobian, its diagonal scaled by `diag_err`, rounded by `to` and factored
/// (`newton_lowered`): (converged, steps, max |F| at the end, max |z - z*|)
fn run_lowered<P: Precision<Native>>(diag_err: f64, tol: f64, to: Lower, max_iter: usize) -> (bool, usize, f64, f64) {
    let (pr, zs) = Cubic::<P>::new(diag_err);
    let mut z = start(&zs);
    let opts = NewtonOpts {
        tol,
        max_iter,
        ..Default::default()
    };
    let rep = newton_lowered::<P, _>(&pr, &mut z, &opts, to).unwrap();
    let (zn, zt) = (scalars::<P>(&parts(&z)), scalars::<P>(&parts(&zs)));
    let err = zn
        .iter()
        .zip(&zt)
        .map(|(&a, &b)| P::to_dd(a - b).hi.abs())
        .fold(0.0, nan_max);
    let f = max_abs::<P>(&pr.residual(&parts(&z), false).unwrap());
    (rep.converged, rep.scale.len(), f, err)
}

/// Pass, written before the first run: with the Jacobian rounded to double-double, whose factors leave a contraction
/// of about cond(J) 2^-104 on top of the quadratic rate, Newton in quad-double converges within 8 steps (exact
/// Newton's bound) to max |F| <= 1e-60 with max |z - z*| <= 1e-59, and in octo-double within 10 steps to
/// max |F| <= 1e-120 with max |z - z*| <= 1e-119. With the Jacobian rounded to f64 (a contraction of about
/// cond(J) 2^-53), quad-double converges within 10 steps to the same bounds. The control: the Jacobian's diagonal 10%
/// off, rounded to double-double, contracts by about 0.1 per step and fails all three checks within 8 steps.
#[test]
fn lowered_newton_reaches_the_precision() {
    let case = |name: &str, r: (bool, usize, f64, f64), ftol: f64, etol: f64, want: [bool; 3]| {
        let (ok, steps, f, err) = r;
        println!("{name}: converged {ok} in {steps} steps, max |F| {f:.3e}, max |z - z*| {err:.3e}");
        assert_eq!(checks((ok, f, err), ftol, etol), want, "{name}");
    };
    case(
        "quad-double, factors in double-double",
        run_lowered::<Quad>(1.0, 1e-60, Lower::Double, 8),
        1e-60,
        1e-59,
        [true; 3],
    );
    case(
        "octo-double, factors in double-double",
        run_lowered::<Octo>(1.0, 1e-120, Lower::Double, 10),
        1e-120,
        1e-119,
        [true; 3],
    );
    case(
        "quad-double, factors in f64",
        run_lowered::<Quad>(1.0, 1e-60, Lower::F64, 10),
        1e-60,
        1e-59,
        [true; 3],
    );
    case(
        "control, diagonal 10% off, factors in double-double",
        run_lowered::<Quad>(1.1, 1e-60, Lower::Double, 8),
        1e-60,
        1e-59,
        [false; 3],
    );
}

/// `newton_lowered` in double-double from the start, its Jacobian rounded by `to`: (report, final z)
fn lowered_dd(to: Lower) -> (NewtonReport, Vec<Vec<f64>>) {
    let (pr, zs) = Cubic::<Double>::new(1.0);
    let mut z = start(&zs);
    let opts = NewtonOpts {
        tol: 1e-28,
        max_iter: 8,
        ..Default::default()
    };
    let rep = newton_lowered::<Double, _>(&pr, &mut z, &opts, to).unwrap();
    (rep, z)
}

/// Pass, written before the first run: in double-double the Jacobian rounded to double-double is the Jacobian, so
/// `newton_lowered` with `Lower::Double` is `newton` bit for bit, in the report and the solution. The control: rounded
/// to f64, the report and the solution differ.
#[test]
fn lowered_to_its_own_precision_is_newton() {
    let (ra, za, _) = chord(0, 8);
    let (rb, zb) = lowered_dd(Lower::Double);
    assert_eq!(
        bits(&ra, &za),
        bits(&rb, &zb),
        "newton_lowered(Double) differs from newton"
    );
    let (rc, zc) = lowered_dd(Lower::F64);
    assert_ne!(bits(&ra, &za), bits(&rc, &zc), "control did not fire");
}

const NR: usize = 12;

/// F(z) = A z + u z_0^3 - b in P, A dense and diagonally dominant, u_i = 1 + cos(i) / 2, b = A z* + u (z*_0)^3 for
/// z*_i = 0.5 + 0.1 sin(i): its Jacobian A + 3 z_0^2 u e_0^T changes only in column 0, a change of rank one at any z.
/// It counts the Jacobians formed.
struct RankOne<P: Precision<Native>> {
    a: Vec<Vec<P::Scalar>>,
    u: Vec<P::Scalar>,
    b: Vec<P::Scalar>,
    jacobians: Cell<usize>,
}

impl<P: Precision<Native>> RankOne<P> {
    fn new() -> (Self, Vec<Vec<f64>>) {
        let s = |x: f64| P::from_dd(DD::from_f64(x));
        let a: Vec<Vec<P::Scalar>> = (0..NR)
            .map(|i| {
                (0..NR)
                    .map(|j| {
                        s(if i == j {
                            4.0
                        } else {
                            0.5 * ((3 * i + 5 * j) as f64).sin()
                        })
                    })
                    .collect()
            })
            .collect();
        let u = (0..NR).map(|i| s(1.0 + 0.5 * (i as f64).cos())).collect();
        let zs: Vec<P::Scalar> = (0..NR).map(|i| s(0.5 + 0.1 * (i as f64).sin())).collect();
        let mut pr = RankOne {
            a,
            u,
            b: vec![P::ZERO; NR],
            jacobians: Cell::new(0),
        };
        pr.b = pr.terms(&zs);
        (pr, to_parts::<P>(&zs))
    }

    /// A z + u z_0^3
    fn terms(&self, z: &[P::Scalar]) -> Vec<P::Scalar> {
        let c = z[0] * z[0] * z[0];
        (0..NR)
            .map(|i| P::dot(self.a[i].iter().copied().zip(z.iter().copied())) + self.u[i] * c)
            .collect()
    }
}

impl<P: Precision<Native>> Problem<P> for RankOne<P> {
    fn len(&self) -> usize {
        NR
    }
    fn residual(&self, z: &[&[f64]], _par: bool) -> Result<Vec<P::Scalar>, String> {
        let z = scalars::<P>(z);
        Ok(self.terms(&z).iter().zip(&self.b).map(|(&t, &b)| t - b).collect())
    }
    fn jacobian(&self, z: &[&[f64]], _par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
        self.jacobians.set(self.jacobians.get() + 1);
        let z = scalars::<P>(z);
        let d = P::from_dd(DD::from_f64(3.0)) * z[0] * z[0];
        let ld = ld_for(NR);
        let mut c = vec![vec![0.0; ld * NR]; P::PARTS];
        for i in 0..NR {
            for j in 0..NR {
                let v = if j == 0 {
                    self.a[i][j] + d * self.u[i]
                } else {
                    self.a[i][j]
                };
                for (q, cq) in c.iter_mut().enumerate() {
                    cq[j * ld + i] = P::part(v, q);
                }
            }
        }
        Ok((c, ld))
    }
}

/// `newton_quasi` on RankOne in P from z* + 1e-1 (1 + cos(3 i) / 2), or with `chord` the chord method (one
/// factorization at the start), within `max_iter` steps: (report, max |z - z*|, Jacobians formed)
fn rank_one<P: Precision<Native>>(tol: f64, max_iter: usize, chord: bool) -> (NewtonReport, f64, usize) {
    let (pr, zs) = RankOne::<P>::new();
    let mut z = zs.clone();
    for (i, v) in z[0].iter_mut().enumerate() {
        *v += 1e-1 * (1.0 + 0.5 * (3.0 * i as f64).cos());
    }
    let opts = NewtonOpts {
        tol,
        max_iter,
        ..Default::default()
    };
    let rep = if chord {
        newton_chord::<P, _>(&pr, &mut z, &opts, max_iter).unwrap()
    } else {
        let f = |z: &[&[f64]]| exact::<P, _>(&pr, z, false);
        let (rep, info) = newton_quasi::<P, _, _>(&pr, &mut z, &opts, &QuasiOpts::default(), f).unwrap();
        assert_eq!(info.factorizations, pr.jacobians.get());
        rep
    };
    let (zn, zt) = (scalars::<P>(&parts(&z)), scalars::<P>(&parts(&zs)));
    let err = zn
        .iter()
        .zip(&zt)
        .map(|(&a, &b)| P::to_dd(a - b).hi.abs())
        .fold(0.0, nan_max);
    (rep, err, pr.jacobians.get())
}

/// the check of a run from one Jacobian: converged, max |F| <= ftol, max |z - z*| <= etol, one Jacobian
fn one_jacobian((rep, err, jac): &(NewtonReport, f64, usize), ftol: f64, etol: f64) -> bool {
    rep.converged && *rep.fnorm.last().unwrap() <= ftol && *err <= etol && *jac == 1
}

/// Pass, written before the first run: on RankOne, whose Jacobian changes by rank one, Broyden's updates learn the
/// change along the steps at a secant's superlinear rate, so from 1e-1 off the root `newton_quasi` converges from one
/// Jacobian within 12 steps in double-double (max |F| <= 1e-28, max |z - z*| <= 1e-27) and within 16 in quad-double
/// (1e-60, 1e-59). The control: the chord method, whose contraction is about |A^-1| |u| 3 |z_0^2 - (z*_0)^2|, near
/// 0.2, needs more than 30 steps and fails both checks within the same steps.
///
/// (On the cubic system, whose Jacobian changes in all 12 unknowns, the updates contract at about the chord's rate for
/// about as many steps as unknowns, as Gay's 2n-step bound has it; there they converged in 14 steps where the chord
/// method needs 17, which is why the problem here is one of rank one.)
#[test]
fn quasi_newton_learns_a_rank_one_change() {
    let show = |name: &str, r: &(NewtonReport, f64, usize)| {
        println!(
            "{name}: converged {} in {} steps, {} Jacobians, max |F| {:.3e}, max |z - z*| {:.3e}, theta {:?}",
            r.0.converged,
            r.0.scale.len(),
            r.2,
            r.0.fnorm.last().unwrap(),
            r.1,
            r.0.theta.iter().map(|t| format!("{t:.1e}")).collect::<Vec<_>>()
        );
    };
    let d = rank_one::<Double>(1e-28, 12, false);
    show("double-double", &d);
    assert!(one_jacobian(&d, 1e-28, 1e-27));
    let q = rank_one::<Quad>(1e-60, 16, false);
    show("quad-double", &q);
    assert!(one_jacobian(&q, 1e-60, 1e-59));
    let dc = rank_one::<Double>(1e-28, 12, true);
    show("control, chord, double-double", &dc);
    assert!(!one_jacobian(&dc, 1e-28, 1e-27), "control did not fire");
    let qc = rank_one::<Quad>(1e-60, 16, true);
    show("control, chord, quad-double", &qc);
    assert!(!one_jacobian(&qc, 1e-60, 1e-59), "quad-double control did not fire");
}

/// Pass, written before the first run: with no updates allowed `newton_quasi` is `newton` bit for bit, in the report
/// and the solution, and forms as many Jacobians. The control: with the default 20 updates, it differs in both.
#[test]
fn quasi_without_updates_is_newton() {
    let run = |max_updates: usize| {
        let (pr, zs) = Cubic::<Double>::new(1.0);
        let mut z = start(&zs);
        let opts = NewtonOpts {
            tol: 1e-28,
            max_iter: 8,
            ..Default::default()
        };
        let q = QuasiOpts {
            max_updates,
            ..Default::default()
        };
        let (rep, _) =
            newton_quasi::<Double, _, _>(&pr, &mut z, &opts, &q, |z| exact::<Double, _>(&pr, z, false)).unwrap();
        (bits(&rep, &z), pr.jacobians.get())
    };
    let (ra, za, ja) = chord(0, 8);
    let want = (bits(&ra, &za), ja);
    assert_eq!(run(0), want, "newton_quasi without updates differs from newton");
    let other = run(20);
    assert!(other.0 != want.0 && other.1 != want.1, "control did not fire");
}

/// `newton_quasi` refuses the residual rule, whose trials form no simplified corrections, and a theta_max past 1/2.
#[test]
fn quasi_refuses_the_residual_rule() {
    let (pr, zs) = Cubic::<Double>::new(1.0);
    let f = |z: &[&[f64]]| exact::<Double, _>(&pr, z, false);
    let mut z = start(&zs);
    let opts = NewtonOpts {
        damping: Damping::Residual,
        ..Default::default()
    };
    assert!(newton_quasi::<Double, _, _>(&pr, &mut z, &opts, &QuasiOpts::default(), f).is_err());
    let q = QuasiOpts {
        theta_max: 0.6,
        ..Default::default()
    };
    assert!(newton_quasi::<Double, _, _>(&pr, &mut z, &NewtonOpts::default(), &q, f).is_err());
}

const NL: usize = 4;

/// F(x) = A x - b, A dense and diagonally dominant with entries of nonzero lo parts, b = A x* in double-double for
/// x*_i = 1 + i / 3; its "Jacobian" is A with the diagonal scaled by 1.1, so that Newton's matrix is 10% off
struct Linear {
    a: Vec<Vec<DD>>,
    b: Vec<DD>,
}

impl Linear {
    fn new() -> (Self, Vec<DD>) {
        Linear::scaled(1.0)
    }

    /// with x* scaled by c
    fn scaled(c: f64) -> (Self, Vec<DD>) {
        let a: Vec<Vec<DD>> = (0..NL)
            .map(|i| {
                (0..NL)
                    .map(|j| {
                        let v = if i == j {
                            4.0
                        } else {
                            0.5 * ((3 * i + 5 * j) as f64).sin()
                        };
                        DD::new(v, v * 1e-17)
                    })
                    .collect()
            })
            .collect();
        let xs: Vec<DD> = (0..NL)
            .map(|i| (DD::from_f64(1.0) + DD::from_f64(i as f64) / DD::from_f64(3.0)).mul_f(c))
            .collect();
        let b = a
            .iter()
            .map(|row| <Double as Precision<Native>>::dot(row.iter().copied().zip(xs.iter().copied())))
            .collect();
        (Linear { a, b }, xs)
    }
}

impl Problem<Double> for Linear {
    fn len(&self) -> usize {
        NL
    }
    fn residual(&self, z: &[&[f64]], _par: bool) -> Result<Vec<DD>, String> {
        let x = scalars::<Double>(z);
        Ok(self
            .a
            .iter()
            .zip(&self.b)
            .map(|(row, &bi)| <Double as Precision<Native>>::dot(row.iter().copied().zip(x.iter().copied())) - bi)
            .collect())
    }
    fn jacobian(&self, _z: &[&[f64]], _par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
        let ld = ld_for(NL);
        let mut c = vec![vec![0.0; ld * NL]; 2];
        for (i, row) in self.a.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                let v = if i == j { v * DD::from_f64(1.1) } else { v };
                (c[0][j * ld + i], c[1][j * ld + i]) = (v.hi, v.lo);
            }
        }
        Ok((c, ld))
    }
}

/// On Linear from x = 0, with Newton's matrix 10% off: `newton_quasi` (`max_updates` 20) or, with `max_updates` 0,
/// the same matrix at every step: (converged, steps, max |x - x*|, factorizations)
fn linear(max_updates: usize) -> (bool, usize, f64, usize) {
    let (pr, xs) = Linear::new();
    let mut z = vec![vec![0.0; NL]; 2];
    let opts = NewtonOpts {
        tol: 1e-28,
        max_iter: 2 * NL,
        ..Default::default()
    };
    let q = QuasiOpts {
        max_updates,
        ..Default::default()
    };
    let (rep, info) =
        newton_quasi::<Double, _, _>(&pr, &mut z, &opts, &q, |z| exact::<Double, _>(&pr, z, false)).unwrap();
    let err = scalars::<Double>(&parts(&z))
        .iter()
        .zip(&xs)
        .map(|(&a, &b)| (a - b).hi.abs())
        .fold(0.0, nan_max);
    (rep.converged, rep.scale.len(), err, info.factorizations)
}

/// Pass, written before the first run: Broyden's good method ends on a linear system of n unknowns in at most 2 n
/// steps (Gay, SIAM J. Numer. Anal. 16 (1979) 623), so on the planted 4 x 4 system with Newton's matrix 10% off,
/// `newton_quasi` converges within 8 steps from one factorization, to max |x - x*| <= 1e-27. The control: the same
/// matrix without updates contracts by about 0.1 per step, needs about 28 steps, and fails within the 8.
#[test]
fn quasi_newton_ends_on_a_linear_system_within_2n_steps() {
    let check = |(ok, steps, err, fac): (bool, usize, f64, usize)| ok && steps <= 2 * NL && err <= 1e-27 && fac == 1;
    let r = linear(20);
    println!(
        "updates: converged {} in {} steps, max |x - x*| {:.3e}, {} factorizations",
        r.0, r.1, r.2, r.3
    );
    assert!(check(r));
    let c = linear(0);
    println!(
        "control, no updates: converged {} in {} steps, max |x - x*| {:.3e}, {} factorizations",
        c.0, c.1, c.2, c.3
    );
    assert!(!check(c), "control did not fire");
}

/// Linear, its residual an error at a state that is not finite
struct Strict(Linear);

impl Problem<Double> for Strict {
    fn len(&self) -> usize {
        NL
    }
    fn residual(&self, z: &[&[f64]], par: bool) -> Result<Vec<DD>, String> {
        if z.iter().any(|q| q.iter().any(|v| !v.is_finite())) {
            return Err("Strict: a state that is not finite".into());
        }
        self.0.residual(z, par)
    }
    fn jacobian(&self, z: &[&[f64]], par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
        self.0.jacobian(z, par)
    }
}

/// `newton_quasi` on Strict with x* scaled by c, from x = 0 with Newton's matrix 10% off, to max |F| <= 1e-28 c:
/// (converged, steps, max |x - x*| / c, steps by the updates)
fn tiny(c: f64) -> Result<(bool, usize, f64, usize), String> {
    let (lin, xs) = Linear::scaled(c);
    let pr = Strict(lin);
    let mut z = vec![vec![0.0; NL]; 2];
    let opts = NewtonOpts {
        tol: 1e-28 * c,
        max_iter: 40,
        ..Default::default()
    };
    let (rep, info) = newton_quasi::<Double, _, _>(&pr, &mut z, &opts, &QuasiOpts::default(), |z| {
        exact::<Double, _>(&pr, z, false)
    })?;
    let err = scalars::<Double>(&parts(&z))
        .iter()
        .zip(&xs)
        .map(|(&a, &b)| (a - b).hi.abs())
        .fold(0.0, nan_max);
    Ok((rep.converged, rep.scale.len(), err / c, info.quasi_steps))
}

/// Pass, written before the first run: with x* scaled by 1e-150, where the steps fall below 1e-154 during the run and
/// |s|^2 leaves f64's range, and by 1e-170, where the first step's does, `newton_quasi` never forms a trial state that
/// is not finite: on Strict, whose residual fails at one, it converges within 40 steps to max |x - x*| <= 1e-27 c.
/// The control: Strict's residual fails at a NaN.
#[test]
fn quasi_newton_at_a_tiny_scale() {
    for c in [1e-150, 1e-170] {
        let r = tiny(c);
        println!("x* scaled by {c:e}: {r:?}");
        let (ok, steps, err, _) = r.unwrap();
        assert!(ok && steps <= 40 && err <= 1e-27, "scale {c:e}");
    }
    let (lin, _) = Linear::new();
    let nan = vec![vec![f64::NAN; NL], vec![0.0; NL]];
    assert!(
        Strict(lin).residual(&parts(&nan), false).is_err(),
        "control did not fire"
    );
}
