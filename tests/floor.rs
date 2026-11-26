//! The floor rule and the precision ladder on planted problems whose roots are irrational: z_i = sqrt(a_i), a_i = 2 + i,
//! so that the residual in each precision has a floor at about its rounding. One problem is well conditioned (a
//! cyclic coupling); the other's Jacobian, the Hilbert matrix of order 22 times diag(2 z), has a condition of about
//! 10^31, past double-double factors' reach.
use multisolver::ladder::{jacobian_ladder, newton_ladder, residual_ladder, Prec, Solve, Stage};
use multisolver::multidouble::linalg::ld_for;
use multisolver::multidouble::od::OD;
use multisolver::newton::{newton, Damping, NewtonOpts, Stop};
use multisolver::prec::nan_max;
use multisolver::{Native, Precision, Problem, DD};
use std::cell::Cell;

/// F(z) = A u, u_i = z_i^2 - a_i, a_i = 2 + i, with A = I + k C (C the cyclic shift) or the Hilbert matrix
/// H_ij = 1 / (i + j + 1) rounded to double-double (`hilbert`), the same matrix in every precision. The root is
/// z*_i = sqrt(a_i) for any nonsingular A, and F's Jacobian is A diag(2 z). The rounded Hilbert matrix of order 22
/// has a condition of 2.72e31 (mpmath's singular values at 80 digits). Double-double factors of the Jacobian, its
/// entries rounded to 2^-106 independently, leave Newton's steps a linear contraction bounded by 2.7e31 2^-106, about
/// 0.3 (about 1e-5 per step in the runs here), and quad-double ones a contraction bounded by 2.7e31 2^-209.
struct Roots {
    n: usize,
    k: f64,
    hilbert: bool,
    residuals: Cell<usize>,
}

impl Roots {
    fn coupled() -> Self {
        Roots {
            n: 12,
            k: 0.3,
            hilbert: false,
            residuals: Cell::new(0),
        }
    }

    fn ill() -> Self {
        Roots {
            n: 22,
            k: 0.0,
            hilbert: true,
            residuals: Cell::new(0),
        }
    }

    /// A's entry (i, j) in P
    fn a<P: Precision<Native>>(&self, i: usize, j: usize) -> P::Scalar {
        if self.hilbert {
            P::from_dd(DD::ONE / DD::from_f64((i + j + 1) as f64))
        } else if i == j {
            P::ONE
        } else if j == (i + 1) % self.n {
            P::from_dd(DD::from_f64(self.k))
        } else {
            P::ZERO
        }
    }

    /// a start off the root by the factor 1 + rel in every unknown, in f64 (two parts, the second zero)
    fn start(&self, rel: f64) -> Vec<Vec<f64>> {
        let hi: Vec<f64> = (0..self.n).map(|i| (2.0 + i as f64).sqrt() * (1.0 + rel)).collect();
        vec![hi, vec![0.0; self.n]]
    }

    /// max |z_i - sqrt(a_i)|, in octo-double
    fn error(&self, z: &[Vec<f64>]) -> f64 {
        (0..self.n)
            .map(|i| {
                let mut p = [0.0; 8];
                for (q, part) in z.iter().enumerate() {
                    p[q] = part[i];
                }
                (OD::new(p) - OD::from_f64(2.0 + i as f64).sqrt()).to_f64().abs()
            })
            .fold(0.0, nan_max)
    }
}

fn scalars<P: Precision<Native>>(z: &[&[f64]]) -> Vec<P::Scalar> {
    (0..z[0].len())
        .map(|i| P::from_raw(&z.iter().map(|p| p[i]).collect::<Vec<_>>()))
        .collect()
}

impl<P: Precision<Native>> Problem<P> for Roots {
    fn len(&self) -> usize {
        self.n
    }
    fn residual(&self, z: &[&[f64]], _par: bool) -> Result<Vec<P::Scalar>, String> {
        self.residuals.set(self.residuals.get() + 1);
        let z = scalars::<P>(z);
        let u: Vec<P::Scalar> = (0..self.n)
            .map(|i| z[i] * z[i] - P::from_dd(DD::from_f64(2.0 + i as f64)))
            .collect();
        Ok((0..self.n)
            .map(|i| P::dot((0..self.n).map(|j| (self.a::<P>(i, j), u[j]))))
            .collect())
    }
    fn jacobian(&self, z: &[&[f64]], _par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
        let z = scalars::<P>(z);
        let two = P::from_dd(DD::from_f64(2.0));
        let ld = ld_for(self.n);
        let mut c = vec![vec![0.0; ld * self.n]; P::PARTS];
        for i in 0..self.n {
            for j in 0..self.n {
                let e = self.a::<P>(i, j) * (two * z[j]);
                for (q, cq) in c.iter_mut().enumerate() {
                    cq[j * ld + i] = P::part(e, q);
                }
            }
        }
        Ok((c, ld))
    }
}

fn opts(tol: f64, floor_cut: f64, damping: Damping) -> NewtonOpts {
    NewtonOpts {
        tol,
        floor_cut,
        damping,
        max_iter: 20,
        par: false,
        ..Default::default()
    }
}

/// Newton in double-double on the coupled problem from `rel` off the root, past its floor (tol 1e-60): (report,
/// residuals evaluated, max |z - z*|)
fn dd_run(rel: f64, o: &NewtonOpts) -> (multisolver::NewtonReport, usize, f64) {
    let pr = Roots::coupled();
    let mut z = pr.start(rel);
    let rep = newton::<multisolver::Double, _>(&pr, &mut z, o).unwrap();
    (rep, pr.residuals.get(), pr.error(&z))
}

/// The floor rule in double-double: pass, written before the first run, with either damping from 1e-3 off the root
/// and tol 1e-60: the run stops at `Stop::Floor` within 8 steps, at max |F| <= 1e-29 with max |z - z*| <= 1e-30, and
/// every step before the last cut max |F| by 10 or more.
#[test]
fn newton_stops_at_the_double_double_floor() {
    for damping in [Damping::Correction, Damping::Residual] {
        let (rep, nres, err) = dd_run(1e-3, &opts(1e-60, 10.0, damping));
        let f = *rep.fnorm.last().unwrap();
        println!(
            "{damping:?}: {:?} after {} steps, {nres} residuals, max |F| {f:.3e}, max |z - z*| {err:.3e}",
            rep.stop,
            rep.scale.len()
        );
        assert!(matches!(rep.stop, Stop::Floor { .. }), "{:?}", rep.stop);
        assert!(rep.scale.len() <= 8 && f <= 1e-29 && err <= 1e-30, "{rep:?}");
        let steps = rep.scale.len();
        assert!(
            (1..steps).all(|i| rep.fnorm[i - 1] >= 10.0 * rep.fnorm[i]),
            "{:?}",
            rep.fnorm
        );
    }
}

/// The control of `newton_stops_at_the_double_double_floor`: with the rule off (floor_cut 0) the run must end
/// otherwise than at the floor and evaluate at least 5 residuals more, with either damping.
#[test]
fn floor_control_rule_off_runs_on() {
    for damping in [Damping::Correction, Damping::Residual] {
        let (on, n_on, _) = dd_run(1e-3, &opts(1e-60, 10.0, damping));
        let (off, n_off, _) = dd_run(1e-3, &opts(1e-60, 0.0, damping));
        println!(
            "{damping:?}: rule on {:?}, {n_on} residuals; off {:?}, {n_off} residuals",
            on.stop, off.stop
        );
        assert!(
            !matches!(off.stop, Stop::Floor { .. }) && n_off >= n_on + 5,
            "control did not fire: {:?} {n_off} {n_on}",
            off.stop
        );
    }
}

/// The floor rule waits for a small step: pass, from 10 times the root (steps that cut max |F| by about 4, with max
/// |dz| of order 1) the run still reaches the floor, max |F| <= 1e-29 and max |z - z*| <= 1e-30, with either damping.
#[test]
fn floor_rule_waits_for_a_small_step() {
    for damping in [Damping::Correction, Damping::Residual] {
        let (rep, _, err) = dd_run(9.0, &opts(1e-60, 10.0, damping));
        let f = *rep.fnorm.last().unwrap();
        println!(
            "{damping:?}: {:?} after {} steps, max |F| {f:.3e}, max |z - z*| {err:.3e}",
            rep.stop,
            rep.scale.len()
        );
        assert!(
            matches!(rep.stop, Stop::Floor { .. }) && f <= 1e-29 && err <= 1e-30,
            "{rep:?}"
        );
    }
}

/// The control of `floor_rule_waits_for_a_small_step`: without the bound on the step (floor_dz infinite) the rule
/// must stop the run at its first slow step, far from the root, max |F| above 1e-3.
#[test]
fn floor_control_without_the_step_bound_stops_early() {
    for damping in [Damping::Correction, Damping::Residual] {
        let o = NewtonOpts {
            floor_dz: f64::INFINITY,
            ..opts(1e-60, 10.0, damping)
        };
        let (rep, _, _) = dd_run(9.0, &o);
        let f = *rep.fnorm.last().unwrap();
        println!(
            "{damping:?}: {:?} after {} steps at max |F| {f:.3e}",
            rep.stop,
            rep.scale.len()
        );
        assert!(
            matches!(rep.stop, Stop::Floor { .. }) && f > 1e-3,
            "control did not fire: {rep:?}"
        );
    }
}

/// the residuals' ladder (Jacobian formed and solved in double-double; 12 steps, then 6 chord steps per stage) on the
/// coupled problem from 1e-3 off the root, tol 1e-110, its first `stages` stages
fn residual_run(floor_cut: f64, stages: usize) -> (multisolver::ladder::LadderReport, f64) {
    let pr = Roots::coupled();
    let mut z = pr.start(1e-3);
    let mut st = residual_ladder(Prec::Double, Solve::Double, 12, 6);
    st.truncate(stages);
    let rep = newton_ladder(&pr, &mut z, &st, &opts(1e-110, floor_cut, Damping::Correction)).unwrap();
    (rep, pr.error(&z))
}

/// The residuals' ladder: pass, written before the first run: it converges to max |F| <= 1e-110 with max |z - z*| <=
/// 1e-110, its double-double and quad-double stages end at the floor and its octo-double stage at the tolerance, the
/// later two stages take at most 3 steps each, and at most 8 Jacobians are formed.
#[test]
fn residual_ladder_reaches_octo_double() {
    let (rep, err) = residual_run(10.0, 3);
    for (st, r) in &rep.stages {
        println!(
            "{:?}: {:?} after {} steps, max |F| {:.3e}",
            st.res,
            r.stop,
            r.scale.len(),
            r.fnorm.last().unwrap()
        );
    }
    println!("{} Jacobians, max |z - z*| {err:.3e}", rep.jacobians);
    assert!(rep.converged && err <= 1e-110, "{err:e}");
    let stops: Vec<Stop> = rep.stages.iter().map(|(_, r)| r.stop).collect();
    assert!(
        stops.len() == 3
            && matches!(stops[0], Stop::Floor { .. })
            && matches!(stops[1], Stop::Floor { .. })
            && stops[2] == Stop::Tol,
        "{stops:?}"
    );
    assert!(rep.stages[1..].iter().all(|(_, r)| r.scale.len() <= 3));
    assert!(rep.jacobians <= 8, "{}", rep.jacobians);
}

/// The controls of `residual_ladder_reaches_octo_double`: without its octo-double stage the ladder must end
/// unconverged at the quad-double floor, above 1e-100; and with the floor rule off its double-double stage must not
/// end at a floor.
#[test]
fn residual_ladder_controls() {
    let (short, _) = residual_run(10.0, 2);
    let (_, last) = short.stages.last().unwrap();
    let f = *last.fnorm.last().unwrap();
    println!("without octo-double: {:?} at {f:.3e}", last.stop);
    assert!(!short.converged && matches!(last.stop, Stop::Floor { .. }) && f > 1e-100);
    let (off, _) = residual_run(0.0, 3);
    println!("rule off: the double-double stage {:?}", off.stages[0].1.stop);
    assert!(!matches!(off.stages[0].1.stop, Stop::Floor { .. }));
}

/// the ladder `stages` on the ill-conditioned problem from 1e-3 off the root, tol 1e-100
fn ill_run(stages: &[Stage]) -> multisolver::ladder::LadderReport {
    let pr = Roots::ill();
    let mut z = pr.start(1e-3);
    newton_ladder(&pr, &mut z, stages, &opts(1e-100, 10.0, Damping::Correction)).unwrap()
}

/// The quad-double solve: pass, written before the first run: on the problem whose Jacobian has a condition of about
/// 2.7e31, the octo-double residual with the quad-double Jacobian solved in quad-double converges to max |F| <= 1e-100
/// within 8 steps.
#[test]
fn quad_double_solve_converges_on_an_ill_conditioned_jacobian() {
    let rep = ill_run(&[Stage::new(Prec::Octo, Prec::Quad, Solve::Quad, 8)]);
    let r = &rep.stages[0].1;
    println!(
        "qd solve: {:?} after {} steps, max |F| {:.3e}",
        r.stop,
        r.scale.len(),
        r.fnorm.last().unwrap()
    );
    assert!(rep.converged && r.scale.len() <= 8);
}

/// The control of `quad_double_solve_converges_on_an_ill_conditioned_jacobian`: the same Jacobian solved in
/// double-double, whose steps contract only linearly, must not converge within 8 steps.
#[test]
fn quad_double_solve_control_double_double_solve_stalls() {
    let rep = ill_run(&[Stage::new(Prec::Octo, Prec::Quad, Solve::Double, 8)]);
    let r = &rep.stages[0].1;
    println!(
        "dd solve: {:?} after {} steps, max |F| {:?}",
        r.stop,
        r.scale.len(),
        r.fnorm
    );
    assert!(!rep.converged, "control did not fire");
}

/// The Jacobians' ladder at the octo-double residual: pass, it converges, in its third stage (the quad-double solve),
/// the first two ending unconverged; the control: its first two stages alone must not converge.
#[test]
fn jacobian_ladder_hands_over_to_the_quad_double_solve() {
    let rep = ill_run(&jacobian_ladder(Prec::Octo, 8));
    for (st, r) in &rep.stages {
        println!(
            "{:?} J, {:?} solve: {:?} after {} steps, max |F| {:.3e}",
            st.jac,
            st.solve,
            r.stop,
            r.scale.len(),
            r.fnorm.last().unwrap()
        );
    }
    assert!(rep.converged && rep.stages.len() == 3 && rep.stages[2].0.solve == Solve::Quad);
    let short = ill_run(&jacobian_ladder(Prec::Octo, 8)[..2]);
    assert!(!short.converged, "control did not fire");
}
