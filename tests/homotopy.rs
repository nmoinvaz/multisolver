//! The homotopy in a held offset (`homotopy`) on a planted path whose end is known, and on paths with and without a
//! fold.
use multisolver::homotopy::{homotopy, HomotopyOpts, HomotopyStop};
use multisolver::multidouble::linalg::ld_for;
use multisolver::prec::nan_max;
use multisolver::{Double, Problem, DD};
use std::sync::atomic::{AtomicUsize, Ordering};

/// F(z) = (I + 0.3 C)(z o z), C the cyclic shift, in double-double
struct Squares(usize);

impl Problem<Double> for Squares {
    fn len(&self) -> usize {
        self.0
    }
    fn residual(&self, z: &[&[f64]], _par: bool) -> Result<Vec<DD>, String> {
        let n = self.0;
        let u: Vec<DD> = (0..n)
            .map(|i| {
                let x = DD::new(z[0][i], z[1][i]);
                x * x
            })
            .collect();
        Ok((0..n).map(|i| u[i] + u[(i + 1) % n].mul_f(0.3)).collect())
    }
    fn jacobian(&self, z: &[&[f64]], _par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
        let n = self.0;
        let ld = ld_for(n);
        let (mut hi, mut lo) = (vec![0.0; ld * n], vec![0.0; ld * n]);
        for i in 0..n {
            let j = (i + 1) % n;
            let d = DD::new(z[0][i], z[1][i]).mul_f(2.0);
            let e = DD::new(z[0][j], z[1][j]).mul_f(0.6);
            (hi[i * ld + i], lo[i * ld + i]) = (d.hi, d.lo);
            (hi[j * ld + i], lo[j * ld + i]) = (e.hi, e.lo);
        }
        Ok((vec![hi, lo], ld))
    }
}

fn opts() -> HomotopyOpts {
    HomotopyOpts {
        floor_f: 1e-28,
        floor_dz: 1e-28,
        polish_tol: 1e-30,
        par: false,
        ..Default::default()
    }
}

/// the planted path: z0, zP = z0 + w, a = F(zP), b = F(z0) - F(zP) (b negated with `wrong_sign`): (report, max |z(1)
/// - zP|, max |b|)
fn planted(wrong_sign: bool) -> (multisolver::homotopy::HomotopyReport, f64, f64) {
    let n = 8;
    let pr = Squares(n);
    let z0: Vec<f64> = (0..n).map(|i| 1.0 + 0.1 * i as f64).collect();
    let zp: Vec<f64> = (0..n).map(|i| z0[i] + 0.4 * ((i as f64) * 1.7).sin()).collect();
    let f = |z: &[f64]| pr.residual(&[z, &vec![0.0; n]], false).unwrap();
    let (f0, fp) = (f(&z0), f(&zp));
    let a = fp.clone();
    let b: Vec<DD> = f0
        .iter()
        .zip(&fp)
        .map(|(&x, &y)| if wrong_sign { y - x } else { x - y })
        .collect();
    let mut z = vec![z0, vec![0.0; n]];
    let rep = homotopy::<Double, _>(&pr, &mut z, &a, &b, &opts()).unwrap();
    let err = (0..n)
        .map(|i| (DD::new(z[0][i], z[1][i]) - DD::from_f64(zp[i])).hi.abs())
        .fold(0.0, nan_max);
    let bmax = b.iter().map(|v| v.hi.abs()).fold(0.0, nan_max);
    (rep, err, bmax)
}

/// Pass, written before the first run: on the planted path (8 unknowns, zP = z0 + 0.4 sin(1.7 i)), the start solves
/// H(z0, 0) to 1e-20 max |b|, the trace reaches t = 1, and its end is zP within 1e-28. The control: b negated, so
/// that z0 does not solve H(z, 0), must fail the start check.
#[test]
fn homotopy_reaches_the_planted_end() {
    let (rep, err, bmax) = planted(false);
    println!(
        "planted: start {:.3e} (max |b| {bmax:.3e}), {} rungs, {} refused, {:?}, end within {err:.3e}, polish {:?}",
        rep.start,
        rep.rungs.len(),
        rep.refusals,
        rep.stop,
        rep.polish
    );
    assert!(
        rep.start <= 1e-20 * bmax && rep.stop == HomotopyStop::Reached && err <= 1e-28,
        "{rep:?}"
    );
    let (c, _, cb) = planted(true);
    println!("control, b negated: start {:.3e}", c.start);
    assert!(c.start > 1e-20 * cb, "control did not fire");
}

/// One unknown: H(z, t) = z^2 - a - (1 - t) b from z0 = 1
struct Square;

impl Problem<Double> for Square {
    fn len(&self) -> usize {
        1
    }
    fn residual(&self, z: &[&[f64]], _par: bool) -> Result<Vec<DD>, String> {
        let x = DD::new(z[0][0], z[1][0]);
        Ok(vec![x * x])
    }
    fn jacobian(&self, z: &[&[f64]], _par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
        let ld = ld_for(1);
        let d = DD::new(z[0][0], z[1][0]).mul_f(2.0);
        let (mut hi, mut lo) = (vec![0.0; ld], vec![0.0; ld]);
        (hi[0], lo[0]) = (d.hi, d.lo);
        Ok((vec![hi, lo], ld))
    }
}

/// the trace of z^2 = a + (1 - t) b from z = 1: (report, last t, last z)
fn square_path(a: f64, b: f64) -> (multisolver::homotopy::HomotopyReport, f64, f64) {
    let mut z = vec![vec![1.0], vec![0.0]];
    let rep = homotopy::<Double, _>(&Square, &mut z, &[DD::from_f64(a)], &[DD::from_f64(b)], &opts()).unwrap();
    let t = rep.rungs.last().map_or(0.0, |r| r.t);
    (rep, t, z[0][0])
}

/// Pass, written before the first run: z^2 = 1 - 2t from z = 1 (a = -1, b = 2) folds at t = 1/2 (z = 0): the
/// trace stops before t = 1, by a refusal run or a step too small, its last t within 1e-3 below 1/2. The control: a =
/// 0.25, b = 0.75, z^2 = 1 - 0.75 t, has no fold and must reach t = 1 at z = 1/2.
#[test]
fn homotopy_stops_at_a_fold() {
    let (rep, t, z) = square_path(-1.0, 2.0);
    println!(
        "fold: {:?} after {} rungs at t {t:.6}, z {z:.3e}",
        rep.stop,
        rep.rungs.len()
    );
    assert!(
        matches!(rep.stop, HomotopyStop::Refused | HomotopyStop::StepTooSmall) && t <= 0.5 && t > 0.499,
        "{rep:?}"
    );
    let (c, tc, zc) = square_path(0.25, 0.75);
    println!("control, no fold: {:?} at t {tc}, z {zc}", c.stop);
    assert!(
        c.stop == HomotopyStop::Reached && (zc - 0.5).abs() < 1e-15,
        "control did not fire"
    );
}

/// `Square` whose Jacobian fails from its evaluation number `.0` on
struct Failing(usize, AtomicUsize);

impl Problem<Double> for Failing {
    fn len(&self) -> usize {
        1
    }
    fn residual(&self, z: &[&[f64]], par: bool) -> Result<Vec<DD>, String> {
        Square.residual(z, par)
    }
    fn jacobian(&self, z: &[&[f64]], par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
        if self.1.fetch_add(1, Ordering::Relaxed) >= self.0 {
            return Err("jacobian failed".into());
        }
        Square.jacobian(z, par)
    }
}

/// Pass, written before the first run: on z^2 = 1 - 0.75 t, a Jacobian that fails at the first corrector's iterate
/// (its second evaluation) ends the trace with the problem's error rather than refusing the rung. The control: no
/// failure, which must return a report.
#[test]
fn homotopy_returns_a_jacobian_error() {
    let run = |from| {
        let mut z = vec![vec![1.0], vec![0.0]];
        let pr = Failing(from, AtomicUsize::new(0));
        homotopy::<Double, _>(&pr, &mut z, &[DD::from_f64(0.25)], &[DD::from_f64(0.75)], &opts())
    };
    let r = run(1);
    println!("failing jacobian: {:?}", r.as_ref().map(|rep| rep.stop));
    assert_eq!(r.err().as_deref(), Some("jacobian failed"));
    let c = run(usize::MAX);
    println!("control, no failure: {:?}", c.as_ref().map(|rep| rep.stop));
    assert!(c.is_ok(), "control did not fire");
}
