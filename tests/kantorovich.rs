//! The a-posteriori Kantorovich quantities (`kantorovich`) on problems whose affine covariant Lipschitz constant is
//! known.
use multisolver::kantorovich::{estimates, from_steps};
use multisolver::multidouble::linalg::ld_for;
use multisolver::newton::{newton, NewtonOpts};
use multisolver::{Double, Problem, DD};

/// c_i (x_i^2 - a_i), decoupled: J(x)^-1 (J(y) - J(x)) v = ((y_i - x_i) v_i / x_i) whatever the row factors c, so
/// for one unknown the simplified correction gives [omega] = 1 / |x0| exactly (|dz_bar| = dz^2 / (2 x0))
struct Squares(Vec<f64>, Vec<f64>);

impl Problem<Double> for Squares {
    fn len(&self) -> usize {
        self.0.len()
    }
    fn residual(&self, z: &[&[f64]], _par: bool) -> Result<Vec<DD>, String> {
        Ok((0..self.0.len())
            .map(|i| {
                let x = DD::new(z[0][i], z[1][i]);
                (x * x - DD::from_f64(self.0[i])).mul_f(self.1[i])
            })
            .collect())
    }
    fn jacobian(&self, z: &[&[f64]], _par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
        let n = self.0.len();
        let ld = ld_for(n);
        let (mut hi, mut lo) = (vec![0.0; ld * n], vec![0.0; ld * n]);
        for i in 0..n {
            let d = DD::new(z[0][i], z[1][i]).mul_f(2.0 * self.1[i]);
            (hi[i * ld + i], lo[i * ld + i]) = (d.hi, d.lo);
        }
        Ok((vec![hi, lo], ld))
    }
}

/// Newton from x0 on x^2 = 2 (one unknown, the 2-norm), full steps: the report's first step's [omega] against 1 / x0
fn first_omega(x0: f64) -> (f64, f64) {
    let pr = Squares(vec![2.0], vec![1.0]);
    let mut z = vec![vec![x0], vec![0.0]];
    let o = NewtonOpts {
        tol: 1e-28,
        par: false,
        ..Default::default()
    };
    let rep = newton::<Double, _>(&pr, &mut z, &o).unwrap();
    let k = estimates(&rep, true);
    (k[0].omega, 1.0 / x0)
}

/// Pass, written before the first run: on x^2 = 2 from x0 = 1.5, 3 and 10, the first step's [omega] is 1 / x0 within
/// 1e-14 relative; the error estimate of every step from the second on is within a factor 2 of |x_k - sqrt 2|. (The
/// pass first written asked the estimate to bound the error from above, which the module's documentation does not
/// claim, [h] being a lower bound: at step 1 it is 0.413 against 0.419.) The control: on two unknowns from (3, 1.5)
/// with row factors (1, 1000), theta read from max |F| (|F_1| / |F_0|, the residual's rate, which its largest row
/// sets, not the corrections') must miss the corrections' [omega]_0 by more than 10%. (On one unknown the two rates
/// agree, which the control first written used.)
#[test]
fn kantorovich_recovers_a_known_omega() {
    for x0 in [1.5, 3.0, 10.0] {
        let (w, want) = first_omega(x0);
        println!("x0 {x0}: [omega]_0 {w:.15e}, 1 / x0 {want:.15e}");
        assert!((w / want - 1.0).abs() <= 1e-14, "x0 {x0}");
    }
    let pr = Squares(vec![2.0], vec![1.0]);
    let mut z = vec![vec![3.0], vec![0.0]];
    let o = NewtonOpts {
        tol: 1e-28,
        par: false,
        ..Default::default()
    };
    let rep = newton::<Double, _>(&pr, &mut z, &o).unwrap();
    let ks = estimates(&rep, true);
    // x_k, the iterate after k steps, from the same run stopped there
    let root = DD::from_f64(2.0).sqrt();
    let iterate = |k: usize| {
        let mut zk = vec![vec![3.0], vec![0.0]];
        let ok = NewtonOpts { max_iter: k, ..o };
        newton::<Double, _>(&pr, &mut zk, &ok).unwrap();
        DD::new(zk[0][0], zk[1][0])
    };
    for (k, s) in ks.iter().enumerate() {
        assert_eq!(s.lam, 1.0);
        let err = (iterate(k) - root).hi.abs();
        if k >= 1 && err > 1e-25 {
            let est = s.error_estimate();
            println!("step {k}: error {err:.3e}, estimate {est:.3e}");
            assert!(
                est >= 0.5 * err && est <= 2.0 * err,
                "step {k}: estimate {est:e}, error {err:e}"
            );
        }
    }
    let pr2 = Squares(vec![2.0, 2.0], vec![1.0, 1000.0]);
    let mut z2 = vec![vec![3.0, 1.5], vec![0.0, 0.0]];
    let rep2 = newton::<Double, _>(&pr2, &mut z2, &o).unwrap();
    let k2 = estimates(&rep2, true);
    let theta_f = [rep2.fnorm[1] / rep2.fnorm[0]];
    let wrong = from_steps(&[1.0], &theta_f, &[k2[0].dz], true);
    let miss = (wrong[0].omega / k2[0].omega - 1.0).abs();
    println!(
        "control, theta from max |F|: [omega]_0 {:.4e} against the corrections' {:.4e}, off by {miss:.3}",
        wrong[0].omega, k2[0].omega
    );
    assert!(miss > 0.1, "control did not fire");
}

/// Pass, written before the first run: for a damped step, [h] = 2 max(theta - (1 - lambda), 0) / lambda^2: at lambda
/// 0.5 and theta 0.6, [h] = 0.8; at theta 0.4 the bound is 0; and a full step gives 2 theta. The control: the same
/// theta 0.6 read as a full step's must give 1.2, not 0.8.
#[test]
fn damped_steps_use_the_triangle_bound() {
    let s = from_steps(&[0.5, 0.5, 1.0], &[0.6, 0.4, 0.3], &[1.0, 1.0, 2.0], true);
    assert!((s[0].h - 0.8).abs() < 1e-15 && s[1].h == 0.0 && (s[2].h - 0.6).abs() < 1e-15);
    assert!((s[2].omega - 0.3).abs() < 1e-15);
    let full = from_steps(&[1.0], &[0.6], &[1.0], true);
    assert!((full[0].h - 0.8).abs() > 0.1, "control did not fire");
}

/// x - 1 with the factor 2 in place of its Jacobian 1
struct Approx;

impl Problem<Double> for Approx {
    fn len(&self) -> usize {
        1
    }
    fn residual(&self, z: &[&[f64]], _par: bool) -> Result<Vec<DD>, String> {
        Ok(vec![DD::new(z[0][0], z[1][0]) - DD::ONE])
    }
    fn jacobian(&self, _z: &[&[f64]], _par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
        let ld = ld_for(1);
        let mut hi = vec![0.0; ld];
        hi[0] = 2.0;
        Ok((vec![hi, vec![0.0; ld]], ld))
    }
}

/// Pass, written before the first run: with approximate factors (x - 1 from 2, the factor 2 for the Jacobian 1),
/// `[omega]` comes out 2 where omega is 0, and the steps say they are not lower bounds when the caller says the
/// factors were not exact. The control: the same report stated exact says lower bound, which is the caller's error
/// the flag exists to record.
#[test]
fn approximate_factors_give_no_lower_bound() {
    let mut z = vec![vec![2.0], vec![0.0]];
    let o = NewtonOpts {
        tol: 1e-28,
        max_iter: 3,
        par: false,
        ..Default::default()
    };
    let rep = newton::<Double, _>(&Approx, &mut z, &o).unwrap();
    let not = estimates(&rep, false);
    println!(
        "approximate factors: [omega]_0 {:.3}, lower bound {}",
        not[0].omega, not[0].lower_bound
    );
    assert!((not[0].omega - 2.0).abs() < 1e-12 && not.iter().all(|s| !s.lower_bound));
    assert!(
        estimates(&rep, true).iter().all(|s| s.lower_bound),
        "control did not fire"
    );
}
