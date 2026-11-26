//! Newton's method with a projection (`newton::newton_with_projection`) on a planted system whose last equation the
//! projection solves exactly.
use multisolver::multidouble::linalg::ld_for;
use multisolver::newton::{exact, newton_with, newton_with_projection, NewtonOpts};
use multisolver::{Double, Problem, DD};
use std::cell::Cell;

const N: usize = 6;

fn a_i(i: usize) -> f64 {
    2.0 + 0.5 * i as f64
}

/// F_i(z) = z_i^2 - a_i - 0.2 z_(i+1), the index cyclic; records the largest |F_(N-1)| it is evaluated at
struct Ring(Cell<f64>);

impl Problem<Double> for Ring {
    fn len(&self) -> usize {
        N
    }
    fn residual(&self, z: &[&[f64]], _par: bool) -> Result<Vec<DD>, String> {
        let x: Vec<DD> = (0..N).map(|i| DD::new(z[0][i], z[1][i])).collect();
        let f: Vec<DD> = (0..N)
            .map(|i| x[i] * x[i] - DD::from_f64(a_i(i)) - x[(i + 1) % N].mul_f(0.2))
            .collect();
        self.0.set(self.0.get().max(f[N - 1].hi.abs()));
        Ok(f)
    }
    fn jacobian(&self, z: &[&[f64]], _par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
        let ld = ld_for(N);
        let (mut hi, mut lo) = (vec![0.0; ld * N], vec![0.0; ld * N]);
        for i in 0..N {
            let d = DD::new(z[0][i], z[1][i]).mul_f(2.0);
            (hi[i * ld + i], lo[i * ld + i]) = (d.hi, d.lo);
            hi[((i + 1) % N) * ld + i] = -0.2;
        }
        Ok((vec![hi, lo], ld))
    }
}

/// z_(N-1) set so that F_(N-1) = 0: z_(N-1) = sqrt(a_(N-1) + 0.2 z_0), plus `off`
fn project(off: f64) -> impl Fn(&mut Vec<Vec<f64>>) -> Result<(), String> {
    move |z: &mut Vec<Vec<f64>>| {
        let x0 = DD::new(z[0][0], z[1][0]);
        let v = (DD::from_f64(a_i(N - 1)) + x0.mul_f(0.2)).sqrt() + DD::from_f64(off);
        (z[0][N - 1], z[1][N - 1]) = (v.hi, v.lo);
        Ok(())
    }
}

fn opts() -> NewtonOpts {
    NewtonOpts {
        tol: 1e-30,
        par: false,
        ..Default::default()
    }
}

/// the start: z_i = sqrt(a_i) + 0.3
fn start() -> Vec<Vec<f64>> {
    vec![(0..N).map(|i| a_i(i).sqrt() + 0.3).collect(), vec![0.0; N]]
}

/// Pass, written before the first run: from 0.3 off the root, Newton with the projection converges to max |F| <=
/// 1e-30, and every point its residual is evaluated at has |F_(N-1)| <= 1e-30 (the projection solves that row). The
/// control: the projection 1e-3 off its row must not converge.
#[test]
fn projection_keeps_its_row_solved() {
    let pr = Ring(Cell::new(0.0));
    let mut z = start();
    let rep = newton_with_projection::<Double, _, _>(
        &pr,
        &mut z,
        &opts(),
        |w| exact::<Double, _>(&pr, w, false),
        project(0.0),
    )
    .unwrap();
    let row = pr.0.get();
    println!(
        "projected: converged {} in {} steps, largest |F_(N-1)| seen {row:.3e}",
        rep.converged,
        rep.scale.len()
    );
    assert!(rep.converged && row <= 1e-30, "{rep:?} {row:e}");
    let pc = Ring(Cell::new(0.0));
    let mut zc = start();
    let c = newton_with_projection::<Double, _, _>(
        &pc,
        &mut zc,
        &opts(),
        |w| exact::<Double, _>(&pc, w, false),
        project(1e-3),
    )
    .unwrap();
    println!(
        "control, projection 1e-3 off: converged {}, {:?}, last max |F| {:.3e}",
        c.converged,
        c.stop,
        c.fnorm.last().unwrap()
    );
    assert!(!c.converged, "control did not fire");
}

/// Pass, written before the first run: the identity projection gives `newton_with`'s run bit for bit (the state and
/// every max |F|). The control: a projection that moves z_0 by one ulp at each point must give other bits.
#[test]
fn identity_projection_is_newton_with() {
    let run = |shift: bool| {
        let pr = Ring(Cell::new(0.0));
        let mut z = start();
        let rep = newton_with_projection::<Double, _, _>(
            &pr,
            &mut z,
            &opts(),
            |w| exact::<Double, _>(&pr, w, false),
            move |w: &mut Vec<Vec<f64>>| {
                if shift {
                    w[0][0] = w[0][0].next_up();
                }
                Ok(())
            },
        )
        .unwrap();
        (z, rep.fnorm)
    };
    let pr = Ring(Cell::new(0.0));
    let mut z = start();
    let plain = newton_with::<Double, _, _>(&pr, &mut z, &opts(), |w| exact::<Double, _>(&pr, w, false)).unwrap();
    let bits = |z: &[Vec<f64>], f: &[f64]| -> Vec<u64> { z.iter().flatten().chain(f).map(|x| x.to_bits()).collect() };
    let (zi, fi) = run(false);
    assert_eq!(bits(&zi, &fi), bits(&z, &plain.fnorm));
    let (zs, fs) = run(true);
    assert_ne!(bits(&zs, &fs), bits(&z, &plain.fnorm), "control did not fire");
}
