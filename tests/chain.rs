//! Chained series steps (`anm::chain`) on a planted branch z(p) = sqrt(a + p) of F = (I + kappa C)(z o z - a - p),
//! C the cyclic shift, whose series in p about any p0 is known.
use multisolver::anm::{chain, ChainOpts};
use multisolver::multidouble::linalg::ld_for;
use multisolver::newton::NewtonOpts;
use multisolver::prec::nan_max;
use multisolver::{Double, Jvp, Native, Precision, Problem, ResidualJet, DD};

const N: usize = 8;
const KAPPA: f64 = 0.3;

/// a_i = 1 + 0.25 i: the branch's nearest singularity from p0 is at p = -(a_min) = -1, so the series about p0 has
/// radius 1 + p0
fn a_i(i: usize) -> f64 {
    1.0 + 0.25 * i as f64
}

/// the problem with its parameter's origin at p0
struct Roots<P: Precision<Native>> {
    p0: f64,
    _p: std::marker::PhantomData<P>,
}

fn sc<P: Precision<Native>>(x: f64) -> P::Scalar {
    P::from_dd(DD::from_f64(x))
}

fn scalars<P: Precision<Native>>(z: &[&[f64]]) -> Vec<P::Scalar> {
    (0..z[0].len())
        .map(|i| P::from_raw(&z.iter().map(|p| p[i]).collect::<Vec<_>>()))
        .collect()
}

impl<P: Precision<Native>> Roots<P> {
    fn at(p0: f64) -> Self {
        Roots {
            p0,
            _p: std::marker::PhantomData,
        }
    }
    /// (I + kappa C) u
    fn couple(u: &[P::Scalar]) -> Vec<P::Scalar> {
        (0..N).map(|i| u[i] + sc::<P>(KAPPA) * u[(i + 1) % N]).collect()
    }
}

impl<P: Precision<Native>> Problem<P> for Roots<P> {
    fn len(&self) -> usize {
        N
    }
    fn residual(&self, z: &[&[f64]], _par: bool) -> Result<Vec<P::Scalar>, String> {
        let z = scalars::<P>(z);
        let u: Vec<P::Scalar> = (0..N).map(|i| z[i] * z[i] - sc::<P>(a_i(i) + self.p0)).collect();
        Ok(Self::couple(&u))
    }
    fn jacobian(&self, z: &[&[f64]], _par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
        let z = scalars::<P>(z);
        let ld = ld_for(N);
        let mut c = vec![vec![0.0; ld * N]; P::PARTS];
        for i in 0..N {
            let j = (i + 1) % N;
            let (d, e) = (z[i] + z[i], sc::<P>(KAPPA) * (z[j] + z[j]));
            for (q, cq) in c.iter_mut().enumerate() {
                cq[i * ld + i] = P::part(d, q);
                cq[j * ld + i] = P::part(e, q);
            }
        }
        Ok((c, ld))
    }
}

impl<P: Precision<Native>> Jvp<P> for Roots<P> {
    fn jvp(&self, z: &[&[f64]], v: &[&[f64]], _par: bool) -> Result<Vec<P::Scalar>, String> {
        let (z, v) = (scalars::<P>(z), scalars::<P>(v));
        let u: Vec<P::Scalar> = (0..N).map(|i| (z[i] + z[i]) * v[i]).collect();
        Ok(Self::couple(&u))
    }
}

impl<P: Precision<Native>> ResidualJet<P> for Roots<P> {
    fn residual_jet(&self, zs: &[Vec<&[f64]>], nout: usize, _par: bool) -> Result<Vec<Vec<P::Scalar>>, String> {
        let z: Vec<Vec<P::Scalar>> = zs.iter().map(|v| scalars::<P>(v)).collect();
        let at = |k: usize, i: usize| if k < z.len() { z[k][i] } else { P::ZERO };
        Ok((0..nout)
            .map(|k| {
                let u: Vec<P::Scalar> = (0..N)
                    .map(|i| {
                        let mut s = (0..=k).fold(P::ZERO, |t, j| t + at(j, i) * at(k - j, i));
                        if k == 0 {
                            s = s - sc::<P>(a_i(i) + self.p0);
                        }
                        if k == 1 {
                            s = s - P::ONE;
                        }
                        s
                    })
                    .collect();
                Self::couple(&u)
            })
            .collect())
    }
}

/// the chain from sqrt(a) at p = 0 to p_max = 3 in double-double: (attempts, max |z - sqrt(a + p_end)|, p_end)
fn run(eps: f64, h_max: f64, halvings: usize, newton_iter: usize) -> (Vec<multisolver::anm::ChainStep>, f64, f64) {
    let mut z = vec![(0..N).map(|i| a_i(i).sqrt()).collect::<Vec<_>>(), vec![0.0; N]];
    // the double-double root at p = 0
    let pr0 = Roots::<Double>::at(0.0);
    let polish = NewtonOpts {
        tol: 1e-30,
        par: false,
        ..Default::default()
    };
    multisolver::newton::newton::<Double, _>(&pr0, &mut z, &polish).unwrap();
    let o = ChainOpts {
        order: 12,
        eps,
        h_max,
        halvings,
        p_max: 3.0,
        steps: 40,
        newton: NewtonOpts {
            tol: 1e-29,
            max_iter: newton_iter,
            par: false,
            ..Default::default()
        },
        par: false,
        ..Default::default()
    };
    let steps = chain::<Double, _, _>(Roots::<Double>::at, &mut z, 0.0, &o).unwrap();
    let p_end = steps
        .iter()
        .filter(|s| s.accepted)
        .map(|s| s.p + s.a)
        .fold(0.0, f64::max);
    let err = (0..N)
        .map(|i| {
            let want = (DD::from_f64(a_i(i)) + DD::from_f64(p_end)).sqrt();
            (DD::new(z[0][i], z[1][i]) - want).hi.abs()
        })
        .fold(0.0, nan_max);
    (steps, err, p_end)
}

/// Pass, written before the first run: with eps 1e-16 and h_max 1 the chain reaches p = 3 with its end within 1e-28
/// of sqrt(a + 3), every step within Cochelin's bound and every accepted prediction's max |F| at most 1e-10. The
/// control: eps 1e10 (the bound off) takes steps of h_max = 1 at the series' radius, whose first prediction must
/// exceed 1e-10.
#[test]
fn chain_follows_a_planted_branch() {
    let (steps, err, p_end) = run(1e-16, 1.0, 3, 12);
    let worst = steps.iter().filter(|s| s.accepted).map(|s| s.f_pred).fold(0.0, nan_max);
    println!(
        "chain: {} attempts to p {p_end}, end within {err:.3e}, steps {:?}, worst prediction {worst:.3e}",
        steps.len(),
        steps.iter().map(|s| (s.a * 1e4).round() / 1e4).collect::<Vec<_>>()
    );
    assert!((p_end - 3.0).abs() < 1e-12 && err <= 1e-28, "{p_end} {err:e}");
    assert!(steps.iter().all(|s| s.a <= s.a_max * (1.0 + 1e-15) && s.accepted));
    assert!(worst <= 1e-10, "{worst:e}");
    let (c, _, _) = run(1e10, 1.0, 3, 12);
    println!(
        "control, bound off: first step {}, prediction {:.3e}",
        c[0].a, c[0].f_pred
    );
    assert!(c[0].f_pred > 1e-10, "control did not fire");
}

/// Pass, written before the first run: with the bound off, steps of up to 2.5 and Newton allowed 5 steps, a refused
/// attempt is followed by one at half its step from the same base, and the chain still reaches p = 3 within 1e-28.
/// The control: with no halvings the chain must stop at its first refusal, short of p = 3.
#[test]
fn chain_halves_a_refused_step() {
    let (steps, err, p_end) = run(1e10, 2.5, 3, 5);
    let refused: Vec<usize> = (0..steps.len()).filter(|&k| !steps[k].accepted).collect();
    println!(
        "halving: {} attempts ({} refused) to p {p_end}, end within {err:.3e}",
        steps.len(),
        refused.len()
    );
    assert!(!refused.is_empty(), "no attempt was refused");
    for &k in &refused {
        assert!(k + 1 < steps.len() && steps[k + 1].p == steps[k].p && steps[k + 1].a == 0.5 * steps[k].a);
    }
    assert!((p_end - 3.0).abs() < 1e-12 && err <= 1e-28);
    let (c, _, pc) = run(1e10, 2.5, 0, 5);
    println!("control, no halvings: {} attempts, stopped at p {pc}", c.len());
    assert!(pc < 3.0 && !c.last().unwrap().accepted, "control did not fire");
}
