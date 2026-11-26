//! The type of a real singularity (`singularity::ratio_exponent`, `dlog_exponent`) and quadratic Pade approximants
//! (`singularity::Shafer`), on planted series f = A (1 - h/s)^mu and A + B sqrt(1 - h/s) + C h.
use multisolver::multidouble::qd::QD;
use multisolver::singularity::{dlog_exponent, ratio_exponent, Cx, DlogOpts, Shafer};
use multisolver::{Double, Native, Precision, Quad, DD};

/// c_1..c_N of (1 - h/s)^mu in P: binom(mu, k) (-1/s)^k, by b_k = b_(k-1) (mu - k + 1) / k, with `inv` the
/// reciprocal of an f64 in P
fn plant<P: Precision<Native>>(mu: f64, s: f64, n: usize, inv: impl Fn(f64) -> P::Scalar) -> Vec<P::Scalar> {
    let d = |x: f64| P::from_dd(DD::from_f64(x));
    let minus_inv_s = -inv(s);
    let (mut b, mut pw) = (P::ONE, P::ONE);
    (1..=n)
        .map(|k| {
            b = b * (d(mu) - d((k - 1) as f64)) * inv(k as f64);
            pw = pw * minus_inv_s;
            b * pw
        })
        .collect()
}

/// (|s / s* - 1|, |mu - mu*|) of an estimate
fn off(e: &multisolver::singularity::Exponent, s: f64, mu: f64) -> (f64, f64) {
    ((e.s / s - 1.0).abs(), (e.mu - mu).abs())
}

/// Pass, written before the first run: on the clean plants of a square-root point (mu = 1/2) and a simple pole
/// (mu = -1) at s = 1.17 with 24 coefficients, the ratio fit over first orders 8..14 gives s and mu within 1e-9, and
/// the Dlog Pade approximants (M = 3..9) in double-double within 1e-12. The control: the plants built at s + 0.5 and
/// judged against s miss s by more than 0.1 with both estimators.
#[test]
fn exponents_of_planted_singularities() {
    for mu in [0.5, -1.0] {
        let c = plant::<Double>(mu, 1.17, 24, |x| DD::ONE / DD::from_f64(x));
        let c64: Vec<f64> = c.iter().map(|v| v.hi).collect();
        let r = ratio_exponent(&c64, 8..15).unwrap();
        let d = dlog_exponent::<Double>(&c, &DlogOpts::default()).unwrap();
        let (rs, rm) = off(&r, 1.17, mu);
        let (ds, dm) = off(&d, 1.17, mu);
        println!(
            "mu {mu}: ratio s {:.12} mu {:.12} over {}; dlog s {:.15} mu {:.15} over {}",
            r.s, r.mu, r.n, d.s, d.mu, d.n
        );
        assert!(rs <= 1e-9 && rm <= 1e-9, "ratio: {rs:e} {rm:e}");
        assert!(ds <= 1e-12 && dm <= 1e-12, "dlog: {ds:e} {dm:e}");
        let cm = plant::<Double>(mu, 1.67, 24, |x| DD::ONE / DD::from_f64(x));
        let cm64: Vec<f64> = cm.iter().map(|v| v.hi).collect();
        let (mr, md) = (
            ratio_exponent(&cm64, 8..15).unwrap(),
            dlog_exponent::<Double>(&cm, &DlogOpts::default()).unwrap(),
        );
        println!("control, built at 1.67: ratio s {:.4}, dlog s {:.4}", mr.s, md.s);
        assert!(
            off(&mr, 1.17, mu).0 > 0.1 && off(&md, 1.17, mu).0 > 0.1,
            "control did not fire"
        );
    }
}

const A: f64 = 1.6323;
const BB: f64 = 0.5;
const C: f64 = -0.2;
const S: f64 = 1.25;

/// f_0..f_(n-1) of A + B sqrt(1 - h/s) + C h in quad-double
fn fold_series(n: usize) -> Vec<<Quad as Precision<Native>>::Scalar> {
    let mut sq = plant::<Quad>(0.5, S, n - 1, |x| QD::from_f64(1.0) / QD::from_f64(x));
    sq.insert(0, <Quad as Precision<Native>>::ONE);
    let q = |x: f64| <Quad as Precision<Native>>::from_dd(DD::from_f64(x));
    (0..n)
        .map(|k| {
            let mut v = q(BB) * sq[k];
            if k == 0 {
                v = v + q(A);
            }
            if k == 1 {
                v = v + q(C);
            }
            v
        })
        .collect()
}

/// (|branch point - s|, the largest distance of the sheets at h = 0.6 from the branch A + C h + B sqrt(1 - h/s) and
/// its mirror A + C h - B sqrt(1 - h/s)) of the approximant of degrees deg, in quad-double
fn shafer_err(deg: (usize, usize, usize)) -> (f64, f64) {
    let f = fold_series(12);
    let sh = Shafer::<Quad>::new(&f, deg).unwrap();
    let s_found = sh
        .branch_points()
        .into_iter()
        .min_by(|a, b| a.abs().total_cmp(&b.abs()))
        .unwrap();
    let q = |x: f64| <Quad as Precision<Native>>::from_dd(DD::from_f64(x));
    let e_s = (s_found - Cx::<Quad>::real(q(S))).abs();
    let h = q(0.6);
    let root = (q(1.0) - h / q(S)).sqrt();
    let base = q(A) + q(C) * h;
    let (want_p, want_m) = (base + q(BB) * root, base - q(BB) * root);
    let (a, b) = sh.sheets(Cx::real(h));
    let d = |x: Cx<Quad>, w: <Quad as Precision<Native>>::Scalar| (x - Cx::real(w)).abs();
    let e = d(a, want_p).min(d(b, want_p)).max(d(a, want_m).min(d(b, want_m)));
    (e_s, e)
}

/// Pass, written before the first run: the quadratic Pade approximant of degrees (0, 1, 2), which
/// A + B sqrt(1 - h/s) + C h satisfies exactly ((f - A - C h)^2 = B^2 (1 - h/s)), places the branch point at s = 1.25
/// within 1e-50 and gives both the branch and its mirror at h = 0.6 within 1e-50, in quad-double. The control:
/// degrees (0, 0, 2), whose constant Q cannot carry C h, miss the mirror by more than 1e-6.
#[test]
fn shafer_continues_through_a_fold() {
    let (es, e) = shafer_err((0, 1, 2));
    let (_, ec) = shafer_err((0, 0, 2));
    println!("Shafer (0, 1, 2): branch point {es:.3e}, sheets {e:.3e}; control, (0, 0, 2): sheets {ec:.3e}");
    assert!(es <= 1e-50 && e <= 1e-50, "{es:e} {e:e}");
    assert!(ec > 1e-6, "control did not fire");
}
