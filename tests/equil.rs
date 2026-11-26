//! Row and column scalings (`equil`) on planted matrices A = D1 B D2 whose equilibrated form B is known.
use multisolver::equil::{curtis_reid, curtis_reid_ruiz, ruiz, EquilOpts, Scaling};

const N: usize = 12;

/// the planted scalings: d1_i = 2^(3 i - 17) (1 + 0.1 i), d2_j = 10^(j - 6) / (1 + 0.05 j)
fn planted_d() -> (Vec<f64>, Vec<f64>) {
    let d1 = (0..N)
        .map(|i| 2f64.powi(3 * i as i32 - 17) * (1.0 + 0.1 * i as f64))
        .collect();
    let d2 = (0..N)
        .map(|j| 10f64.powi(j as i32 - 6) / (1.0 + 0.05 * j as f64))
        .collect();
    (d1, d2)
}

/// A = D1 B D2 column-major with leading dimension ld = N + 4 (padding rows that must not be read)
fn planted(b: impl Fn(usize, usize) -> f64) -> (Vec<f64>, usize) {
    let (d1, d2) = planted_d();
    let ld = N + 4;
    let mut a = vec![f64::NAN; ld * N];
    for j in 0..N {
        for i in 0..N {
            a[j * ld + i] = d1[i] * b(i, j) * d2[j];
        }
    }
    (a, ld)
}

/// the largest |1 - max-norm| over the rows and columns of D_r |A| D_c
fn deviation(a: &[f64], ld: usize, s: &Scaling) -> f64 {
    let (mut r, mut c) = (vec![0.0f64; N], vec![0.0f64; N]);
    for j in 0..N {
        for i in 0..N {
            let v = s.dr[i] * a[j * ld + i].abs() * s.dc[j];
            r[i] = r[i].max(v);
            c[j] = c[j].max(v);
        }
    }
    r.iter().chain(&c).map(|&x| (1.0 - x).abs()).fold(0.0, f64::max)
}

/// B with every |b_ij| = 1 and alternating signs
fn unit(i: usize, j: usize) -> f64 {
    if (i + 2 * j).is_multiple_of(3) {
        -1.0
    } else {
        1.0
    }
}

/// B with entries in [0.1, 1] of no pattern of scalings
fn varied(i: usize, j: usize) -> f64 {
    0.1 + 0.9 * (((7 * i + 3 * j * j + 5) % 17) as f64 / 16.0)
}

/// Pass, written before the first run: on A = D1 B D2 with |b_ij| = 1, Curtis and Reid's fit is exact: D_r |A| D_c has
/// every entry 1 within 1e-12 (its deviation), and D_r d1 is one constant within 1e-12 relative. The control: the rows
/// alone (cols false) leave a deviation above 0.1.
#[test]
fn curtis_reid_recovers_a_planted_scaling() {
    let (a, ld) = planted(unit);
    let s = curtis_reid(N, N, ld, &a, &EquilOpts::default());
    let e = deviation(&a, ld, &s);
    let (d1, _) = planted_d();
    let k: Vec<f64> = s.dr.iter().zip(&d1).map(|(x, y)| x * y).collect();
    let spread = k.iter().map(|&x| (x / k[0] - 1.0).abs()).fold(0.0, f64::max);
    let rows = curtis_reid(
        N,
        N,
        ld,
        &a,
        &EquilOpts {
            cols: false,
            ..Default::default()
        },
    );
    let c = deviation(&a, ld, &rows);
    println!("Curtis-Reid on the unit planted matrix: deviation {e:.3e}, D_r d1 spread {spread:.3e}; control, rows only {c:.3e}");
    assert!(e <= 1e-12 && spread <= 1e-12 && s.err <= 1e-12, "{e:e} {spread:e}");
    assert!(c > 0.1, "control did not fire");
}

/// Pass, written before the first run: Ruiz's iteration on A = D1 B D2 with b_ij in [0.1, 1] reaches a deviation
/// below 1e-12 within 200 steps, from unit scalings and from Curtis and Reid's; the control: one step leaves more than
/// 1e-3.
#[test]
fn ruiz_equilibrates_rows_and_columns() {
    let (a, ld) = planted(varied);
    let o = EquilOpts::default();
    let (r, cr) = (ruiz(N, N, ld, &a, &o), curtis_reid_ruiz(N, N, ld, &a, &o));
    let (er, ecr) = (deviation(&a, ld, &r), deviation(&a, ld, &cr));
    let one = ruiz(N, N, ld, &a, &EquilOpts { it_max: 1, ..o });
    let c = deviation(&a, ld, &one);
    println!(
        "Ruiz: {er:.3e} in {} steps; from Curtis-Reid {ecr:.3e} in {}; control, one step {c:.3e}",
        r.iterations, cr.iterations
    );
    assert!(er <= 1e-12 && ecr <= 1e-12 && r.iterations < 200 && cr.iterations < 200);
    assert!(c > 1e-3, "control did not fire");
}

/// B with |b_ij| = 1 except 1e-25 at (i, j) with (i + j) % 5 == 1: entries negligible in their rows and columns
fn with_negligible(i: usize, j: usize) -> f64 {
    if (i + j) % 5 == 1 {
        1e-25
    } else {
        unit(i, j)
    }
}

/// Pass, written before the first run: with negligible entries (1e-25 among ones), Curtis and Reid's fit with floor
/// 1e-12 weights them down and leaves a deviation below 1e-5, where the plain fit, steered by their logarithms, leaves
/// one above 1 (the control).
#[test]
fn curtis_reid_floor_ignores_negligible_entries() {
    let (a, ld) = planted(with_negligible);
    let floor = curtis_reid(
        N,
        N,
        ld,
        &a,
        &EquilOpts {
            floor: 1e-12,
            ..Default::default()
        },
    );
    let plain = curtis_reid(N, N, ld, &a, &EquilOpts::default());
    let (ef, ep) = (deviation(&a, ld, &floor), deviation(&a, ld, &plain));
    println!(
        "with negligible entries: floor 1e-12 {ef:.3e} after {} refits; control, no floor {ep:.3e}",
        floor.iterations
    );
    assert!(ef <= 1e-5, "{ef:e}");
    assert!(ep > 1.0, "control did not fire");
}

/// the row and column max-norms of D_r |A| D_c
fn norms(a: &[f64], ld: usize, s: &Scaling) -> Vec<f64> {
    let (mut r, mut c) = (vec![0.0f64; N], vec![0.0f64; N]);
    for j in 0..N {
        for i in 0..N {
            let v = s.dr[i] * a[j * ld + i].abs() * s.dc[j];
            r[i] = r[i].max(v);
            c[j] = c[j].max(v);
        }
    }
    r.into_iter().chain(c).collect()
}

/// Pass, written before the first run: the scalings rounded to powers of two are powers of two and keep every row's
/// and column's max-norm in [0.5, 2]. The controls: the unrounded ones are not all powers of two, and the rounded row
/// factors times 2^-100, still powers of two, must fail the [0.5, 2] check.
#[test]
fn pow2_scalings_are_exact() {
    let (a, ld) = planted(varied);
    let s = ruiz(N, N, ld, &a, &EquilOpts::default());
    let p = s.pow2();
    let is_pow2 = |x: f64| x > 0.0 && x.to_bits() & ((1u64 << 52) - 1) == 0;
    assert!(p.dr.iter().chain(&p.dc).all(|&x| is_pow2(x)));
    let within = |v: &[f64]| v.iter().all(|&x| (0.5..=2.0).contains(&x));
    let nv = norms(&a, ld, &p);
    let (lo, hi) = nv
        .iter()
        .fold((f64::INFINITY, 0.0f64), |(a, b), &x| (a.min(x), b.max(x)));
    println!("powers of two: max-norms in [{lo:.3}, {hi:.3}]");
    assert!(within(&nv), "{lo} {hi}");
    assert!(!s.dr.iter().chain(&s.dc).all(|&x| is_pow2(x)), "control did not fire");
    let mut low = p.clone();
    low.dr.iter_mut().for_each(|x| *x *= 2f64.powi(-100));
    assert!(
        low.dr.iter().all(|&x| is_pow2(x)) && !within(&norms(&a, ld, &low)),
        "control did not fire"
    );
}

/// Pass, written before the first run: scalings near the ends of f64's range stay finite. pow2 of a factor of
/// 1.43e308 is 2^1023 (2^1024 is infinite); the matrix with rows (1e300, 1e-300, 1e-300) and (1e300, 1e300, 1e300)
/// twice reaches Ruiz's tolerance from Curtis and Reid's scaling with every factor finite and nonzero; and a 1 x 32
/// row of 1e-320 gets factors near 1e160 with its scaled entries 1 within 1e-12. The products' range is the unit
/// tests' (`equil::tests`), with their control.
#[test]
fn scalings_stay_in_range() {
    let big = Scaling {
        dr: vec![1.43e308],
        dc: vec![1.0],
        iterations: 0,
        err: 0.0,
    };
    assert_eq!(big.pow2().dr[0], 2f64.powi(1023));
    assert!(2f64.powi(1023) * 2.0 == f64::INFINITY);
    let ld = 3;
    let a = [1e300, 1e300, 1e300, 1e-300, 1e300, 1e300, 1e-300, 1e300, 1e300];
    let s = curtis_reid_ruiz(3, 3, ld, &a, &EquilOpts::default());
    println!(
        "extreme 3 x 3: dr {:?}, dc {:?}, deviation {:.3e} in {} steps",
        s.dr, s.dc, s.err, s.iterations
    );
    assert!(s.dr.iter().chain(&s.dc).all(|&x| x.is_finite() && x > 0.0) && s.err <= 1e-12);
    let row = vec![1e-320; 32];
    let t = curtis_reid(1, 32, 1, &row, &EquilOpts::default());
    let worst =
        t.dc.iter()
            .map(|&c| (t.dr[0] * 1e-320 * c - 1.0).abs())
            .fold(0.0, f64::max);
    println!(
        "1 x 32 of 1e-320: dr {:.3e}, dc {:.3e}, scaled entries within {worst:.3e} of 1",
        t.dr[0], t.dc[0]
    );
    assert!(t.dr[0].is_finite() && t.dc.iter().all(|&c| c.is_finite()) && worst <= 1e-12);
}
