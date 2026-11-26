//! The singularity locator on planted series with known answers, in f64 (K = 30 orders), double-double (K = 60) and
//! quad-double (K = 120). The pass criteria were fixed before the first run (negative-mass-raw, I191-rpi5/README.md).
//!
//! A planted series is the sum of parts c (1 - u / s)^alpha (a conjugate pair of parts, 2 Re of one, when Im s > 0),
//! with the coefficient recurrence c_k = c_(k-1) (k - 1 - alpha) / (k s) evaluated in the lane's arithmetic, then
//!
//! * dressed: multiplied by 1 / (1 + (u / tau)^2), a pair of poles at +- i tau farther out (tau = 4 |s|), which
//!   corrects the coefficients algebraically in 1 / k;
//! * noisy: every coefficient multiplied by 1 + delta xi, xi xorshift noise of unit variance, delta = 1e-10, 1e-19,
//!   1e-38 in f64, double-double and quad-double.
//!
//! The cases (nearest singularity first; the series about u = 0, the nearest singularity s, exponent alpha, amplitude
//! c):
//!
//! * P1 to P6, pure real: (c, alpha, s) = (1.7, -1, 2), (-0.8, -2, -1.5), (1.3, 1/2, 2.5), (0.9, -1/2, -1.75), (2.2, 1/3,
//!   1.25), (1.1, 1e-8, 2);
//! * D1 to D6, the same six dressed with tau = 4 |s|;
//! * C1 to C3, pure conjugate pairs: (c, alpha, s) = (1.7, -1, 1 + 1.25 i), (1.3, 1/2, 0.9 + 1.4 i), (1.1, -1/2,
//!   1.59 + 0.16 i), the last at 5.7 degrees from the real axis; C4 and C5, C1 and C2 dressed with tau = 4 |s|;
//! * T1, T2, T3, a nearest singularity with a farther one of equal amplitude behind it: (1, 1/2, 2) with (1, -1/2,
//!   -2.8) on the other side, (1, 1/2, 2) with (1, -1/2, 2.8) on the same side, and the poles (1, -1, 2) and (1, -1,
//!   3);
//! * N1, N2, the pure cases P3 and C2 with noise; O8, the pure case P6 (alpha = 1e-8, the regime of #183's O8) with
//!   noise in double-double and quad-double and noiseless in f64;
//! * N3, the dressed case D3 with noise: printed, not gated (the interplay of truncation and noise).
//!
//! Criteria, each cell (case, lane) of the table, with the error of s relative to |s| and of alpha absolute:
//!
//! 1. the kind is `Real` for P, D, T, N1 and O8, and `Pair` for C and N2;
//! 2. the error of s is at most `tol_s` and of alpha at most `tol_alpha`, by tier: pure single (P1 to P6) 1e4 u and
//!    1e6 u (u = 2^-52, 2^-104, 2^-208); pure pair and two-pole (C1 to C3, T3) 1e5 u and 1e7 u; dressed (D1 to D6, C4,
//!    C5) 1e-2 and 0.1 in f64, 1e-6 and 1e-4 in double-double, 1e-9 and 1e-7 in quad-double; noisy (N1, O8) the larger
//!    of 2 delta and 1e4 u, and of 70 delta and 1e6 u (N2: 30 delta and 1e5 u, 1300 delta and 1e7 u: ten times the
//!    largest amplification of the noise measured in dev_03b, 0.14 and 6.6 for a single singularity, 2.5 and 127 for
//!    a pair, in units of delta); and for O8 the error of alpha at most 1e-9, a tenth of alpha. T1 and T2 have the
//!    tolerances of the dressed tier;
//! 3. the bars cover: the error of s at most its bar and the error of alpha at most its bar;
//! 4. the bars are useful: the bar of s over |s| at most 100 `tol_s` and the bar of alpha at most 100 `tol_alpha`.
//!
//! The controls: `wrong-sigma`, `wrong-alpha`, `pair-by-ratio`, `bar-shrunk`, `window-early` and `window-short`
//! (README, "Controls"); each feeds a check of this table a wrong input and must fail it.
use multisolver::locate::{locate, locate_at, Kind, Location, Model, Opts, Real};
use multisolver::multidouble::{dd::DD, qd::QD};

#[derive(Clone, Copy, Debug, PartialEq)]
enum Lane {
    F64,
    Dd,
    Qd,
}

impl Lane {
    fn k(self) -> usize {
        match self {
            Lane::F64 => 30,
            Lane::Dd => 60,
            Lane::Qd => 120,
        }
    }
    fn unit(self) -> f64 {
        match self {
            Lane::F64 => <f64 as Real>::UNIT,
            Lane::Dd => <DD as Real>::UNIT,
            Lane::Qd => <QD as Real>::UNIT,
        }
    }
    fn delta(self) -> f64 {
        match self {
            Lane::F64 => 1e-10,
            Lane::Dd => 1e-19,
            Lane::Qd => 1e-38,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Tier {
    Pure,
    PureHard,
    Dressed,
    Weak,
}

#[derive(Clone, Copy, Debug)]
struct Part {
    c: f64,
    alpha: f64,
    s: (f64, f64),
}

#[derive(Clone, Debug)]
struct Case {
    id: &'static str,
    tier: Tier,
    parts: Vec<Part>,
    /// 0 for none, else the poles at +- i tau
    tau: f64,
    noisy: bool,
    /// no noise in f64 (O8)
    noisy_not_f64: bool,
    kind: Kind,
    /// the alpha tolerance of a tenth of alpha
    tenth: bool,
    /// false: the row is printed and not asserted (N3)
    gated: bool,
}

fn real(c: f64, alpha: f64, s: f64) -> Part {
    Part { c, alpha, s: (s, 0.0) }
}

fn pair(c: f64, alpha: f64, a: f64, b: f64) -> Part {
    Part { c, alpha, s: (a, b) }
}

fn modulus(p: &Part) -> f64 {
    p.s.0.hypot(p.s.1)
}

fn cases() -> Vec<Case> {
    let single = [
        real(1.7, -1.0, 2.0),
        real(-0.8, -2.0, -1.5),
        real(1.3, 0.5, 2.5),
        real(0.9, -0.5, -1.75),
        real(2.2, 1.0 / 3.0, 1.25),
        real(1.1, 1e-8, 2.0),
    ];
    let mut v = Vec::new();
    let mk = |id, tier, parts: Vec<Part>, tau, kind| Case {
        id,
        tier,
        parts,
        tau,
        noisy: false,
        noisy_not_f64: false,
        kind,
        tenth: false,
        gated: true,
    };
    for (i, id) in ["P1", "P2", "P3", "P4", "P5", "P6"].into_iter().enumerate() {
        v.push(mk(id, Tier::Pure, vec![single[i]], 0.0, Kind::Real));
    }
    for (i, id) in ["D1", "D2", "D3", "D4", "D5", "D6"].into_iter().enumerate() {
        v.push(mk(
            id,
            Tier::Dressed,
            vec![single[i]],
            4.0 * modulus(&single[i]),
            Kind::Real,
        ));
    }
    let pairs = [
        pair(1.7, -1.0, 1.0, 1.25),
        pair(1.3, 0.5, 0.9, 1.4),
        pair(1.1, -0.5, 1.59, 0.16),
    ];
    for (i, id) in ["C1", "C2", "C3"].into_iter().enumerate() {
        v.push(mk(id, Tier::PureHard, vec![pairs[i]], 0.0, Kind::Pair));
    }
    v.push(mk(
        "C4",
        Tier::Dressed,
        vec![pairs[0]],
        4.0 * modulus(&pairs[0]),
        Kind::Pair,
    ));
    v.push(mk(
        "C5",
        Tier::Dressed,
        vec![pairs[1]],
        4.0 * modulus(&pairs[1]),
        Kind::Pair,
    ));
    v.push(mk(
        "T1",
        Tier::Dressed,
        vec![real(1.0, 0.5, 2.0), real(1.0, -0.5, -2.8)],
        0.0,
        Kind::Real,
    ));
    v.push(mk(
        "T2",
        Tier::Dressed,
        vec![real(1.0, 0.5, 2.0), real(1.0, -0.5, 2.8)],
        0.0,
        Kind::Real,
    ));
    v.push(mk(
        "T3",
        Tier::PureHard,
        vec![real(1.0, -1.0, 2.0), real(1.0, -1.0, 3.0)],
        0.0,
        Kind::Real,
    ));
    let mut n1 = mk("N1", Tier::Weak, vec![single[2]], 0.0, Kind::Real);
    n1.noisy = true;
    let mut n2 = mk("N2", Tier::Weak, vec![pairs[1]], 0.0, Kind::Pair);
    n2.noisy = true;
    let mut o8 = mk("O8", Tier::Weak, vec![single[5]], 0.0, Kind::Real);
    o8.noisy = true;
    o8.noisy_not_f64 = true;
    o8.tenth = true;
    let mut n3 = mk(
        "N3",
        Tier::Dressed,
        vec![single[2]],
        4.0 * modulus(&single[2]),
        Kind::Real,
    );
    n3.noisy = true;
    n3.gated = false;
    v.extend([n1, n2, o8, n3]);
    v
}

/// (sigma tolerance relative, alpha tolerance absolute)
fn tolerances(case: &Case, lane: Lane) -> (f64, f64) {
    let u = lane.unit();
    let d = if case.noisy && !(case.noisy_not_f64 && lane == Lane::F64) {
        lane.delta()
    } else {
        0.0
    };
    let pair = case.kind == Kind::Pair;
    match case.tier {
        Tier::Pure => (1e4 * u, 1e6 * u),
        Tier::PureHard => (1e5 * u, 1e7 * u),
        Tier::Dressed => match lane {
            Lane::F64 => (1e-2, 0.1),
            Lane::Dd => (1e-6, 1e-4),
            Lane::Qd => (1e-9, 1e-7),
        },
        Tier::Weak => {
            if pair {
                ((30.0 * d).max(1e5 * u), (1300.0 * d).max(1e7 * u))
            } else {
                ((2.0 * d).max(1e4 * u), (70.0 * d).max(1e6 * u))
            }
        }
    }
}

/// a deterministic sequence of unit variance (xorshift, uniform on [-sqrt 3, sqrt 3))
fn noise(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    ((*state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) * 3f64.sqrt()
}

/// a wrong input of a control: the nearest singularity planted at s (1 + sigma_scale) and alpha + alpha_shift, formed
/// in the lane's arithmetic (the checks keep the right values)
#[derive(Clone, Copy, Debug, Default)]
struct Wrong {
    sigma_scale: f64,
    alpha_shift: f64,
}

/// the coefficients c_0 ... c_k of c (1 - u / s)^alpha (2 Re of it for a pair) by the recurrence in T
fn part_series<T: Real>(p: &Part, k: usize, w: Wrong) -> Vec<T> {
    let scale = T::from_f64(1.0) + T::from_f64(w.sigma_scale);
    let (a, b) = (T::from_f64(p.s.0) * scale, T::from_f64(p.s.1) * scale);
    let n2 = a * a + b * b;
    // 1 / s = (a - i b) / |s|^2
    let (ire, iim) = (a / n2, -(b / n2));
    let alpha = T::from_f64(p.alpha) + T::from_f64(w.alpha_shift);
    let amp = T::from_f64(p.c);
    let is_pair = p.s.1 != 0.0;
    let (mut re, mut im) = (amp, T::from_f64(0.0));
    let mut out = vec![if is_pair { amp + amp } else { amp }];
    for j in 1..=k {
        let f = (T::from_f64((j - 1) as f64) - alpha) / T::from_f64(j as f64);
        let (fr, fi) = (f * ire, f * iim);
        let (nr, ni) = (re * fr - im * fi, re * fi + im * fr);
        (re, im) = (nr, ni);
        out.push(if is_pair { nr + nr } else { nr });
    }
    out
}

/// the planted series of a case in the lane's arithmetic, noise included when the case has it in this lane
fn series<T: Real>(case: &Case, lane: Lane, k: usize, w: Wrong) -> Vec<T> {
    let mut c = vec![T::from_f64(0.0); k + 1];
    for (i, p) in case.parts.iter().enumerate() {
        for (x, y) in c
            .iter_mut()
            .zip(part_series::<T>(p, k, if i == 0 { w } else { Wrong::default() }))
        {
            *x = *x + y;
        }
    }
    if case.tau > 0.0 {
        // 1 / (1 + (u / tau)^2) = sum_i (-1)^i (u / tau)^(2 i)
        let tau = T::from_f64(case.tau);
        let m = -(T::from_f64(1.0) / (tau * tau));
        let mut h = vec![T::from_f64(0.0); k + 1];
        let mut pw = T::from_f64(1.0);
        for i in (0..=k).step_by(2) {
            h[i] = pw;
            pw = pw * m;
        }
        c = (0..=k)
            .map(|n| (0..=n).fold(T::from_f64(0.0), |s, i| s + c[i] * h[n - i]))
            .collect();
    }
    if case.noisy && !(case.noisy_not_f64 && lane == Lane::F64) {
        let mut st = 0x9E37_79B9_7F4A_7C15u64;
        let delta = T::from_f64(lane.delta());
        for x in c.iter_mut() {
            *x = *x * (T::from_f64(1.0) + delta * T::from_f64(noise(&mut st)));
        }
    }
    c
}

/// the row of one cell
#[derive(Clone, Debug)]
struct Row {
    id: &'static str,
    lane: Lane,
    kind: Kind,
    want_kind: Kind,
    model: String,
    first: usize,
    degree: usize,
    err_s: f64,
    err_a: f64,
    bar_s_rel: f64,
    bar_a: f64,
    tol_s: f64,
    tol_a: f64,
    tenth: bool,
    skew: f64,
    /// the farther singularity: (error of s relative to its modulus, bar of s over |s|, error of alpha, bar of alpha)
    far: Option<(f64, f64, f64, f64)>,
}

fn describe<T: Real>(loc: &Location<T>, case: &Case, lane: Lane) -> Row {
    let p = &case.parts[0];
    let (tol_s, tol_a) = tolerances(case, lane);
    let n = &loc.nearest;
    let err_s = (n.re - T::from_f64(p.s.0))
        .to_f64()
        .hypot((n.im - T::from_f64(p.s.1)).to_f64())
        / modulus(p);
    let err_a = (n.alpha - T::from_f64(p.alpha)).to_f64().abs();
    let far = (case.parts.len() > 1)
        .then_some(loc.farther.as_ref())
        .flatten()
        .map(|f| {
            let q = &case.parts[1];
            (
                (f.re - T::from_f64(q.s.0))
                    .to_f64()
                    .hypot((f.im - T::from_f64(q.s.1)).to_f64())
                    / modulus(q),
                f.bar / modulus(q),
                (f.alpha - T::from_f64(q.alpha)).to_f64().abs(),
                f.alpha_bar,
            )
        });
    Row {
        id: case.id,
        lane,
        kind: loc.kind,
        want_kind: case.kind,
        model: format!("{:?}", loc.model),
        first: loc.first,
        degree: loc.degree,
        err_s,
        err_a,
        bar_s_rel: n.bar / modulus(p),
        bar_a: n.alpha_bar,
        tol_s,
        tol_a,
        tenth: case.tenth,
        skew: loc.skew,
        far,
    }
}

fn run_cell<T: Real>(case: &Case, lane: Lane) -> Row {
    let c = series::<T>(case, lane, lane.k(), Wrong::default());
    let loc = locate(&c, &Default::default()).expect("no location");
    describe(&loc, case, lane)
}

fn run(case: &Case, lane: Lane) -> Row {
    match lane {
        Lane::F64 => run_cell::<f64>(case, lane),
        Lane::Dd => run_cell::<DD>(case, lane),
        Lane::Qd => run_cell::<QD>(case, lane),
    }
}

/// what a control does to one cell: the series of a wrong input located (all of `locate`, or one cell of `locate_at`),
/// the row measured against the right truth, then `post` applied to the row; None when the locator finds nothing
#[derive(Clone, Copy)]
struct Control {
    name: &'static str,
    wrong: Wrong,
    /// Some((model, first fraction of K, degree)): `locate_at`
    at: Option<(Model, f64, usize)>,
    /// the bars multiplied by this
    bar_scale: f64,
}

fn run_control_cell<T: Real>(case: &Case, lane: Lane, c: &Control) -> Option<Row> {
    let series = series::<T>(case, lane, lane.k(), c.wrong);
    let opts = Opts::default();
    let loc = match c.at {
        None => locate(&series, &opts)?,
        Some((model, frac, degree)) => {
            locate_at(&series, model, (frac * lane.k() as f64).round() as usize, degree, &opts)?
        }
    };
    let mut r = describe(&loc, case, lane);
    r.bar_s_rel *= c.bar_scale;
    r.bar_a *= c.bar_scale;
    Some(r)
}

fn run_control(case: &Case, lane: Lane, c: &Control) -> Option<Row> {
    match lane {
        Lane::F64 => run_control_cell::<f64>(case, lane, c),
        Lane::Dd => run_control_cell::<DD>(case, lane, c),
        Lane::Qd => run_control_cell::<QD>(case, lane, c),
    }
}

/// x <= bound, false for a NaN
fn within(x: f64, bound: f64) -> bool {
    x <= bound
}

/// the criteria a row fails
fn failed(r: &Row) -> Vec<&'static str> {
    let mut f = Vec::new();
    if r.kind != r.want_kind {
        f.push("1 kind");
    }
    if !within(r.err_s, r.tol_s) || !within(r.err_a, r.tol_a) || (r.tenth && !within(r.err_a, 1e-9)) {
        f.push("2 tolerance");
    }
    if !within(r.err_s, r.bar_s_rel) || !within(r.err_a, r.bar_a) {
        f.push("3 bar covers");
    }
    if !within(r.bar_s_rel, 100.0 * r.tol_s) || !within(r.bar_a, 100.0 * r.tol_a) {
        f.push("4 bar useful");
    }
    f
}

fn print_row(r: &Row) {
    println!(
        "{:>3} {:>4} {:?} {:<9} first {:>3} deg {:>2} | s: err {:.1e} bar {:.1e} tol {:.1e} | alpha: err {:.1e} bar {:.1e} tol {:.1e} | skew {:.0e}{}{}",
        r.id,
        format!("{:?}", r.lane),
        r.kind,
        r.model,
        r.first,
        r.degree,
        r.err_s,
        r.bar_s_rel,
        r.tol_s,
        r.err_a,
        r.bar_a,
        r.tol_a,
        r.skew,
        r.far
            .map(|f| format!(" | farther: s err {:.1e} bar {:.1e}, alpha err {:.1e} bar {:.1e}", f.0, f.1, f.2, f.3))
            .unwrap_or_default(),
        if failed(r).is_empty() { String::new() } else { format!("  FAILS {:?}", failed(r)) },
    );
}

/// All cells of the table: every criterion holds in each of the 23 gated cases in each of the 3 lanes (N3 is printed),
/// except D6 in f64, which is **NOT MET** (criteria 1, 2 and 4; see `d6_f64_is_not_met`): 68 of the 69 gated cells meet
/// all four criteria.
#[test]
fn planted_suite() {
    let mut bad = Vec::new();
    for case in cases() {
        for lane in [Lane::F64, Lane::Dd, Lane::Qd] {
            let r = run(&case, lane);
            print_row(&r);
            if case.gated && !(case.id == "D6" && lane == Lane::F64) && !failed(&r).is_empty() {
                bad.push(format!("{} {:?}: {:?}", r.id, r.lane, failed(&r)));
            }
        }
    }
    assert!(bad.is_empty(), "failed cells:\n{}", bad.join("\n"));
}

/// D6 in f64 (alpha = 1e-8, dressed, K = 30) does not meet criteria 1, 2 and 4: the dressing's own coefficients swamp
/// the singular ones up to order about 16 and a clean window would start near 0.8 K, which the default grid (starts 0.2
/// to 0.6 K) does not contain. Criterion 3 holds: the locator says `Unresolved`, with a bar of the order of |s| that
/// covers its error, so it gives no confident wrong answer. This test records that behavior; it does not meet the
/// criteria, and the README, the note and the pull request report the cell as NOT MET.
#[test]
fn d6_f64_is_not_met() {
    let case = cases().into_iter().find(|c| c.id == "D6").unwrap();
    let r = run(&case, Lane::F64);
    print_row(&r);
    assert_eq!(
        failed(&r),
        vec!["1 kind", "2 tolerance", "4 bar useful"],
        "D6 f64 changed"
    );
    assert_eq!(r.kind, Kind::Unresolved);
    assert!(r.bar_s_rel > 0.05, "the bar is above the resolution limit");
}

/// the six controls of the README on one cell
fn controls_of(case: &Case, lane: Lane) -> Vec<Control> {
    let (ts, ta) = tolerances(case, lane);
    let k = lane.k() as f64;
    let at = |model, first: f64, degree| Control {
        name: "",
        wrong: Wrong::default(),
        at: Some((model, first / k, degree)),
        bar_scale: 1.0,
    };
    vec![
        Control {
            name: "wrong-sigma",
            wrong: Wrong {
                sigma_scale: 10.0 * ts,
                alpha_shift: 0.0,
            },
            at: None,
            bar_scale: 1.0,
        },
        Control {
            name: "wrong-alpha",
            wrong: Wrong {
                sigma_scale: 0.0,
                alpha_shift: 10.0 * ta,
            },
            at: None,
            bar_scale: 1.0,
        },
        Control {
            name: "pair-by-ratio",
            ..at(Model::Ratio, 0.5 * k, 2)
        },
        Control {
            name: "bar-shrunk",
            wrong: Wrong::default(),
            at: None,
            bar_scale: 1e-3,
        },
        Control {
            name: "window-early",
            ..at(Model::Ratio, 2.0, 2)
        },
        Control {
            name: "window-short",
            ..at(Model::Ratio, k - 7.0, 2)
        },
    ]
}

/// The development run of the controls (README, "Controls"): what each control does to each cell, printed. Not part of
/// the record: `cargo test --release --test locate controls_dev -- --ignored --nocapture`.
#[test]
#[ignore]
fn controls_dev() {
    for case in cases() {
        for lane in [Lane::F64, Lane::Dd, Lane::Qd] {
            for c in controls_of(&case, lane) {
                match run_control(&case, lane, &c) {
                    None => println!("{:>13} {:>3} {:>4?}: locator finds nothing", c.name, case.id, lane),
                    Some(r) => println!(
                        "{:>13} {:>3} {:>4?}: {:?} {} first {} deg {} err s {:.1e} alpha {:.1e} bar s {:.1e} alpha {:.1e} | fails {:?}",
                        c.name, case.id, lane, r.kind, r.model, r.first, r.degree, r.err_s, r.err_a, r.bar_s_rel, r.bar_a, failed(&r)
                    ),
                }
            }
        }
    }
}

/// whether a control fails a check of the table on one cell; the locator finding nothing counts as failing
fn fires(case: &Case, lane: Lane, name: &str, check: &str) -> bool {
    let c = controls_of(case, lane).into_iter().find(|c| c.name == name).unwrap();
    match run_control(case, lane, &c) {
        None => true,
        Some(r) => failed(&r).iter().any(|f| f.starts_with(check)),
    }
}

const LANES: [Lane; 3] = [Lane::F64, Lane::Dd, Lane::Qd];

/// The controls, each on the cells its development run (`controls_dev`, dev_11; README amendment A3) named, computed
/// before they were declared:
///
/// * `wrong-sigma` and `wrong-alpha` fail check 2 in all 72 cells (24 cases x 3 lanes);
/// * `pair-by-ratio` fails checks 1 and 2 in the 18 cells of the pair cases C1 to C5 and N2;
/// * `bar-shrunk` (bars x 1e-3) fails check 3 in every cell except ten, where it is silent (the error is 0, or at least
///   a thousand times below the bar): P1 f64 and double-double, D1, D2 and C4 in double-double and quad-double, T1 and T2
///   in quad-double; those ten stay silent, and are reported as silent;
/// * `window-early` (ratio model, first order 2, degree 2) fails check 2 on the cases with a farther singularity, T1,
///   T2 and T3, in all three lanes: a window that starts too early fails on a planted case;
/// * `window-short` (ratio model, the last 8 orders, degree 2) fails check 2 on the noisy cases N1 and N2 and on the
///   two poles T3, in all three lanes: a window that is too short fails on a planted case.
#[test]
fn controls_fire() {
    let all = cases();
    let get = |id: &str| all.iter().find(|c| c.id == id).unwrap();
    let mut silent_found = Vec::new();
    for case in &all {
        for lane in LANES {
            assert!(
                fires(case, lane, "wrong-sigma", "2"),
                "wrong-sigma silent on {} {:?}",
                case.id,
                lane
            );
            assert!(
                fires(case, lane, "wrong-alpha", "2"),
                "wrong-alpha silent on {} {:?}",
                case.id,
                lane
            );
            if !fires(case, lane, "bar-shrunk", "3") {
                silent_found.push((case.id, lane));
            }
        }
    }
    let silent = [
        ("P1", Lane::F64),
        ("P1", Lane::Dd),
        ("D1", Lane::Dd),
        ("D1", Lane::Qd),
        ("D2", Lane::Dd),
        ("D2", Lane::Qd),
        ("C4", Lane::Dd),
        ("C4", Lane::Qd),
        ("T1", Lane::Qd),
        ("T2", Lane::Qd),
    ];
    assert_eq!(
        silent_found, silent,
        "bar-shrunk fires on all cells but its ten silent ones"
    );
    for id in ["C1", "C2", "C3", "C4", "C5", "N2"] {
        for lane in LANES {
            assert!(
                fires(get(id), lane, "pair-by-ratio", "1"),
                "pair-by-ratio kind {id} {lane:?}"
            );
            assert!(
                fires(get(id), lane, "pair-by-ratio", "2"),
                "pair-by-ratio tolerance {id} {lane:?}"
            );
        }
    }
    for id in ["T1", "T2", "T3"] {
        for lane in LANES {
            assert!(
                fires(get(id), lane, "window-early", "2"),
                "window-early silent on {id} {lane:?}"
            );
        }
    }
    for id in ["N1", "N2", "T3"] {
        for lane in LANES {
            assert!(
                fires(get(id), lane, "window-short", "2"),
                "window-short silent on {id} {lane:?}"
            );
        }
    }
}

/// FNV-1a over the bits of one result: s, alpha (every part, peeled off as f64 roundings of the remainders), and the
/// bars
fn fold<T: Real>(h: &mut u64, loc: &multisolver::locate::Location<T>) {
    let mut mix = |x: f64| {
        for b in x.to_bits().to_le_bytes() {
            *h = (*h ^ b as u64).wrapping_mul(0x0000_0100_0000_01B3);
        }
    };
    let n = &loc.nearest;
    for v in [n.re, n.im, n.alpha] {
        // the 8 parts of an octo-double at most; past the last part the remainder is 0
        let mut r = v;
        for _ in 0..8 {
            let hi = r.to_f64();
            mix(hi);
            r = r - T::from_f64(hi);
        }
    }
    mix(n.bar);
    mix(n.alpha_bar);
    mix(loc.first as f64);
    mix(loc.degree as f64);
}

/// the fingerprint of the locator's results on every cell of the table; `flip` changes the lowest bit of coefficient 20
/// of the f64 series of D3 (the first choice, coefficient 10 of P1, left the fingerprint unchanged: the pure pole's
/// results do not move by one ulp of a coefficient; dev_12)
fn fingerprint(flip: bool) -> u64 {
    let mut h = 0xCBF2_9CE4_8422_2325u64;
    for case in cases().iter() {
        for lane in LANES {
            let opts = Opts::default();
            match lane {
                Lane::F64 => {
                    let mut c = series::<f64>(case, lane, lane.k(), Wrong::default());
                    if flip && case.id == "D3" {
                        c[20] = f64::from_bits(c[20].to_bits() + 1);
                    }
                    fold(&mut h, &locate(&c, &opts).unwrap());
                }
                Lane::Dd => fold(
                    &mut h,
                    &locate(&series::<DD>(case, lane, lane.k(), Wrong::default()), &opts).unwrap(),
                ),
                Lane::Qd => fold(
                    &mut h,
                    &locate(&series::<QD>(case, lane, lane.k(), Wrong::default()), &opts).unwrap(),
                ),
            }
        }
    }
    h
}

/// The results are bitwise the same on every backend of multidouble and every target: the fingerprint of the whole table
/// equals the constant recorded from the default build on aarch64, and the build with `--features portable` gives the
/// same constant. The controls: one f64 coefficient of D3 changed in its lowest bit changes the fingerprint, and so
/// does a quad-double result changed by 2^-70 of its remainder below the leading part, which the f64 rounding of
/// that remainder does not see.
#[test]
fn bitwise_across_backends() {
    const EXPECTED: u64 = 0xf3a1_3cba_157d_2e3c;
    let h = fingerprint(false);
    println!("fingerprint {h:#018x}");
    assert_eq!(h, EXPECTED, "the fingerprint of the table changed");
    assert_ne!(fingerprint(true), h, "control did not fire");
    let case = &cases()[0];
    let loc = locate(
        &series::<QD>(case, Lane::Qd, Lane::Qd.k(), Wrong::default()),
        &Opts::default(),
    )
    .unwrap();
    let re = loc.nearest.re;
    let rem = (re - QD::from_f64(re.to_f64())).to_f64();
    let mut low = loc.clone();
    low.nearest.re = re + QD::from_f64(rem * 2f64.powi(-70));
    assert!(rem != 0.0 && (low.nearest.re - re).to_f64() != 0.0);
    let (mut a, mut b) = (0u64, 0u64);
    fold(&mut a, &loc);
    fold(&mut b, &low);
    assert_ne!(a, b, "control did not fire: a quad-double's low parts");
}

/// Windows no fit can take give None, without overflow or a panic: a first order past K, degrees of usize::MAX and
/// usize::MAX / 2 (in `locate_at` and in `Opts`), and degree 0. A repeated degree counts once, so that one start with
/// the degree 3 thrice has no neighbor and gives None; a `first` of 1 is taken as 2. The controls: the same series of
/// P1 in a valid window, and with the degrees 2, 3 and 4, gives a location.
#[test]
fn infeasible_windows_give_none() {
    let c = series::<f64>(&cases()[0], Lane::F64, Lane::F64.k(), Wrong::default());
    let opts = Opts::default();
    assert!(locate_at(&c, Model::Ratio, usize::MAX, 3, &opts).is_none());
    assert!(locate_at(&c, Model::Ratio, 8, usize::MAX, &opts).is_none());
    assert!(locate_at(&c, Model::ThreeTerm, 8, usize::MAX / 2, &opts).is_none());
    assert!(locate_at(&c, Model::Ratio, 8, 0, &opts).is_none());
    assert!(locate_at(&c, Model::ThreeTerm, 8, 0, &opts).is_none());
    let huge = Opts {
        ratio_degrees: vec![usize::MAX],
        three_term_degrees: vec![usize::MAX / 2],
        ..Opts::default()
    };
    assert!(locate(&c, &huge).is_none());
    let zero = Opts {
        ratio_degrees: vec![0],
        three_term_degrees: vec![0],
        ..Opts::default()
    };
    assert!(locate(&c, &zero).is_none());
    let once = |degrees: Vec<usize>| Opts {
        starts: vec![0.3],
        ratio_degrees: degrees,
        three_term_degrees: vec![],
        ..Opts::default()
    };
    assert!(locate(&c, &once(vec![3, 3, 3])).is_none());
    assert_eq!(locate_at(&c, Model::Ratio, 1, 3, &opts).map(|l| l.first), Some(2));
    assert!(
        locate_at(&c, Model::Ratio, 8, 3, &opts).is_some() && locate(&c, &once(vec![2, 3, 4])).is_some(),
        "control did not fire"
    );
}
