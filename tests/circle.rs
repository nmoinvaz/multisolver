//! The winding on a circle and the zeros inside (`circle`), on planted functions g(lambda) = e^(b lambda)
//! prod (lambda - z_j) / prod (lambda - p_k), whose ln|g| and arg g are formed in f64.
use multisolver::circle::{circle, CircleOpts};
use multisolver::winding::C64;

/// (ln|g|, arg g) of e^(b lambda) prod (lambda - z) / prod (lambda - p)
fn planted(zeros: Vec<C64>, poles: Vec<C64>, b: C64) -> impl Fn(C64) -> Result<(f64, f64), String> {
    move |l: C64| {
        let (mut la, mut arg) = (b.0 * l.0 - b.1 * l.1, b.0 * l.1 + b.1 * l.0);
        for &z in &zeros {
            let d = (l.0 - z.0, l.1 - z.1);
            la += d.0.hypot(d.1).ln();
            arg += d.1.atan2(d.0);
        }
        for &p in &poles {
            let d = (l.0 - p.0, l.1 - p.1);
            la -= d.0.hypot(d.1).ln();
            arg -= d.1.atan2(d.0);
        }
        Ok((la, arg))
    }
}

/// the largest distance of found zeros from the planted ones (each matched to its nearest)
fn zero_err(found: &[C64], want: &[C64]) -> f64 {
    want.iter()
        .map(|w| {
            found
                .iter()
                .map(|f| (f.0 - w.0).hypot(f.1 - w.1))
                .fold(f64::INFINITY, f64::min)
        })
        .fold(0.0, f64::max)
}

/// Pass, written before the first run: on |lambda - 0.1| = 1, with zeros 0.3 + 0.2i and -0.4 - 0.1i inside, one zero
/// and one pole outside, and e^((3 - 2i) lambda): the winding is 2 within 1e-9, the circle resolved, and with n0 = 64
/// the zeros are found within 1e-12, each error estimate at least the error; with a pole inside in place of a zero,
/// the winding is 0 and no zeros are located. The control: from n0 = 8 points, where the trapezoid rule's error is
/// about 0.45^8, the zeros must miss by more than 1e-6.
#[test]
fn circle_locates_planted_zeros() {
    let zs = [(0.3, 0.2), (-0.4, -0.1)];
    let g = planted(vec![zs[0], zs[1], (2.5, 0.3)], vec![(-1.9, 1.4)], (3.0, -2.0));
    let o = CircleOpts {
        n0: 64,
        ..Default::default()
    };
    let r = circle(g, (0.1, 0.0), 1.0, &o).unwrap();
    let e = zero_err(&r.zeros, &zs);
    println!(
        "two zeros: winding {:.12}, {} points, resolved {}, zeros within {e:.3e} (estimates {:?})",
        r.turns,
        r.t.len() - 1,
        r.resolved,
        r.zeros_err
    );
    assert!(r.resolved && (r.turns - 2.0).abs() < 1e-9 && r.zeros.len() == 2);
    assert!(e <= 1e-12, "{e:e}");
    for (z, &est) in r.zeros.iter().zip(&r.zeros_err) {
        let actual = zs
            .iter()
            .map(|w| (z.0 - w.0).hypot(z.1 - w.1))
            .fold(f64::INFINITY, f64::min);
        assert!(est >= actual, "estimate {est:e} below the error {actual:e}");
    }
    let gp = planted(vec![(0.3, 0.2)], vec![(-0.4, -0.1)], (3.0, -2.0));
    let rp = circle(gp, (0.1, 0.0), 1.0, &o).unwrap();
    println!("a zero and a pole: winding {:.12}", rp.turns);
    assert!(rp.turns.abs() < 1e-9 && rp.zeros.is_empty());
    let few = circle(
        planted(vec![zs[0], zs[1], (2.5, 0.3)], vec![(-1.9, 1.4)], (3.0, -2.0)),
        (0.1, 0.0),
        1.0,
        &CircleOpts {
            n0: 8,
            ..Default::default()
        },
    )
    .unwrap();
    let ef = zero_err(&few.zeros, &zs);
    println!("control, n0 = 8: zeros within {ef:.3e}");
    assert!(ef > 1e-6, "control did not fire");
}

/// Pass, written before the first run: two zeros 0.004 apart and 0.005 inside the unit circle (at angle 0.3), which
/// the 16 starting points straddle, give the winding 2 with the alias rule, and the circle is resolved. The control:
/// without the alias rule (alias infinite), from 16 points, the winding must come out otherwise than 2.
#[test]
fn alias_rule_resolves_a_close_pair() {
    let a: f64 = 0.3;
    let zs = [
        (0.995 * (a - 0.002).cos(), 0.995 * (a - 0.002).sin()),
        (0.995 * (a + 0.002).cos(), 0.995 * (a + 0.002).sin()),
    ];
    let g = planted(zs.to_vec(), vec![], (0.0, 0.0));
    let r = circle(&g, (0.0, 0.0), 1.0, &CircleOpts::default()).unwrap();
    println!(
        "close pair: winding {:.9}, {} points, resolved {}",
        r.turns,
        r.t.len() - 1,
        r.resolved
    );
    assert!(r.resolved && (r.turns - 2.0).abs() < 1e-9);
    let flat = CircleOpts {
        alias: f64::INFINITY,
        ..Default::default()
    };
    let c = circle(&g, (0.0, 0.0), 1.0, &flat).unwrap();
    println!(
        "control, no alias rule: winding {:.9}, {} points",
        c.turns,
        c.t.len() - 1
    );
    assert!((c.turns - 2.0).abs() > 0.5, "control did not fire");
}
