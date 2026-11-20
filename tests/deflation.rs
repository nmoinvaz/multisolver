//! Deflation on planted roots: systems of four unknowns with four known roots, their rows mixed by a dense matrix,
//! whose roots are found one after another from one start; the deflated Jacobian against central differences in
//! octo-double; and the residual at and next to a known root.

use multisolver::deflation::{roots_from, DeflationOpts, Found, Without};
use multisolver::gcr::{newton_refined, GcrOpts};
use multisolver::multidouble::linalg::ld_for;
use multisolver::newton::{exact, newton, newton_lowered, Damping, Lower, NewtonOpts};
use multisolver::prec::{max_abs, nan_max, parts};
use multisolver::{Double, Jvp, Native, Octo, Precision, Problem, Quad, DD};

const N: usize = 4;

fn s<P: Precision<Native>>(x: f64) -> P::Scalar {
    P::from_dd(DD::from_f64(x))
}

fn scalars<P: Precision<Native>>(z: &[&[f64]]) -> Vec<P::Scalar> {
    (0..z[0].len())
        .map(|i| P::from_raw(&z.iter().map(|p| p[i]).collect::<Vec<_>>()))
        .collect()
}

fn to_parts<P: Precision<Native>>(v: &[P::Scalar]) -> Vec<Vec<f64>> {
    (0..P::PARTS)
        .map(|q| v.iter().map(|&x| P::part(x, q)).collect())
        .collect()
}

/// F(z) = B G(z) with G_1 = (z_1 - 1)(z_1 + 1/2), G_2 = (z_2 - 2)(z_2 - 3/4), G_3 = z_3 - z_1 z_2 / 2 and
/// G_4 = z_4 - z_3^2, and B the dense matrix 2 I plus off-diagonal entries 0.3 sin(1 + 3 i + 5 j), diagonally
/// dominant. Its four roots ([`ROOTS`]) are exact in binary.
struct Four<P: Precision<Native>> {
    b: Vec<Vec<P::Scalar>>,
}

/// z_1 in {1, -1/2}, z_2 in {2, 3/4}, z_3 = z_1 z_2 / 2, z_4 = z_3^2
const ROOTS: [[f64; N]; 4] = [
    [1.0, 2.0, 1.0, 1.0],
    [1.0, 0.75, 0.375, 0.140625],
    [-0.5, 2.0, -0.5, 0.25],
    [-0.5, 0.75, -0.1875, 0.03515625],
];

impl<P: Precision<Native>> Four<P> {
    fn new() -> Self {
        let b = (0..N)
            .map(|i| {
                (0..N)
                    .map(|j| {
                        s::<P>(if i == j {
                            2.0
                        } else {
                            0.3 * ((1 + 3 * i + 5 * j) as f64).sin()
                        })
                    })
                    .collect()
            })
            .collect();
        Four { b }
    }

    fn g(z: &[P::Scalar]) -> Vec<P::Scalar> {
        let h = s::<P>(0.5);
        vec![
            (z[0] - P::ONE) * (z[0] + h),
            (z[1] - s::<P>(2.0)) * (z[1] - s::<P>(0.75)),
            z[2] - z[0] * z[1] * h,
            z[3] - z[2] * z[2],
        ]
    }

    fn mix(&self, v: &[P::Scalar]) -> Vec<P::Scalar> {
        self.b
            .iter()
            .map(|row| P::dot(row.iter().copied().zip(v.iter().copied())))
            .collect()
    }
}

impl<P: Precision<Native>> Problem<P> for Four<P> {
    fn len(&self) -> usize {
        N
    }
    fn residual(&self, z: &[&[f64]], _par: bool) -> Result<Vec<P::Scalar>, String> {
        Ok(self.mix(&Self::g(&scalars::<P>(z))))
    }
    fn jacobian(&self, z: &[&[f64]], _par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
        let z = scalars::<P>(z);
        let h = s::<P>(0.5);
        // dG/dz by columns
        let mut gz = vec![vec![P::ZERO; N]; N];
        gz[0][0] = s::<P>(2.0) * z[0] - h;
        gz[0][2] = -z[1] * h;
        gz[1][1] = s::<P>(2.0) * z[1] - s::<P>(2.75);
        gz[1][2] = -z[0] * h;
        gz[2][2] = P::ONE;
        gz[2][3] = -s::<P>(2.0) * z[2];
        gz[3][3] = P::ONE;
        let ld = ld_for(N);
        let mut c = vec![vec![0.0; ld * N]; P::PARTS];
        for (j, col) in gz.iter().enumerate() {
            for (i, &v) in self.mix(col).iter().enumerate() {
                for (q, cq) in c.iter_mut().enumerate() {
                    cq[j * ld + i] = P::part(v, q);
                }
            }
        }
        Ok((c, ld))
    }
}

impl<P: Precision<Native>> Jvp<P> for Four<P> {
    /// B (dG/dz v)
    fn jvp(&self, z: &[&[f64]], v: &[&[f64]], _par: bool) -> Result<Vec<P::Scalar>, String> {
        let (z, v) = (scalars::<P>(z), scalars::<P>(v));
        let h = s::<P>(0.5);
        let gv = vec![
            (s::<P>(2.0) * z[0] - h) * v[0],
            (s::<P>(2.0) * z[1] - s::<P>(2.75)) * v[1],
            v[2] - (z[1] * v[0] + z[0] * v[1]) * h,
            v[3] - s::<P>(2.0) * z[2] * v[2],
        ];
        Ok(self.mix(&gv))
    }
}

/// the start of every search
fn z0<P: Precision<Native>>() -> Vec<Vec<f64>> {
    to_parts::<P>(&[s::<P>(0.1), s::<P>(0.3), P::ZERO, P::ZERO])
}

/// root k of ROOTS as part arrays
fn root<P: Precision<Native>>(k: usize) -> Vec<Vec<f64>> {
    to_parts::<P>(&ROOTS[k].map(s::<P>))
}

/// max |z - r| over the entries
fn dist<P: Precision<Native>>(z: &[Vec<f64>], r: &[f64; N]) -> f64 {
    scalars::<P>(&parts(z))
        .iter()
        .zip(r)
        .map(|(&a, &b)| P::to_dd(a - s::<P>(b)).hi.abs())
        .fold(0.0, nan_max)
}

/// The checks of a list of roots against the four planted ones: [there are 4, no two nearest the same planted root,
/// each within `ztol` of a planted root with max |F| <= `ftol`]
fn checks<P: Precision<Native>>(
    pr: &impl Problem<P>,
    planted: &[[f64; N]; 4],
    roots: &[Vec<Vec<f64>>],
    ztol: f64,
    ftol: f64,
) -> [bool; 3] {
    let nearest = |z: &Vec<Vec<f64>>| {
        (0..4)
            .min_by(|&a, &b| dist::<P>(z, &planted[a]).total_cmp(&dist::<P>(z, &planted[b])))
            .unwrap()
    };
    let mut ks: Vec<usize> = roots.iter().map(nearest).collect();
    let accurate = roots.iter().all(|z| {
        let f = max_abs::<P>(&pr.residual(&parts(z), false).unwrap());
        dist::<P>(z, &planted[nearest(z)]) <= ztol && f <= ftol
    });
    ks.sort();
    ks.dedup();
    [roots.len() == 4, ks.len() == roots.len(), accurate]
}

fn opts(tol: f64) -> NewtonOpts {
    NewtonOpts {
        tol,
        max_iter: 100,
        par: false,
        ..Default::default()
    }
}

/// F(z) = B G(z) with (G_1, G_2) the real and imaginary parts of (z_1 + i z_2)^4 - 1, G_3 = z_3 - z_1 z_2 and
/// G_4 = z_4 - (z_1^2 - z_2^2) / 2, and B Four's matrix. Its roots ([`UNITY`]) are z_1 + i z_2 = 1, -1, i, -i, exact in
/// binary, and its Jacobian is singular only where z_1 = z_2 = 0.
struct Unity<P: Precision<Native>> {
    four: Four<P>,
}

/// the four roots of Unity
const UNITY: [[f64; N]; 4] = [
    [1.0, 0.0, 0.0, 0.5],
    [-1.0, 0.0, 0.0, 0.5],
    [0.0, 1.0, 0.0, -0.5],
    [0.0, -1.0, 0.0, -0.5],
];

impl<P: Precision<Native>> Problem<P> for Unity<P> {
    fn len(&self) -> usize {
        N
    }
    fn residual(&self, z: &[&[f64]], _par: bool) -> Result<Vec<P::Scalar>, String> {
        let v = scalars::<P>(z);
        let (x, y) = (v[0], v[1]);
        let (x2, y2) = (x * x, y * y);
        let g = [
            x2 * x2 - s::<P>(6.0) * x2 * y2 + y2 * y2 - P::ONE,
            s::<P>(4.0) * x * y * (x2 - y2),
            v[2] - x * y,
            v[3] - (x2 - y2) * s::<P>(0.5),
        ];
        Ok(self.four.mix(&g))
    }
    fn jacobian(&self, z: &[&[f64]], _par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
        let v = scalars::<P>(z);
        let (x, y) = (v[0], v[1]);
        let (x2, y2) = (x * x, y * y);
        // d(Re)/dx = d(Im)/dy = 4 x^3 - 12 x y^2 and d(Im)/dx = -d(Re)/dy = 12 x^2 y - 4 y^3 (Cauchy and Riemann)
        let a = x * (s::<P>(4.0) * x2 - s::<P>(12.0) * y2);
        let b = y * (s::<P>(12.0) * x2 - s::<P>(4.0) * y2);
        let mut gz = vec![vec![P::ZERO; N]; N];
        (gz[0][0], gz[0][1], gz[0][2], gz[0][3]) = (a, b, -y, -x);
        (gz[1][0], gz[1][1], gz[1][2], gz[1][3]) = (-b, a, -x, y);
        (gz[2][2], gz[3][3]) = (P::ONE, P::ONE);
        let ld = ld_for(N);
        let mut c = vec![vec![0.0; ld * N]; P::PARTS];
        for (j, col) in gz.iter().enumerate() {
            for (i, &v) in self.four.mix(col).iter().enumerate() {
                for (q, cq) in c.iter_mut().enumerate() {
                    cq[j * ld + i] = P::part(v, q);
                }
            }
        }
        Ok((c, ld))
    }
}

/// full Newton steps: the residual rule without halvings
fn full_steps(tol: f64) -> NewtonOpts {
    NewtonOpts {
        damping: Damping::Residual,
        max_halvings: 0,
        ..opts(tol)
    }
}

/// the start of Unity's searches
fn u0<P: Precision<Native>>() -> Vec<Vec<f64>> {
    to_parts::<P>(&[0.45, 0.6, 0.1, -0.2].map(s::<P>))
}

/// roots_from on Unity from u0, 5 searches by full Newton steps (`newton_lowered` with double-double factors when
/// `lowered`)
fn search<P: Precision<Native>>(tol: f64, lowered: bool) -> (Unity<P>, Found) {
    let pr = Unity { four: Four::<P>::new() };
    let o = full_steps(tol);
    let found = roots_from::<P, _>(&pr, &u0::<P>(), 5, &DeflationOpts::default(), |g, z| {
        if lowered {
            newton_lowered::<P, _>(g, z, &o, Lower::Double)
        } else {
            newton::<P, _>(g, z, &o)
        }
    })
    .unwrap();
    (pr, found)
}

/// Pass, written before the first run: from the one start (0.45, 0.6, 0.1, -0.2), deflation with full Newton steps
/// (the residual rule without halvings, the paper's undamped iteration) finds the four roots of Unity one after
/// another and a fifth search ends without a root: in double-double to max |G| <= 1e-30, each root within 1e-29 of a
/// planted one with max |F| <= 1e-30, and in quad-double with `newton_lowered` (double-double factors) to 1e-62,
/// within 1e-61 and max |F| <= 1e-62. The controls: two searches by plain Newton from the same start return the same
/// root and fail the check of distinct roots; and a found root moved by 1e-20 fails the check of accuracy.
///
/// The test's first version asked the same of Four from (0.1, 0.3, 0, 0) with the correction rule, and failed it:
/// the searches found two roots. Deflation scales the undeflated Newton step and keeps its direction, and the
/// correction rule follows the continuous Newton path, which deflation does not change, so its searches rarely leave
/// the first root's path. (In a scratch study over 49 starts, the correction rule found a second root of Unity from
/// none of them, full steps all four from 42.)
#[test]
fn deflation_finds_every_root_from_one_start() {
    let (pr, found) = search::<Double>(1e-30, false);
    let d: Vec<String> = found
        .roots
        .iter()
        .map(|z| format!("({:.3}, {:.3})", z[0][0], z[0][1]))
        .collect();
    println!(
        "double-double: {} roots {d:?}, steps {:?}, ended: {:?}",
        found.roots.len(),
        found.reports.iter().map(|r| r.scale.len()).collect::<Vec<_>>(),
        found.ended
    );
    assert_eq!(checks::<Double>(&pr, &UNITY, &found.roots, 1e-29, 1e-30), [true; 3]);
    assert!(found.ended.is_some(), "a fifth root");
    let (prq, fq) = search::<Quad>(1e-62, true);
    println!(
        "quad-double, double-double factors: {} roots, steps {:?}, ended: {:?}",
        fq.roots.len(),
        fq.reports.iter().map(|r| r.scale.len()).collect::<Vec<_>>(),
        fq.ended
    );
    assert_eq!(checks::<Quad>(&prq, &UNITY, &fq.roots, 1e-61, 1e-62), [true; 3]);
    assert!(fq.ended.is_some(), "a fifth root in quad-double");

    // the controls
    let plain: Vec<Vec<Vec<f64>>> = (0..2)
        .map(|_| {
            let mut z = u0::<Double>();
            assert!(newton::<Double, _>(&pr, &mut z, &full_steps(1e-30)).unwrap().converged);
            z
        })
        .collect();
    let c = checks::<Double>(&pr, &UNITY, &plain, 1e-29, 1e-30);
    println!("control, plain Newton twice: {c:?}");
    assert!(!c[1], "control, plain Newton");
    let mut moved = found.roots.clone();
    let z = DD::new(moved[1][0][2], moved[1][1][2]) + DD::from_f64(1e-20);
    (moved[1][0][2], moved[1][1][2]) = (z.hi, z.lo);
    assert_eq!(
        checks::<Double>(&pr, &UNITY, &moved, 1e-29, 1e-30),
        [true, true, false],
        "control, moved root"
    );
}

/// The Jacobian of Without in P (roots 0 and 1 of ROOTS deflated, at a point away from them) against the central
/// difference of Without's residual in octo-double with h = 2^-133, whose truncation error is of order h^2 ~ 1e-80:
/// max over the entries of the difference, relative to max |J|, NaN when any entry is. `rank_one` false drops the
/// term F w^T; `nan` sets entry (1, 1) to NaN.
fn jacobian_error<P: Precision<Native>>(rank_one: bool, nan: bool) -> f64 {
    let at = [0.3, 1.1, 0.2, -0.4];
    let roots: Vec<Vec<Vec<f64>>> = (0..2).map(root::<P>).collect();
    let pr = Four::<P>::new();
    let g = Without::new(&pr, &roots, DeflationOpts::default()).unwrap();
    let z = to_parts::<P>(&at.map(s::<P>));
    let (c, ld) = if rank_one {
        g.jacobian(&parts(&z), false).unwrap()
    } else {
        // M J alone
        let (m, _) = g.operator(&parts(&z)).unwrap();
        let (mut c, ld) = pr.jacobian(&parts(&z), false).unwrap();
        for j in 0..N {
            for i in 0..N {
                let v = m * P::from_raw(&c.iter().map(|q| q[j * ld + i]).collect::<Vec<_>>());
                for (q, cq) in c.iter_mut().enumerate() {
                    cq[j * ld + i] = P::part(v, q);
                }
            }
        }
        (c, ld)
    };
    let mut c = c;
    if nan {
        c[0][0] = f64::NAN;
    }
    // the difference in octo-double
    let pro = Four::<Octo>::new();
    let rooto: Vec<Vec<Vec<f64>>> = (0..2).map(root::<Octo>).collect();
    let go = Without::new(&pro, &rooto, DeflationOpts::default()).unwrap();
    let h = 2f64.powi(-133);
    let inv2h = s::<Octo>(2f64.powi(132));
    let mut worst = 0.0f64;
    let mut big = 0.0f64;
    for j in 0..N {
        let shifted = |sgn: f64| {
            let mut v = at.map(s::<Octo>);
            v[j] = v[j] + s::<Octo>(sgn * h);
            go.residual(&parts(&to_parts::<Octo>(&v)), false).unwrap()
        };
        let (fp, fm) = (shifted(1.0), shifted(-1.0));
        for i in 0..N {
            let fd = (fp[i] - fm[i]) * inv2h;
            // the octo-double difference rounded to P through its leading parts
            let fd_parts: Vec<f64> = (0..P::PARTS)
                .map(|q| <Octo as Precision<Native>>::part(fd, q))
                .collect();
            let jp = P::from_raw(&c.iter().map(|q| q[j * ld + i]).collect::<Vec<_>>());
            worst = nan_max(worst, P::to_dd(jp - P::new(&fd_parts)).hi.abs());
            big = nan_max(big, P::to_dd(jp).hi.abs());
        }
    }
    worst / big
}

/// Pass, written before the first run: the deflated Jacobian M (J + F w^T), with two roots deflated, agrees with the
/// octo-double central difference of M F to 1e-30 relative in double-double and 1e-62 in quad-double. The controls:
/// M J without the rank-one term F w^T misses by more than 1e-30; and the Jacobian with one entry NaN fails the
/// check.
#[test]
fn deflated_jacobian_is_exact() {
    let e2 = jacobian_error::<Double>(true, false);
    let e4 = jacobian_error::<Quad>(true, false);
    let ec = jacobian_error::<Double>(false, false);
    let en = jacobian_error::<Double>(true, true);
    println!("double-double {e2:.3e}, quad-double {e4:.3e}, control without F w^T {ec:.3e}, with a NaN {en:.3e}");
    assert!(e2 <= 1e-30 && e4 <= 1e-62);
    assert!(ec > 1e-30, "control did not fire");
    assert!(en.is_nan() || en > 1e-30, "control, a NaN entry");
}

/// Without's J v (from Four's) against its Jacobian times v, in P, at a point away from roots 0 and 1, deflated: max
/// over the rows of the difference, relative to max |J v|. `rank_one` false forms M J v alone.
fn jvp_error<P: Precision<Native>>(rank_one: bool) -> f64 {
    let at = [0.3, 1.1, 0.2, -0.4].map(s::<P>);
    let v = [0.7, -0.2, 0.5, 0.9].map(s::<P>);
    let roots: Vec<Vec<Vec<f64>>> = (0..2).map(root::<P>).collect();
    let pr = Four::<P>::new();
    let g = Without::new(&pr, &roots, DeflationOpts::default()).unwrap();
    let (z, vp) = (to_parts::<P>(&at), to_parts::<P>(&v));
    let jv = if rank_one {
        g.jvp(&parts(&z), &parts(&vp), false).unwrap()
    } else {
        let (m, _) = g.operator(&parts(&z)).unwrap();
        pr.jvp(&parts(&z), &parts(&vp), false)
            .unwrap()
            .iter()
            .map(|&x| m * x)
            .collect()
    };
    let (c, ld) = g.jacobian(&parts(&z), false).unwrap();
    let mut worst = 0.0f64;
    let mut big = 0.0f64;
    for (i, &jvi) in jv.iter().enumerate() {
        let row = P::dot((0..N).map(|j| (P::from_raw(&c.iter().map(|q| q[j * ld + i]).collect::<Vec<_>>()), v[j])));
        worst = nan_max(worst, P::to_dd(jvi - row).hi.abs());
        big = nan_max(big, P::to_dd(row).hi.abs());
    }
    worst / big
}

/// Pass, written before the first run: Without's Jacobian-vector product M (J v + F (w . v)), with two roots
/// deflated, agrees with its Jacobian times v to 1e-30 relative in double-double and 1e-62 in quad-double; and
/// `newton_refined`, preconditioned by the exact factors, on Four with root 0 deflated, from root 3 moved by 0.05 in
/// every unknown with full steps, converges to root 3 within 1e-28 with max |F| <= 1e-28. The control: M J v without
/// the rank-one term misses by more than 1e-30.
#[test]
fn deflated_jvp_is_the_jacobians() {
    let e2 = jvp_error::<Double>(true);
    let e4 = jvp_error::<Quad>(true);
    let ec = jvp_error::<Double>(false);
    println!("double-double {e2:.3e}, quad-double {e4:.3e}, control without F (w . v) {ec:.3e}");
    assert!(e2 <= 1e-30 && e4 <= 1e-62);
    assert!(ec > 1e-30, "control did not fire");

    let pr = Four::<Double>::new();
    let g = Without::new(&pr, &[root::<Double>(0)], DeflationOpts::default()).unwrap();
    let mut z = to_parts::<Double>(&ROOTS[3].map(|r| DD::from_f64(r + 0.05)));
    let (rep, products) = newton_refined::<Double, _, _>(&g, &mut z, &full_steps(1e-28), &GcrOpts::default(), |z| {
        exact::<Double, _>(&g, z, false)
    })
    .unwrap();
    let f = max_abs::<Double>(&pr.residual(&parts(&z), false).unwrap());
    let d = dist::<Double>(&z, &ROOTS[3]);
    println!(
        "newton_refined: converged {} in {} steps, {products} products, |z - r_3| {d:.3e}, max |F| {f:.3e}",
        rep.converged,
        rep.scale.len()
    );
    assert!(rep.converged && d <= 1e-28 && f <= 1e-28);
}

/// Pass, written before the first run: with power 1024 and shift 1, at distance 2 from the deflated root 0, where
/// |z - r|^p = 2^1024 overflows f64 while M = 1 + 2^-1024, the operator is finite, with |M - 1| <= 1e-300 and
/// max |w| <= 1e-300; with the default power at distance 1e200, where |z - r|^2 overflows, M = 1 and w = 0; and with
/// shift 0 at that distance, where M = |z - r|^-2 is below f64's range, there is no operator. The control: at the
/// root itself there is no operator, so the check of a finite one fails.
#[test]
fn deflation_operator_is_finite_far_from_a_root() {
    let pr = Four::<Double>::new();
    let r = root::<Double>(0);
    let at = |dz: f64| to_parts::<Double>(&[DD::from_f64(ROOTS[0][0] + dz), DD::ONE.mul_f(2.0), DD::ONE, DD::ONE]);
    let op = |power: u32, shift: f64, z: &[Vec<f64>]| {
        let g = Without::new(&pr, std::slice::from_ref(&r), DeflationOpts { power, shift }).unwrap();
        g.operator(&parts(z))
    };
    let small = |o: Option<(DD, Vec<DD>)>, wtol: f64| {
        o.is_some_and(|(m, w)| {
            (m - DD::ONE).hi.abs() <= 1e-300 && w.iter().map(|x| x.hi.abs()).fold(0.0, nan_max) <= wtol
        })
    };
    let near = op(1024, 1.0, &at(2.0));
    println!("power 1024 at distance 2: {near:?}");
    assert!(small(near, 1e-300), "power 1024 at distance 2");
    let far = op(2, 1.0, &at(1e200));
    println!("power 2 at distance 1e200: {far:?}");
    assert!(far
        .as_ref()
        .is_some_and(|(m, w)| *m == DD::ONE && w.iter().all(|x| *x == DD::ZERO)));
    assert!(op(2, 0.0, &at(1e200)).is_none(), "shift 0 at distance 1e200");
    assert!(!small(op(1024, 1.0, &r), 1e-300), "control did not fire");
}

/// Pass, written before the first run: with root 0 deflated, the residual is +inf in every row, never NaN, at the
/// root itself and at the root with z_1 moved by 1e-200 (where |z - r|^-2 overflows), and is finite at the root moved
/// by 1e-120; Newton from the root itself returns unconverged, without an error. The control: at the start z0 the
/// residual is finite, so the check of +inf fails.
#[test]
fn deflated_residual_is_infinite_at_a_known_root() {
    let pr = Four::<Double>::new();
    let r = root::<Double>(0);
    let g = Without::new(&pr, std::slice::from_ref(&r), DeflationOpts::default()).unwrap();
    let all_inf = |z: &[Vec<f64>]| {
        g.residual(&parts(z), false)
            .unwrap()
            .iter()
            .all(|v| v.hi == f64::INFINITY)
    };
    let moved = |dz: f64| {
        let mut z = r.clone();
        z[1][0] = dz;
        z
    };
    let finite_at = |z: &[Vec<f64>]| g.residual(&parts(z), false).unwrap().iter().all(|v| v.hi.is_finite());
    println!(
        "at the root {}, moved by 1e-200 {}, finite moved by 1e-120 {}",
        all_inf(&r),
        all_inf(&moved(1e-200)),
        finite_at(&moved(1e-120))
    );
    assert!(all_inf(&r) && all_inf(&moved(1e-200)) && finite_at(&moved(1e-120)));
    let mut z = r.clone();
    let rep = newton::<Double, _>(&g, &mut z, &opts(1e-30)).unwrap();
    println!("Newton from the root: converged {}, |G| {:?}", rep.converged, rep.fnorm);
    assert!(!rep.converged);
    assert!(!all_inf(&z0::<Double>()), "control did not fire");
}
