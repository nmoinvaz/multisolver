//! The asymptotic numerical method on a planted branch: a cyclic system whose root is a quadratic polynomial in the
//! parameter. The crate multishoot tests it on a multiple-shooting eigenproblem.

use multisolver::anm::{series, series_with, Branch};
use multisolver::multidouble::linalg::ld_for;
use multisolver::newton::{lowered, Lower};
use multisolver::prec::{nan_max, parts};
use multisolver::{Double, JetOrders, Jvp, Native, Precision, Problem, Quad, Recompute, ResidualJet, DD};

const N: usize = 10;
const KAPPA: f64 = 0.3;
const LAM0: f64 = 0.7;

/// F_i(z, lam) = z_i^3 - w_i(lam)^3 + kappa (z_(i+1) - w_(i+1)(lam)), the index cyclic, with the quadratic
/// w_i(lam) = a_i + b_i lam + c_i lam^2, so the branch through lam0 is z(lam0 + h) = w(lam0 + h): z_1 = b + 2 c lam0,
/// z_2 = c, z_k = 0 for k >= 3. `freeze`: the series with lam a constant (the control). `nan_jvp`: the products J v
/// all NaN (a broken refinement, the control of the recorded residuals). `by_order`: the series one order at a time
/// by `residual_jet` (0), by `PolyOrders` (1), or by `PolyOrders` without its redo (2, the control).
struct Poly<P: Precision<Native>> {
    a: Vec<P::Scalar>,
    b: Vec<P::Scalar>,
    c: Vec<P::Scalar>,
    freeze: bool,
    nan_jvp: bool,
    by_order: u8,
}

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

/// the product of two series truncated to n terms
fn conv<P: Precision<Native>>(x: &[P::Scalar], y: &[P::Scalar], n: usize) -> Vec<P::Scalar> {
    (0..n)
        .map(|k| (0..=k).fold(P::ZERO, |t, j| t + x[j] * y[k - j]))
        .collect()
}

impl<P: Precision<Native>> Poly<P> {
    fn new(freeze: bool) -> Self {
        Poly {
            a: (0..N).map(|i| s::<P>(1.0 + 0.1 * (i as f64).sin())).collect(),
            b: (0..N).map(|i| s::<P>(0.2 * (i as f64).cos())).collect(),
            c: (0..N).map(|i| s::<P>(0.05 * (2.0 * i as f64).sin())).collect(),
            freeze,
            nan_jvp: false,
            by_order: 0,
        }
    }
    /// w_i(lam0 + h) as a series of n terms
    fn w(&self, i: usize, n: usize) -> Vec<P::Scalar> {
        let l = s::<P>(LAM0);
        let two = s::<P>(2.0);
        let mut w = vec![P::ZERO; n.max(3)];
        w[0] = self.a[i] + self.b[i] * l + self.c[i] * l * l;
        if !self.freeze {
            w[1] = self.b[i] + two * self.c[i] * l;
            w[2] = self.c[i];
        }
        w.truncate(n);
        w
    }
    /// the root at lam0
    fn root(&self) -> Vec<Vec<f64>> {
        to_parts::<P>(&(0..N).map(|i| self.w(i, 1)[0]).collect::<Vec<_>>())
    }
}

impl<P: Precision<Native>> Problem<P> for Poly<P> {
    fn len(&self) -> usize {
        N
    }
    fn residual(&self, z: &[&[f64]], par: bool) -> Result<Vec<P::Scalar>, String> {
        Ok(self.residual_jet(&[z.to_vec()], 1, par)?.remove(0))
    }
    fn jacobian(&self, z: &[&[f64]], _par: bool) -> Result<(Vec<Vec<f64>>, usize), String> {
        let z = scalars::<P>(z);
        let ld = ld_for(N);
        let mut c = vec![vec![0.0; ld * N]; P::PARTS];
        for i in 0..N {
            let d = s::<P>(3.0) * z[i] * z[i];
            for (q, cq) in c.iter_mut().enumerate() {
                cq[i * ld + i] = P::part(d, q);
                cq[((i + 1) % N) * ld + i] = P::part(s::<P>(KAPPA), q);
            }
        }
        Ok((c, ld))
    }
}

impl<P: Precision<Native>> Jvp<P> for Poly<P> {
    fn jvp(&self, z: &[&[f64]], v: &[&[f64]], _par: bool) -> Result<Vec<P::Scalar>, String> {
        let (z, v) = (scalars::<P>(z), scalars::<P>(v));
        if self.nan_jvp {
            return Ok(vec![s::<P>(f64::NAN); N]);
        }
        Ok((0..N)
            .map(|i| s::<P>(3.0) * z[i] * z[i] * v[i] + s::<P>(KAPPA) * v[(i + 1) % N])
            .collect())
    }
}

impl<P: Precision<Native>> ResidualJet<P> for Poly<P> {
    fn residual_jet(&self, zs: &[Vec<&[f64]>], nout: usize, _par: bool) -> Result<Vec<Vec<P::Scalar>>, String> {
        let zk: Vec<Vec<P::Scalar>> = zs.iter().map(|z| scalars::<P>(z)).collect();
        // the series of unknown i, n terms
        let ser = |i: usize| -> Vec<P::Scalar> {
            (0..nout)
                .map(|k| if k < zk.len() { zk[k][i] } else { P::ZERO })
                .collect()
        };
        let cube = |x: &[P::Scalar]| conv::<P>(&conv::<P>(x, x, nout), x, nout);
        let rows: Vec<Vec<P::Scalar>> = (0..N)
            .map(|i| {
                let j = (i + 1) % N;
                let (zi, zj, wi, wj) = (ser(i), ser(j), self.w(i, nout), self.w(j, nout));
                let (z3, w3) = (cube(&zi), cube(&wi));
                (0..nout)
                    .map(|k| z3[k] - w3[k] + s::<P>(KAPPA) * (zj[k] - wj[k]))
                    .collect()
            })
            .collect();
        Ok((0..nout).map(|k| rows.iter().map(|r| r[k]).collect()).collect())
    }
    fn jet_orders<'a>(&'a self, _order: usize) -> Result<Box<dyn JetOrders<P> + 'a>, String>
    where
        P: 'a,
    {
        if self.by_order == 0 {
            return Ok(Box::new(Recompute::new(self)));
        }
        Ok(Box::new(PolyOrders {
            pr: self,
            x: vec![Vec::new(); N],
            sq: vec![Vec::new(); N],
            redo: self.by_order == 1,
        }))
    }
}

/// The polynomial's series one order at a time: each unknown's coefficients and its square's, kept between calls.
/// The square's order k - 1, formed with z_(k-1) zero by the call before, is formed again once z_(k-1) is given
/// (`redo`; the control keeps it). Every sum is `conv`'s, term for term, so order k is `residual_jet`'s, bitwise.
struct PolyOrders<'a, P: Precision<Native>> {
    pr: &'a Poly<P>,
    x: Vec<Vec<P::Scalar>>,
    sq: Vec<Vec<P::Scalar>>,
    redo: bool,
}

/// order m of the product of the series x and y, their coefficients past the given ones zero, summed as `conv` does
fn conv_at<P: Precision<Native>>(x: &[P::Scalar], y: &[P::Scalar], m: usize) -> P::Scalar {
    let at = |v: &[P::Scalar], j: usize| if j < v.len() { v[j] } else { P::ZERO };
    (0..=m).fold(P::ZERO, |t, j| t + at(x, j) * at(y, m - j))
}

impl<P: Precision<Native>> JetOrders<P> for PolyOrders<'_, P> {
    fn next(&mut self, z: &[&[f64]], _par: bool) -> Result<Vec<P::Scalar>, String> {
        let z = scalars::<P>(z);
        let k = self.x[0].len() + 1;
        for ((x, sq), &zi) in self.x.iter_mut().zip(&mut self.sq).zip(&z) {
            x.push(zi);
            if sq.len() == k {
                if self.redo {
                    sq[k - 1] = conv_at::<P>(x, x, k - 1);
                }
            } else {
                sq.push(conv_at::<P>(x, x, k - 1));
            }
            sq.push(conv_at::<P>(x, x, k));
        }
        Ok((0..N)
            .map(|i| {
                let j = (i + 1) % N;
                let z3 = conv_at::<P>(&self.sq[i], &self.x[i], k);
                let wi = self.pr.w(i, k + 1);
                let w3 = conv::<P>(&conv::<P>(&wi, &wi, k + 1), &wi, k + 1)[k];
                let wj = self.pr.w(j, k + 1)[k];
                z3 - w3 + s::<P>(KAPPA) * (P::ZERO - wj)
            })
            .collect())
    }
}

/// The series of order 8 at the root of the polynomial branch in P, refined once
fn poly_branch<P: Precision<Native>>(freeze: bool, nan_jvp: bool) -> Branch<P> {
    poly_branch_by::<P>(freeze, nan_jvp, 0)
}

/// `poly_branch` with the residual's orders from `Poly::by_order`'s source
fn poly_branch_by<P: Precision<Native>>(freeze: bool, nan_jvp: bool, by_order: u8) -> Branch<P> {
    let mut pr = Poly::<P>::new(freeze);
    pr.nan_jvp = nan_jvp;
    pr.by_order = by_order;
    let z0 = pr.root();
    series::<P, _>(&pr, &parts(&z0), 8, 1, false).unwrap()
}

/// a series of order 8 against the true branch: [|z_1 - w'| / |w'|, |z_2 - c| / |c|, max over k >= 3 of |z_k|,
/// |z(0.3) - w(lam0 + 0.3)| / |w(lam0 + 0.3)|], max norms, NaN where any entry is
fn poly_errors<P: Precision<Native>>(br: &Branch<P>) -> [f64; 4] {
    let truth = Poly::<P>::new(false);
    let rel = |x: &[P::Scalar], y: &[P::Scalar]| {
        let d = x
            .iter()
            .zip(y)
            .map(|(&a, &b)| P::to_dd(a - b).hi.abs())
            .fold(0.0, nan_max);
        d / y.iter().map(|&b| P::to_dd(b).hi.abs()).fold(0.0, nan_max)
    };
    let w1: Vec<P::Scalar> = (0..N).map(|i| truth.w(i, 2)[1]).collect();
    let high = (3..=8)
        .flat_map(|k| br.z[k].iter().map(|&v| P::to_dd(v).hi.abs()))
        .fold(0.0, nan_max);
    let wa: Vec<P::Scalar> = (0..N)
        .map(|i| {
            let w = truth.w(i, 3);
            let a = s::<P>(0.3);
            w[0] + (w[1] + w[2] * a) * a
        })
        .collect();
    [
        rel(&br.z[1], &w1),
        rel(&br.z[2], &truth.c),
        high,
        rel(&br.eval(0.3), &wa),
    ]
}

/// the accuracy checks of the polynomial branch: [z_1, z_2, z(0.3)] within tol, [z_3..z_8] within tol_high
fn poly_checks(e: [f64; 4], tol: f64, tol_high: f64) -> [bool; 4] {
    [e[0] <= tol, e[1] <= tol, e[2] <= tol_high, e[3] <= tol]
}

/// Pass, written before the first run: at the root of the polynomial branch the series of order 8 gives z_1 and z_2
/// to 1e-29 relative, every higher order below 1e-28, and z(0.3) to 1e-29, in double-double; in quad-double the same
/// to 1e-60 and 1e-59. The controls, each a wrong series in double-double: with the parameter frozen in the
/// residual's series, z_1, z_2 and z(0.3) miss (its higher orders are zero); z_2 off by 1e-20 fails z_2 and z(0.3);
/// z_3 = 1e-20 in one entry, and a NaN in z_5, fail the higher orders and z(0.3).
#[test]
fn anm_reproduces_a_polynomial_branch() {
    let d = poly_errors(&poly_branch::<Double>(false, false));
    println!(
        "double-double: z_1 {:.3e}, z_2 {:.3e}, z_3..z_8 {:.3e}, z(0.3) {:.3e}",
        d[0], d[1], d[2], d[3]
    );
    assert_eq!(poly_checks(d, 1e-29, 1e-28), [true; 4]);
    let q = poly_errors(&poly_branch::<Quad>(false, false));
    println!(
        "quad-double: z_1 {:.3e}, z_2 {:.3e}, z_3..z_8 {:.3e}, z(0.3) {:.3e}",
        q[0], q[1], q[2], q[3]
    );
    assert_eq!(poly_checks(q, 1e-60, 1e-59), [true; 4]);
    let control = |name: &str, br: &Branch<Double>, want: [bool; 4]| {
        let e = poly_errors(br);
        println!(
            "control, {name}: z_1 {:.3e}, z_2 {:.3e}, z_3..z_8 {:.3e}, z(0.3) {:.3e}",
            e[0], e[1], e[2], e[3]
        );
        assert_eq!(poly_checks(e, 1e-29, 1e-28), want, "control {name} did not fire");
    };
    control(
        "parameter frozen",
        &poly_branch::<Double>(true, false),
        [false, false, true, false],
    );
    let good = poly_branch::<Double>(false, false);
    let mut br = good.clone();
    br.z[2][0] = br.z[2][0] + DD::from_f64(1e-20);
    control("z_2 off by 1e-20", &br, [true, false, true, false]);
    let mut br = good.clone();
    br.z[3][0] = DD::from_f64(1e-20);
    control("z_3 = 1e-20", &br, [true, true, false, false]);
    let mut br = good;
    br.z[5][0] = DD::from_f64(f64::NAN);
    control("NaN in z_5", &br, [true, true, false, false]);
}

/// Pass, written before the first run: the solves of the polynomial branch's eight orders have relative residuals of
/// at most 1e-30 after one refinement, and its frozen branch, constant in the parameter, has an infinite radius. The
/// control: with the products J v all NaN every recorded residual is NaN, and the check rejects them.
#[test]
fn anm_records_its_solve_residuals() {
    let check = |br: &Branch<Double>| br.residual.iter().copied().fold(0.0, nan_max) <= 1e-30;
    let br = poly_branch::<Double>(false, false);
    println!("residuals: {:?}", br.residual);
    assert!(check(&br));
    let frozen = poly_branch::<Double>(true, false);
    println!("frozen branch: radius {}", frozen.radius(1e-8));
    assert_eq!(frozen.radius(1e-8), f64::INFINITY);
    let bad = poly_branch::<Double>(false, true);
    println!("control, products NaN: residuals {:?}", bad.residual);
    assert!(
        bad.residual.iter().all(|r| r.is_nan()) && !check(&bad),
        "control did not fire"
    );
}

/// the number of entries of x and y whose parts differ in any bit
fn bits_differ<P: Precision<Native>>(x: &[P::Scalar], y: &[P::Scalar]) -> usize {
    x.iter()
        .zip(y)
        .filter(|(&u, &v)| (0..P::PARTS).any(|q| P::part(u, q).to_bits() != P::part(v, q).to_bits()))
        .count()
}

/// Pass, written before the first run: the series of order 8 whose residual orders come one at a time from a
/// problem that keeps its lower orders (`PolyOrders`) is the series from `residual_jet`, bit for bit, in double- and
/// quad-double. The control: without the redo of order k - 1 once z_(k-1) is given, the coefficients differ, first
/// at order 2.
#[test]
fn anm_takes_the_orders_from_jet_orders() {
    fn run<P: Precision<Native>>(name: &str) {
        let base = poly_branch_by::<P>(false, false, 0);
        let by_order = poly_branch_by::<P>(false, false, 1);
        let stale = poly_branch_by::<P>(false, false, 2);
        let per_order =
            |b: &Branch<P>| -> Vec<usize> { base.z.iter().zip(&b.z).map(|(x, y)| bits_differ::<P>(x, y)).collect() };
        let (d1, d2) = (per_order(&by_order), per_order(&stale));
        let first = d2.iter().position(|&n| n > 0);
        println!("{name}: differing entries per order, by order {d1:?}, control {d2:?}");
        assert!(
            d1.iter().all(|&n| n == 0),
            "{name}: the orders from jet_orders changed the series"
        );
        assert_eq!(first, Some(2), "{name}: control did not fire as written");
    }
    run::<Double>("double-double");
    run::<Quad>("quad-double");
}

/// The series of order 8 at the root of the polynomial branch in quad-double, the Jacobian there rounded by `to` and
/// factored, each order's solve refined `refine` times
fn poly_branch_lowered(to: Lower, refine: usize) -> Branch<Quad> {
    let pr = Poly::<Quad>::new(false);
    let z0 = pr.root();
    series_with::<Quad, _, _>(&pr, &parts(&z0), 8, refine, false, |z| {
        lowered::<Quad, _>(&pr, z, to, false)
    })
    .unwrap()
}

/// Pass, written before the first run: in quad-double, with the Jacobian at the root rounded to double-double and
/// each order's solve refined twice by the exact products (each refinement contracting its error by about
/// cond(J) 2^-104), or rounded to f64 and refined six times (about cond(J) 2^-53 each), the series passes the
/// quad-double checks of `anm_reproduces_a_polynomial_branch`: z_1, z_2 and z(0.3) to 1e-60, the higher orders below
/// 1e-59. The control: rounded to double-double and not refined, the solves keep the factors' error of about 1e-31
/// and fail all four checks.
#[test]
fn anm_with_lowered_factors_reaches_the_precision() {
    for (name, to, refine, want) in [
        ("double-double factors, refined twice", Lower::Double, 2, [true; 4]),
        ("f64 factors, refined six times", Lower::F64, 6, [true; 4]),
        (
            "control, double-double factors, not refined",
            Lower::Double,
            0,
            [false; 4],
        ),
    ] {
        let e = poly_errors(&poly_branch_lowered(to, refine));
        println!(
            "{name}: z_1 {:.3e}, z_2 {:.3e}, z_3..z_8 {:.3e}, z(0.3) {:.3e}",
            e[0], e[1], e[2], e[3]
        );
        assert_eq!(poly_checks(e, 1e-60, 1e-59), want, "{name}");
    }
}
