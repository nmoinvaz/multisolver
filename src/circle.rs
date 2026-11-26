//! The winding number of an analytic function g on a circle |lambda - c| = r, and the places of its zeros inside: a
//! local test of a single zero or a few, such as a mode of det M(lambda) located by a polygon's count first.
//!
//! g is given by its values (ln|g|, arg g). Its exponential growth across the circle is removed first: with beta the
//! slope of ln g between c - 2r and c + 2r, the phase of h(lambda) = g(lambda) exp(-beta (lambda - c)) has the winding
//! of g, as the factor is entire and has no zeros. The phase is followed from `n0` equal arcs, and an arc is bisected
//! while its wrapped step exceeds `tol`, or while its length times the local scale of ln h (the largest |Delta ln h|
//! / arc over the arc and `neighbours` arcs each side, cyclically) exceeds `alias`. ln|h| does not wrap, so a close
//! pair of zeros between two samples shows in it. Then every arc whose step exceeds tol / 2 is bisected once more and
//! its halves are checked to add up to it, until a pass adds nothing. The circle is resolved when every step is at
//! most tol, no halves mismatched, and every arc above `min_arc` meets the alias rule.
//!
//! When the winding W is 1 to `max_zeros`, h(lambda) / (lambda - c)^W has no winding, so its logarithm L is periodic on
//! the circle. Integration by parts then gives the power sums of the zeros z_j inside,
//!
//! ```text
//! s_k = sum_j (z_j - c)^k = -(k / 2 pi i) closed int (lambda - c)^(k-1) L d lambda
//!     = -(k r^k / n) sum_m L(lambda_m) e^(i k t_m),        t_m = 2 pi m / n, k = 1 .. W,
//! ```
//!
//! the second form being the trapezoid rule on the n0 equal points, which converges geometrically for a periodic
//! analytic integrand. Newton's identities give the polynomial whose roots are the zeros, which Durand and Kerner's
//! iteration finds (E. Durand, Solutions numeriques des equations algebriques, Masson 1960; I. O. Kerner, Numer. Math.
//! 8 (1966) 290, doi:10.1007/BF02162564). The same sums on every other point (n0 / 2) give second values, and their
//! distance from the first is the error estimate: conservative, as the n0 / 2 sums are the less accurate.
use crate::winding::{cabs, cadd, cdiv, cmul, csub, wrap, C64};
use std::f64::consts::TAU;

/// The settings of [`circle`]
#[derive(Clone, Copy, Debug)]
pub struct CircleOpts {
    /// the starting arcs, equal, and the points of the power sums (even)
    pub n0: usize,
    /// the largest arg step accepted
    pub tol: f64,
    /// the largest arc length times the local scale of ln h (infinite: no alias rule)
    pub alias: f64,
    /// the arcs each side whose scales enter an arc's
    pub neighbours: usize,
    /// the shortest arc length bisected
    pub min_arc: f64,
    /// the evaluations at most
    pub max_evals: usize,
    /// the largest winding whose zeros are located
    pub max_zeros: usize,
}

impl Default for CircleOpts {
    fn default() -> Self {
        CircleOpts {
            n0: 16,
            tol: std::f64::consts::FRAC_PI_4,
            alias: std::f64::consts::FRAC_PI_2,
            neighbours: 3,
            min_arc: 1e-6,
            max_evals: 20_000,
            max_zeros: 6,
        }
    }
}

/// What [`circle`] found: the samples (fractions t of the turn and the detrended values (ln|h|, arg h)), beta, the
/// winding (the sum of the arg steps over 2 pi), whether the circle is resolved, the mismatched halves, the
/// evaluations of g, and, for a winding 1 to `max_zeros`, the zeros inside and the error estimate of each
#[derive(Clone, Debug)]
pub struct Circle {
    pub c: C64,
    pub r: f64,
    pub beta: C64,
    pub t: Vec<f64>,
    pub val: Vec<(f64, f64)>,
    pub turns: f64,
    pub resolved: bool,
    pub mismatches: usize,
    pub evals: usize,
    pub zeros: Vec<C64>,
    pub zeros_err: Vec<f64>,
}

/// Newton's identities: the coefficients e_1..e_W of prod (u - u_j) = u^W - e_1 u^(W-1) + e_2 u^(W-2) - ... from the
/// power sums s_1..s_W of the u_j
fn elementary(s: &[C64]) -> Vec<C64> {
    let mut e: Vec<C64> = vec![(1.0, 0.0)];
    for k in 1..=s.len() {
        let mut acc = (0.0, 0.0);
        for i in 1..=k {
            let term = cmul(e[k - i], s[i - 1]);
            acc = if i % 2 == 1 { cadd(acc, term) } else { csub(acc, term) };
        }
        e.push(cdiv(acc, (k as f64, 0.0)));
    }
    e
}

/// the roots of u^W - e_1 u^(W-1) + ... + (-1)^W e_W by Durand and Kerner's iteration from points on a circle of the
/// coefficients' scale
fn roots(e: &[C64]) -> Vec<C64> {
    let w = e.len() - 1;
    // p(u) with the signs of Newton's identities
    let p = |u: C64| {
        let mut v = (1.0, 0.0);
        for (k, &ek) in e.iter().enumerate().skip(1) {
            let c = if k % 2 == 1 { (-ek.0, -ek.1) } else { ek };
            v = cadd(cmul(v, u), c);
        }
        v
    };
    let scale = (1..=w)
        .map(|k| cabs(e[k]).powf(1.0 / k as f64))
        .fold(0.0, f64::max)
        .max(1e-300);
    let mut u: Vec<C64> = (0..w)
        .map(|j| {
            let a = 0.4 + TAU * j as f64 / w as f64;
            (scale * a.cos(), scale * a.sin())
        })
        .collect();
    for _ in 0..500 {
        let mut moved = 0.0f64;
        for j in 0..w {
            let mut den = (1.0, 0.0);
            for (k, &uk) in u.iter().enumerate() {
                if k != j {
                    den = cmul(den, csub(u[j], uk));
                }
            }
            let d = cdiv(p(u[j]), den);
            u[j] = csub(u[j], d);
            moved = moved.max(cabs(d) / scale);
        }
        if moved < 1e-15 {
            break;
        }
    }
    u
}

/// The winding of g on |lambda - c| = r and the zeros inside (the module's documentation); g(lambda) returns (ln|g|,
/// arg g) or an error, which ends the scan
pub fn circle<F: FnMut(C64) -> Result<(f64, f64), String>>(
    mut g: F,
    c: C64,
    r: f64,
    opts: &CircleOpts,
) -> Result<Circle, String> {
    if !(r > 0.0 && opts.n0 >= 4 && opts.n0.is_multiple_of(2)) {
        return Err(format!("circle: r > 0 and an even n0 >= 4, not r {r}, n0 {}", opts.n0));
    }
    let mut evals = 0usize;
    let mut eval = |lam: C64, evals: &mut usize| -> Result<(f64, f64), String> {
        *evals += 1;
        if *evals > opts.max_evals {
            return Err(format!("circle: more than {} evaluations", opts.max_evals));
        }
        g(lam)
    };
    let (lp, ap) = eval(cadd(c, (2.0 * r, 0.0)), &mut evals)?;
    let (lm, am) = eval(csub(c, (2.0 * r, 0.0)), &mut evals)?;
    let beta = cdiv((lp - lm, wrap(ap - am)), (4.0 * r, 0.0));
    let at = |t: f64| cadd(c, (r * (TAU * t).cos(), r * (TAU * t).sin()));
    let mut h = |t: f64, evals: &mut usize| -> Result<(f64, f64), String> {
        let lam = at(t);
        let (la, arg) = eval(lam, evals)?;
        let v = csub((la, arg), cmul(beta, csub(lam, c)));
        Ok((v.0, wrap(v.1)))
    };
    let n0 = opts.n0;
    let mut ts: Vec<f64> = (0..=n0).map(|i| i as f64 / n0 as f64).collect();
    let mut vs: Vec<(f64, f64)> = Vec::with_capacity(n0 + 1);
    for &t in &ts[..n0] {
        vs.push(h(t, &mut evals)?);
    }
    vs.push(vs[0]);
    let len = TAU * r;
    let step = |vs: &[(f64, f64)], i: usize| wrap(vs[i + 1].1 - vs[i].1);
    let scales = |ts: &[f64], vs: &[(f64, f64)]| -> Vec<f64> {
        (0..ts.len() - 1)
            .map(|i| (vs[i + 1].0 - vs[i].0).hypot(step(vs, i)) / ((ts[i + 1] - ts[i]) * len))
            .collect()
    };
    let local = |sc: &[f64], i: usize| {
        let n = sc.len() as isize;
        let nb = opts.neighbours as isize;
        (-nb..=nb)
            .map(|j| sc[(i as isize + j).rem_euclid(n) as usize])
            .fold(0.0, crate::prec::nan_max)
    };
    let mut checked: Vec<(f64, f64)> = Vec::new();
    let mut mismatches = 0;
    loop {
        // bisection to the step and alias rules
        loop {
            let sc = scales(&ts, &vs);
            let mut split: Vec<usize> = Vec::new();
            for i in 0..ts.len() - 1 {
                if (ts[i + 1] - ts[i]) * len <= opts.min_arc {
                    continue;
                }
                if step(&vs, i).abs() > opts.tol || (ts[i + 1] - ts[i]) * len * local(&sc, i) > opts.alias {
                    split.push(i);
                }
            }
            if split.is_empty() {
                break;
            }
            for &i in split.iter().rev() {
                let m = 0.5 * (ts[i] + ts[i + 1]);
                let v = h(m, &mut evals)?;
                ts.insert(i + 1, m);
                vs.insert(i + 1, v);
            }
        }
        // verification of every arc above tol / 2, once
        let mut added = false;
        let mut i = 0;
        while i < ts.len() - 1 {
            let d = step(&vs, i);
            let key = (ts[i], ts[i + 1]);
            if d.abs() > opts.tol / 2.0 && !checked.contains(&key) && (ts[i + 1] - ts[i]) * len > opts.min_arc {
                checked.push(key);
                let m = 0.5 * (ts[i] + ts[i + 1]);
                let v = h(m, &mut evals)?;
                ts.insert(i + 1, m);
                vs.insert(i + 1, v);
                added = true;
                if (step(&vs, i) + step(&vs, i + 1) - d).abs() > 1e-6 {
                    mismatches += 1;
                }
                i += 2;
            } else {
                i += 1;
            }
        }
        if !added {
            break;
        }
    }
    let steps: Vec<f64> = (0..ts.len() - 1).map(|i| step(&vs, i)).collect();
    let turns = steps.iter().sum::<f64>() / TAU;
    let sc = scales(&ts, &vs);
    let alias_left = (0..ts.len() - 1)
        .filter(|&i| (ts[i + 1] - ts[i]) * len > opts.min_arc && (ts[i + 1] - ts[i]) * len * local(&sc, i) > opts.alias)
        .count();
    let max_step = steps.iter().map(|s| s.abs()).fold(0.0, f64::max);
    let resolved = max_step <= opts.tol && mismatches == 0 && alias_left == 0;
    let mut out = Circle {
        c,
        r,
        beta,
        t: ts.clone(),
        val: vs.clone(),
        turns,
        resolved,
        mismatches,
        evals,
        zeros: Vec::new(),
        zeros_err: Vec::new(),
    };
    let w = turns.round();
    if resolved && (turns - w).abs() < 1e-6 && w >= 1.0 && w as usize <= opts.max_zeros {
        let w = w as usize;
        // L on the n0 base points: the phase continued through every sample, less that of (lambda - c)^W
        let mut phase = vs[0].1;
        let mut lnh: Vec<C64> = Vec::with_capacity(n0);
        let mut next = 0usize;
        for i in 0..ts.len() - 1 {
            if next < n0 && ts[i] == next as f64 / n0 as f64 {
                let t = ts[i];
                lnh.push((vs[i].0 - w as f64 * r.ln(), phase - w as f64 * TAU * t));
                next += 1;
            }
            phase += steps[i];
        }
        debug_assert_eq!(lnh.len(), n0);
        // the zeros u_j = (z_j - c) / r from the scaled sums s_k / r^k = -(k / n) sum_m L_m e^(i k t_m)
        let zeros_of = |every: usize| -> Vec<C64> {
            let pts: Vec<usize> = (0..n0).step_by(every).collect();
            let s: Vec<C64> = (1..=w)
                .map(|k| {
                    let mut acc = (0.0, 0.0);
                    for &m in &pts {
                        let a = TAU * (k * m) as f64 / n0 as f64;
                        acc = cadd(acc, cmul(lnh[m], (a.cos(), a.sin())));
                    }
                    let f = -(k as f64) / pts.len() as f64;
                    (f * acc.0, f * acc.1)
                })
                .collect();
            roots(&elementary(&s))
                .into_iter()
                .map(|u| cadd(c, (r * u.0, r * u.1)))
                .collect()
        };
        let (z, mut z2) = (zeros_of(1), zeros_of(2));
        let mut err = Vec::with_capacity(w);
        for &zi in &z {
            let (j, d) = z2
                .iter()
                .enumerate()
                .map(|(j, &y)| (j, cabs(csub(zi, y))))
                .fold((0, f64::INFINITY), |a, b| if b.1 < a.1 { b } else { a });
            err.push(d);
            z2.remove(j);
        }
        out.zeros = z;
        out.zeros_err = err;
    }
    Ok(out)
}
