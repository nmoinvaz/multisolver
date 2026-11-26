//! The a-posteriori Kantorovich quantities of a damped Newton run, from the norms its report keeps (Deuflhard, Newton
//! Methods for Nonlinear Problems, Springer 2004, sections 2.1.1 and 3.3).
//!
//! For the affine covariant Lipschitz condition |J(x)^-1 (J(y) - J(x)) v| <= omega |y - x| |v|, a full step dz_k and
//! its simplified correction dz_bar_k = -J(z_k)^-1 F(z_k + dz_k) satisfy |dz_bar_k| <= (omega / 2) |dz_k|^2. So
//!
//! ```text
//! theta_k = |dz_bar_k| / |dz_k|,   [h_k] = 2 theta_k,   [omega]_k = [h_k] / |dz_k|,
//! ```
//!
//! lower bounds of the Kantorovich quantity h_k = omega |dz_k| and of omega, not enclosures. A damped step with
//! lambda < 1 satisfies |dz_bar - (1 - lambda) dz| <= (lambda^2 / 2) omega |dz|^2, and the triangle inequality gives
//! `[h_k]` = 2 max(theta_k - (1 - lambda), 0) / lambda^2. Newton's method from z_k converges (Newton-Mysovskikh)
//! where h_k < 2, and while it does, |z_k - z*| <= |dz_k| / (1 - h_k / 2). The estimate with `[h_k]` in place of h_k
//! is not a bound, as `[h_k]` is one from below.
//!
//! These hold for corrections with the exact Jacobian's factors at each step's iterate (`newton`, or `newton_with`
//! with `exact`). With other factors (`newton_chord`'s reused ones, `newton_lowered`'s, any approximation through
//! `newton_with`) the simplified correction also carries the factors' mismatch, and the values are not lower bounds:
//! for F(x) = x - 1 from x = 2 with the factor 2 in place of 1, `[omega]` is 2 where omega is 0. Nor are they for
//! `newton_with_projection` with a projection that moves its trials, whose simplified corrections also carry the
//! projection's displacement; the identity gives `newton_with`'s steps and keeps them. A report records neither its
//! factors nor a projection, so the caller states them, and each step says whether it is a lower bound.
use crate::newton::NewtonReport;

/// The quantities of one step: its lambda, |dz|, theta, `[h]` and `[omega]` (NaN where |dz| is not known), and
/// whether `[h]` and `[omega]` are lower bounds (the step's factors the exact Jacobian's at its iterate)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Step {
    pub lam: f64,
    pub dz: f64,
    pub theta: f64,
    pub h: f64,
    pub omega: f64,
    pub lower_bound: bool,
}

impl Step {
    /// |dz| / (1 - `[h]` / 2), the error estimate of the step's start, infinite for `[h]` >= 2
    pub fn error_estimate(&self) -> f64 {
        if self.h < 2.0 {
            self.dz / (1.0 - self.h / 2.0)
        } else {
            f64::INFINITY
        }
    }
}

/// The steps' quantities from their lambdas, contractions theta = |dz_bar| / |dz| and norms |dz| (the module's
/// documentation), in order; `exact`: the corrections used the exact Jacobian's factors at each iterate and carry no
/// projection's displacement
pub fn from_steps(lam: &[f64], theta: &[f64], dz: &[f64], exact: bool) -> Vec<Step> {
    lam.iter()
        .zip(theta)
        .zip(dz)
        .map(|((&lam, &theta), &dz)| {
            let h = if lam == 1.0 {
                2.0 * theta
            } else {
                2.0 * (theta - (1.0 - lam)).max(0.0) / (lam * lam)
            };
            Step {
                lam,
                dz,
                theta,
                h,
                omega: h / dz,
                lower_bound: exact,
            }
        })
        .collect()
}

/// The quantities of every step a report of the correction rule records (`NewtonReport::scale`, `theta`, `dz_bar`),
/// |dz_k| = |dz_bar_k| / theta_k; empty for the residual rule, which records no theta. `exact` states that the run's
/// factors were the exact Jacobian's at each iterate (`newton`) and that no projection moved its trials; otherwise
/// every step's `lower_bound` is false (the module's documentation).
pub fn estimates(rep: &NewtonReport, exact: bool) -> Vec<Step> {
    let n = rep.theta.len().min(rep.dz_bar.len()).min(rep.scale.len());
    let dz: Vec<f64> = (0..n).map(|k| rep.dz_bar[k] / rep.theta[k]).collect();
    from_steps(&rep.scale[..n], &rep.theta[..n], &dz, exact)
}
