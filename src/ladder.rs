//! Newton's method through a ladder of precisions: each stage runs the damped loop of [`crate::newton`] with its
//! residual, its Jacobian and its factors in precisions of its own, and ends at its floor by the floor rule
//! ([`NewtonOpts::floor_cut`]). The next stage then continues from that state in a higher precision.
//!
//! A residual in double-double has a floor of about 2^-106 of its terms, so a tolerance below it is not a stop rule:
//! the floor rule ends the stage at the first full step that cuts max |F| by less than `floor_cut` (10) with max |dz| at
//! most `floor_dz`, or that the damping refuses there. A stage also ends at its own `max_iter` and at a refused step.
//! Two ladders are typical:
//!
//! * the residuals': double-double, quad-double and octo-double residuals with one Jacobian and solve, the later
//!   stages taking chord steps with the last factors formed;
//! * the Jacobians': with the residual in P, a Jacobian formed in double-double, then in quad-double with
//!   double-double factors, then in quad-double with quad-double factors (multidouble's `QDLu`), whose steps contract
//!   by cond(J) 2^-209 where the double-double factors' contract by cond(J) 2^-104.
//!
//! A chord stage ([`Stage::chord`]) takes its steps with the factors in hand, from this stage or an earlier one. A
//! chord step that cuts max |F| by less than `floor_cut`, or that the damping refuses, has new factors formed for the
//! next step instead of ending the stage. Only a step with factors formed at its own iterate stops a stage at the
//! floor.
use crate::newton::{check_opts, damped, Factored, FactorsIn, Lower, Lowered, NewtonOpts, NewtonReport, Source, Stop};
use crate::Problem;
use multidouble::precision::{Double, Octo, Precision, Quad};
use multidouble::simd::Native;
use std::rc::Rc;

/// The precision of a stage's residual or Jacobian
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prec {
    Double,
    Quad,
    Octo,
}

impl Prec {
    /// the number of f64 parts of a scalar
    pub fn parts(self) -> usize {
        match self {
            Prec::Double => 2,
            Prec::Quad => 4,
            Prec::Octo => 8,
        }
    }
}

/// The precision a stage's factors solve in
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Solve {
    /// the f64 LU of the Jacobian's leading parts ([`Lower::F64`])
    F64,
    /// the double-double LU ([`Lower::Double`])
    Double,
    /// the quad-double LU (`QDLu`, [`FactorsIn`])
    Quad,
    /// the octo-double LU (`ODLu`, [`FactorsIn`])
    Octo,
}

/// One stage of [`newton_ladder`]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage {
    /// the residual's precision, and the state's
    pub res: Prec,
    /// the precision the Jacobian is formed in, at the state's leading parts
    pub jac: Prec,
    /// the precision its factors solve in
    pub solve: Solve,
    /// Newton steps at most in the stage
    pub max_iter: usize,
    /// chord steps with the factors in hand (the module's documentation)
    pub chord: bool,
}

impl Stage {
    /// a stage that forms its factors at every step
    pub fn new(res: Prec, jac: Prec, solve: Solve, max_iter: usize) -> Stage {
        Stage {
            res,
            jac,
            solve,
            max_iter,
            chord: false,
        }
    }

    /// the stage with chord steps
    pub fn with_chord(self) -> Stage {
        Stage { chord: true, ..self }
    }
}

/// The residuals' ladder: double-double, quad-double and octo-double residuals, the Jacobian formed in `jac` and
/// solved in `solve`; the first stage takes up to `max_iter` steps, each later one up to `max_hi`, as chord steps
pub fn residual_ladder(jac: Prec, solve: Solve, max_iter: usize, max_hi: usize) -> Vec<Stage> {
    vec![
        Stage::new(Prec::Double, jac, solve, max_iter),
        Stage::new(Prec::Quad, jac, solve, max_hi).with_chord(),
        Stage::new(Prec::Octo, jac, solve, max_hi).with_chord(),
    ]
}

/// The Jacobians' ladder at the residual `res`: the Jacobian formed in double-double and solved there, then formed in
/// quad-double and solved in double-double, then formed and solved in quad-double; `max_iter` steps each
pub fn jacobian_ladder(res: Prec, max_iter: usize) -> Vec<Stage> {
    vec![
        Stage::new(res, Prec::Double, Solve::Double, max_iter),
        Stage::new(res, Prec::Quad, Solve::Double, max_iter),
        Stage::new(res, Prec::Quad, Solve::Quad, max_iter),
    ]
}

/// What [`newton_ladder`] did: each stage run with its report, whose `stop` says why it ended and whose last `fnorm`
/// is the floor or the tolerance it reached; the Jacobians formed; and whether the last stage converged
#[derive(Clone, Debug)]
pub struct LadderReport {
    pub stages: Vec<(Stage, NewtonReport)>,
    pub jacobians: usize,
    pub converged: bool,
}

/// the factors of one stage
enum Lu {
    Low(Lowered),
    Quad(FactorsIn<Quad>),
    Octo(FactorsIn<Octo>),
}

impl<P: Precision<Native>> Factored<P> for Lu {
    fn solve_neg(&self, r: &[P::Scalar]) -> Vec<P::Scalar> {
        match self {
            Lu::Low(l) => Factored::<P>::solve_neg(l, r),
            Lu::Quad(f) => Factored::<P>::solve_neg(f, r),
            Lu::Octo(f) => Factored::<P>::solve_neg(f, r),
        }
    }
}

/// the leading k parts of z, zero parts added past its own
fn lead(z: &[&[f64]], k: usize) -> Vec<Vec<f64>> {
    let n = z[0].len();
    (0..k)
        .map(|q| z.get(q).map_or_else(|| vec![0.0; n], |p| p.to_vec()))
        .collect()
}

/// The factors of a stage: formed at the step's iterate, or with `chord` the ones in hand until they go stale
struct Kept<'a, Pr: ?Sized> {
    pr: &'a Pr,
    st: Stage,
    par: bool,
    kept: &'a mut Option<Rc<Lu>>,
    stale: bool,
    formed: usize,
}

impl<Pr> Kept<'_, Pr>
where
    Pr: Problem<Double> + Problem<Quad> + Problem<Octo> + ?Sized,
{
    /// the Jacobian at z in the stage's precision, factored in its solve's
    fn form(&self, z: &[&[f64]]) -> Result<Lu, String> {
        let zj = lead(z, self.st.jac.parts());
        let zj: Vec<&[f64]> = zj.iter().map(Vec::as_slice).collect();
        let (c, ld) = match self.st.jac {
            Prec::Double => Problem::<Double>::jacobian(self.pr, &zj, self.par)?,
            Prec::Quad => Problem::<Quad>::jacobian(self.pr, &zj, self.par)?,
            Prec::Octo => Problem::<Octo>::jacobian(self.pr, &zj, self.par)?,
        };
        let n = Problem::<Double>::len(self.pr);
        Ok(match self.st.solve {
            Solve::F64 => Lu::Low(Lowered::factor(n, ld, c, Lower::F64, self.par)?),
            Solve::Double => Lu::Low(Lowered::factor(n, ld, c, Lower::Double, self.par)?),
            Solve::Quad => Lu::Quad(FactorsIn::factor(n, ld, c, self.par)?),
            Solve::Octo => Lu::Octo(FactorsIn::factor(n, ld, c, self.par)?),
        })
    }
}

impl<P: Precision<Native>, Pr> Source<P, Rc<Lu>> for Kept<'_, Pr>
where
    Pr: Problem<Double> + Problem<Quad> + Problem<Octo> + ?Sized,
{
    fn get(&mut self, z: &[&[f64]]) -> Result<(Rc<Lu>, bool), String> {
        if self.st.chord && !self.stale {
            if let Some(f) = self.kept.as_ref() {
                return Ok((f.clone(), false));
            }
        }
        let f = Rc::new(self.form(z)?);
        *self.kept = Some(f.clone());
        self.stale = false;
        self.formed += 1;
        Ok((f, true))
    }
    fn stale(&mut self) {
        self.stale = true;
    }
}

/// z with k parts: zero parts added, or the leading k kept (the state rounded to the stage's precision)
fn resize(z: &mut Vec<Vec<f64>>, k: usize) {
    let n = z[0].len();
    z.truncate(k);
    z.resize(k, vec![0.0; n]);
}

/// Newton's method from z (any number of part arrays, updated in place and left with the last stage's parts) through
/// `stages` in turn, each the damped loop of `opts` (its `max_iter` replaced by the stage's) in the stage's
/// precisions, until one converges to `opts.tol` (or `opts.tol_dz`). A stage that ends otherwise hands its state to
/// the next, except one whose residual is not finite, which ends the ladder. With `opts.floor_cut` 0 the stages end
/// only at their step caps and refusals.
pub fn newton_ladder<Pr>(
    pr: &Pr,
    z: &mut Vec<Vec<f64>>,
    stages: &[Stage],
    opts: &NewtonOpts,
) -> Result<LadderReport, String>
where
    Pr: Problem<Double> + Problem<Quad> + Problem<Octo> + ?Sized,
{
    check_opts(opts)?;
    if stages.is_empty() || z.is_empty() {
        return Err("newton_ladder: no stages or no state".into());
    }
    let mut kept: Option<Rc<Lu>> = None;
    let mut out = LadderReport {
        stages: Vec::new(),
        jacobians: 0,
        converged: false,
    };
    for &st in stages {
        resize(z, st.res.parts());
        let o = NewtonOpts {
            max_iter: st.max_iter,
            ..*opts
        };
        let mut src = Kept {
            pr,
            st,
            par: opts.par,
            kept: &mut kept,
            stale: false,
            formed: 0,
        };
        let rep = match st.res {
            Prec::Double => damped::<Double, Pr, Rc<Lu>>(pr, z, &o, &mut src)?,
            Prec::Quad => damped::<Quad, Pr, Rc<Lu>>(pr, z, &o, &mut src)?,
            Prec::Octo => damped::<Octo, Pr, Rc<Lu>>(pr, z, &o, &mut src)?,
        };
        out.jacobians += src.formed;
        let (done, stop) = (rep.converged, rep.stop);
        out.stages.push((st, rep));
        if done || stop == Stop::NotFinite {
            out.converged = done;
            break;
        }
    }
    Ok(out)
}
