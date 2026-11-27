- Every check must be able to fail: each new accuracy check comes with a control that feeds it a wrong result (a test
  that asserts the failure), and it is run once to see it fire.
- Planted truth first: the solvers are tested on problems with a known solution (a planted linear system, a
  polynomial branch, a planted singular value decomposition, an analytic function with zeros at known places), with
  the pass criteria written in the test's doc comment before the first run.
- Results must not depend on multidouble's SIMD backend: run `cargo test --release` on the default build, with
  `--features portable`, with `-C target-feature=+avx2,+fma`, and on aarch64 (NEON). Comparisons across backends are
  bitwise.
- The solvers know nothing of any one problem: they see a `Problem` (and `Jvp`, `ResidualJet`) and nothing else.
  Equations, layouts and constants belong to the crate that uses them; tests here use model problems.
- A bound the documentation states as a theorem is proved in `lean/`, in exact arithmetic, with controls in
  `lean/Controls` where its conclusion fails without a hypothesis; `lake build` there after changing either.
- `cargo fmt` (max width 120) and `cargo clippy --all-targets -- -D warnings` (and once with `--features portable`)
  before committing.
- Commit messages: imperative, a lowercase module prefix where one applies (`newton: ...`, `winding: ...`), 1-3
  sentences of body saying what changed and why, no attribution trailers.
