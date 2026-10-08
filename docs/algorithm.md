# Original local 8+1 algorithm

This document specifies the unfiltered, independent-beta `projected995` method. The package removes experimental alternatives while preserving the original numerical order in the message and correction kernels.

## Coordinates and GBP

At a frozen linearization, the normalized objective is

\[
E(x)=\tfrac12 x^\top A x-b^\top x,\qquad r=b-Ax.
\]

The physical right-Lie increment is \(\delta_i=S_ix_i\). Pose 0 is eliminated exactly; group 0 starts at the first **free** node, not at the fixed root. \(D_i=A_{ii}\) is the full node diagonal block.

For directed message \(e:s\to t\), let \(p_e\) denote its prepared precision and \(\eta_e\) its shifted information vector. Precision remains frozen during the inner mean solve. With reverse slot \(\bar e\), one synchronous mean sweep is

\[
\eta_e^+=c_e+G_e\!\left(\sum_{q:\operatorname{dst}(q)=s}\eta_q-\eta_{\bar e}\right),
\quad x_i=P_i^{-1}\left(u_i+\sum_{e:\operatorname{dst}(e)=i}\eta_e\right).
\]

Here \(P_i=U_i+\sum_{e\to i}p_e\). Factor linear terms are shifted into the node right-hand side, so \(\eta^{\rm canonical}=\eta+\text{offset}\) and **canonical-zero initialization is \(\eta=-\text{offset}\)**. It is not zero shifted eta. The nonlinear path prepares precision independently with damping 0.5 and tolerance 1e-10 over four consecutive passes; no mean sweeps are hidden in that preparation.

## Local modes and balanced correction

Groups are contiguous blocks of 32 free-node IDs. For each group, form the physical Neumann matrix \(N_g\) using physical unary terms and factors entirely inside the group; cut-edge factors are excluded from \(N_g\). Select its 4 lowest eigenvectors for SE2 or 12 for SE3, denoted \(\Phi_g\). Map them to normalized coordinates and orthonormalize:

\[
V_g=\operatorname{qr}(S_g^{-1}\Phi_g),\qquad \Pi_g=V_gV_g^\top.
\]

Use the **principal local Hessian** \(H_g=A_{gg}\), which retains the full node diagonals including cut-edge contributions, to set the correction amplitude. Define

\[
Q_g=V_g(V_g^\top H_gV_g)^{-1}V_g^\top,
\qquad M_g=\tfrac12\operatorname{blockdiag}(D_i^{-1})_{i\in g},
\]

\[
B_g=.45\left[Q_g+(I-Q_gH_g)M_g(I-H_gQ_g)\right].
\]

`preconditioner._step` applies this via small dense factors; it does not explicitly form a global B or solve a global system. Its node-diagonal term is \(.225D_i^{-1}\). The local mode basis and projected curvature are rebuilt at every new linearization.

For the audited PSD pairwise factor model, \(H_g\preceq2\operatorname{blockdiag}(D_i)\), hence \(M_g\preceq H_g^{-1}\) and \(B_g\preceq .45H_g^{-1}\). Also \(A\preceq2\operatorname{blockdiag}(H_g)\). Therefore, for assembled block-diagonal B,

\[
BAB\preceq .9B,
\quad E(x+Br)-E(x)\leq-.55r^\top Br.
\]

This bound covers the **spatial correction alone**, with exact supported injection. It does not establish energy monotonicity for inertia or subsequent GBP sweeps.

## Independent adaptive beta and eta injection

After eight mean sweeps, write \(x_k,r_k\) for the pre-correction checkpoint and \(x_{k-1}\) for the previous pre-correction checkpoint. Each group computes

\[
d_{k,g}=B_gr_{k,g}+\beta_{k,g}\Pi_g(x_{k,g}-x_{k-1,g}).
\]

The first checkpoint has no history. Each group starts with \(t_g=1,\beta_g=0\). After applying the current correction, it tests

\[
r_{k,g}^\top\big[(x_{k,g}-x_{k-1,g})+d_{k,g}\big]<0.
\]

A negative test resets **that group's next** clock and beta to 1 and 0. Otherwise,

\[
t_g^+=\frac{1+\sqrt{1+4t_g^2}}2,
\qquad \beta_g^+=\min\!\left(.995,\frac{t_g-1}{t_g^+}\right).
\]

This borrows the Nesterov/FISTA clock and a gradient-direction adaptive restart heuristic. The test happens after the correction and does not undo it. No shared beta, shared restart clock, local backtracking, or energy filter is added. Acceleration theory for a standard proximal iteration is not a convergence theorem for this full GBP message cycle.

Let \(\Sigma_i=\sum_{e\to i}p_e\). The incoming-message lift is

\[
L_e=p_e\Sigma_{\operatorname{dst}(e)}^\dagger P_{\operatorname{dst}(e)},
\qquad \eta_e\leftarrow\eta_e+L_ed_{\operatorname{dst}(e)}.
\]

This distributes the mean correction over **factor-supported incoming eta**, preserving precision. When \(\Sigma_i\) is full rank, the resulting mean change is \(d_i\). With deficient rank it is the supported projection; the implementation reports rank and lift error rather than claiming an unrestricted inverse. All archived k0 cases have full incoming rank under the recorded numerical threshold. Injection changes the actual canonical message state, and the next eight GBP sweeps respond to that state.

## Cycle accounting

- Frozen convergence replay observes the trajectory after each eight actual mean sweeps. It stops on the joint normal/message threshold or cap **before that row's correction**. Thus 1,768 sweeps correspond to 221 checkpoints and 220 corrections. At 8,192 sweeps there are 1,023 corrections.
- Nonlinear solves perform **20 complete IBP cycles** per linearization: 160 mean sweeps and 20 corrections, including correction 20 before retraction. The returned state is post-correction, although the trace's first residual columns remain pre-correction observations.
- IBP evaluates F once per actual mean sweep and once per checkpoint diagnostic. Pure GBP reuses that diagnostic proposal as its next accepted sweep. Frozen F counts are therefore \(9c\) for IBP and \(8c+1\) for GBP, for c completed checkpoints.
- A nonlinear solve adds one final returned-state diagnostic evaluation: 181 F calls for IBP and 162 for GBP per 20-cycle linearization. Precision sweeps are recorded separately from mean sweeps and F calls.

Trace columns are named in `solver.TRACE_COLUMNS`. Reserved columns retain the original 18-column layout and stay zero. Fresh runs also store each group's beta history; original archives retain their original mean/max-beta columns.

## Benchmark model and transport

The archived experiments use right SE exponential retraction (translation followed by rotation), the residual \(\log(Z_{ij}^{-1}T_i^{-1}T_j)\), Huber threshold 5, and exactly fixed pose 0. Initialization composes a deterministic measurement tree, preferring consecutive node IDs and reporting the nonconsecutive connecting edges. Supplied vertex estimates are not used by the archived runs.

`log_raw` deliberately pairs the original file information with the selected Lie-log coordinates. This is the recorded research objective, not a claim that quaternion-vector g2o information has been transformed to a rotation-vector metric. The FR079 TORO translation convention also differs from native g2o. Comparing numerical costs requires matching these choices. Cubicle additionally floors information eigenvalues at 1: 6,307 of its 16,869 factors change, with raw minimum eigenvalue approximately -6.2374e6. This is a substantial declared input repair, distinct from the tiny roundoff PSD projection during message warping. Neither model repair nor robust reweighting is an acceleration result.

At a new chart, let the previous increment satisfy \(x_{old}\approx q+Jx_{new}\), where

\[
a_i=\log(T_{old,i}^{-1}T_{new,i}),\quad
q_i=S_{old,i}^{-1}a_i,\quad
J_i=S_{old,i}^{-1}J_r(a_i)^{-1}S_{new,i}.
\]

For each incoming message,

\[
p_{new}=J^\top p_{old}J,
\quad \eta_{new}^{canonical}=J^\top(\eta_{old}^{canonical}-p_{old}q).
\]

Subtract the **new** factor offset to restore shifted eta. This is an exact Gaussian pullback for the affine chart model; the Lie-chart relation itself is a first-order approximation. For invertible precision, covariance transforms as \(C_{new}=J^{-1}C_{old}J^{-\top}\), not by leaving variance unchanged. The tests verify both potential and covariance identities. Tiny negative precision eigenvalues from roundoff are projected to PSD with the relative projection recorded, then precision is re-certified on the new factor model. Beta/history are reset and local bases rebuilt.

The common outer loop accepts a right-Lie step with Armijo coefficient 1e-4 and at most 30 halvings. This global nonlinear acceptance is distinct from a filter on the inner local correction. Saved-pose costs and accepted-step inequalities are independently verified.
